#!/usr/bin/env python3.12
"""
ONNX Runtime加速版 - 简化实现
用法: python import_docs_onnx_simple.py [限制文件数]
"""

import sys
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lancedb
import pyarrow as pa
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import onnxruntime as ort


class ONNXImporter:
    def __init__(self):
        model_path = os.path.expanduser("~/.hermes/models/embeddings/bge-small-zh-v1.5")
        
        print("🚀 初始化模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载PyTorch模型
        print("⚡ 加载PyTorch模型用于导出...")
        self.torch_model = AutoModel.from_pretrained(model_path)
        self.torch_model.eval()
        
        # 直接导出ONNX（简化版）
        onnx_path = "/tmp/bge_small_zh.onnx"
        if not os.path.exists(onnx_path):
            print("📦 导出ONNX模型...")
            dummy_input = torch.zeros(1, 128, dtype=torch.long)
            dummy_mask = torch.ones(1, 128, dtype=torch.long)
            # PyTorch 2.x dynamo模式：dynamic_shapes 只包含输入参数名，
            # 输出不要放进去；attention_mask 通过 kwargs 传入
            from torch.export import Dim
            dynamic_shapes = {
                'input_ids': {0: Dim('batch_size'), 1: Dim('seq_len')},
                'attention_mask': {0: Dim('batch_size'), 1: Dim('seq_len')},
            }
            torch.onnx.export(
                self.torch_model,
                (dummy_input,),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamo=True,
                kwargs={'attention_mask': dummy_mask},
                dynamic_shapes=dynamic_shapes,
                opset_version=11
            )
            print(f"✓ ONNX模型已保存: {onnx_path}")
        
        # 创建ONNX Runtime会话
        print("⚡ 初始化ONNX Runtime...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        print("✓ ONNX Runtime就绪")
        
        # 连接数据库
        db_path = os.path.expanduser("~/.hermes/rag_vector_db")
        print(f"📁 连接数据库: {db_path}")
        self.db = lancedb.connect(db_path)
        
        existing = self.db.list_tables()
        # list_tables() 返回 ListTablesResponse 对象，不是普通 list
        tables = existing.tables if hasattr(existing, 'tables') else list(existing)
        if "memories" in tables:
            self.table = self.db.open_table("memories")
        else:
            schema = pa.schema([
                ("id", pa.string()),
                ("content", pa.string()),
                ("vector", pa.list_(pa.float32(), 512)),
                ("source", pa.string()),
                ("metadata", pa.string()),
                ("timestamp", pa.string()),
                ("access_count", pa.int32()),
            ])
            self.table = self.db.create_table("memories", schema=schema)
        
        print("✅ 初始化完成\n")
    
    def encode(self, texts, batch_size=64):
        """ONNX编码（逐条推理，避免batch维度固定问题）"""
        all_embeddings = []
        
        # 先批量 tokenize（高效）
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # 逐条推理（ONNX模型batch维度固定为1）
        for i in range(len(texts)):
            ort_inputs = {
                'input_ids': input_ids[i:i+1].astype(np.int64),
                'attention_mask': attention_mask[i:i+1].astype(np.int64)
            }
            
            outputs = self.session.run(None, ort_inputs)
            last_hidden = outputs[0]  # [1, seq_len, hidden]
            
            # Mean pooling
            mask = attention_mask[i].astype(np.float32)
            mask_expanded = np.expand_dims(mask, axis=-1)
            mask_expanded = np.broadcast_to(mask_expanded, last_hidden.shape)
            
            sum_emb = np.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), 1e-9, None)
            embedding = sum_emb / sum_mask  # [1, hidden]
            
            all_embeddings.append(embedding[0])
        
        return np.array(all_embeddings)
    
    def chunk(self, content, filepath):
        """按标题切片"""
        sections = re.split(r'\n## ', content)
        chunks = []
        
        for i, section in enumerate(sections):
            if len(section) < 100:
                continue
            
            if i == 0:
                header = "文档开头"
                body = section
            else:
                lines = section.split('\n', 1)
                header = lines[0].strip()[:50] if lines else f"章节{i}"
                body = lines[1] if len(lines) > 1 else ""
            
            if len(body) > 800:
                body = body[:800] + "..."
            
            chunks.append({
                "content": f"## {header}\n\n{body}",
                "source": str(filepath),
                "metadata": json.dumps({"section": i, "header": header})
            })
        
        return chunks
    
    def import_docs(self, memory_dir, max_files=None):
        """导入文档"""
        memory_dir = Path(memory_dir)
        files = list(memory_dir.rglob("*.md"))
        if max_files:
            files = files[:max_files]
        
        print(f"📂 找到 {len(files)} 个文件\n")
        
        # 收集所有chunks
        all_chunks = []
        t0 = time.time()
        
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                if len(content) >= 100:
                    chunks = self.chunk(content, f.relative_to(memory_dir))
                    all_chunks.extend(chunks)
            except:
                pass
        
        t1 = time.time()
        print(f"✓ 收集完成: {len(all_chunks)} 块 ({t1-t0:.1f}秒)")
        
        if not all_chunks:
            print("⚠️ 没有有效内容")
            return 0
        
        # ONNX批量编码
        print(f"\n⚡ ONNX编码 {len(all_chunks)} 个chunks...")
        t0 = time.time()
        texts = [c["content"] for c in all_chunks]
        embeddings = self.encode(texts, batch_size=64)
        t1 = time.time()
        encode_time = t1 - t0
        print(f"✓ 编码完成: {embeddings.shape} ({encode_time:.1f}秒)")
        print(f"  速度: {len(all_chunks)/encode_time:.1f} chunks/秒")
        
        # 批量写入
        print(f"\n💾 写入数据库...")
        t0 = time.time()
        
        from uuid import uuid4
        records = []
        for i, chunk in enumerate(all_chunks):
            records.append({
                "id": str(uuid4()),
                "content": chunk["content"],
                "vector": embeddings[i].tolist(),
                "source": chunk["source"],
                "metadata": chunk["metadata"],
                "timestamp": datetime.now().isoformat(),
                "access_count": 0
            })
        
        self.table.add(records)
        t1 = time.time()
        write_time = t1 - t0
        print(f"✓ 写入完成: {len(records)} 条 ({write_time:.1f}秒)")
        
        # 统计
        total = (time.time() - t0) + encode_time + write_time
        print(f"\n{'='*60}")
        print("📊 ONNX加速导入统计")
        print(f"{'='*60}")
        print(f"  文件数: {len(files)}")
        print(f"  Chunks: {len(all_chunks)}")
        print(f"  总用时: {total:.1f}秒")
        print(f"  速度: {len(files)/total:.1f} 文件/秒")
        
        return len(records)


if __name__ == "__main__":
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    print(f"{'='*60}")
    print("🚀 ONNX Runtime简化版 - 文档导入")
    print(f"{'='*60}\n")
    
    importer = ONNXImporter()
    count = importer.import_docs("/root/.hermes/openclaw_memories_from_server", max_files)
    print(f"\n✅ 完成! 导入 {count} 条记忆")
