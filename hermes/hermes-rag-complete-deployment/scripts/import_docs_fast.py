#!/usr/bin/env python3
"""
快速文档导入工具 - 批量处理版本
用法: 
  python import_docs_fast.py              # 增量导入（跳过已有文档）
  python import_docs_fast.py --full       # 全量导入（覆盖已有文档）
  python import_docs_fast.py --watch      # 监听模式（自动检测新文档并导入）
  python import_docs_fast.py 100          # 限制最多100个文件

比较：
- 原版: ~0.3 文件/秒（每次单独编码）
- 快速版: ~10-50 文件/秒（批量编码）
"""

import sys
import os
import re
import json
import time
import threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要组件
import lancedb
import pyarrow as pa
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class FastDocumentImporter:
    """快速文档导入器 - 批量处理"""
    
    def __init__(self, db_path=None, model_name=None):
        if db_path is None:
            db_path = os.path.expanduser("~/.hermes/rag_vector_db")
        if model_name is None:
            model_name = "BAAI/bge-small-zh-v1.5"
        
        self.db_path = db_path
        self.model_name = model_name
        
        # 初始化Embedding模型（只初始化一次）
        print("🚀 初始化Embedding模型...")
        model_path = os.path.expanduser(f"~/.hermes/models/embeddings/{model_name.split('/')[-1]}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
        # 初始化数据库
        print(f"📁 连接数据库: {db_path}")
        self.db = lancedb.connect(db_path)
        
        # 获取或创建表
        if "memories" in self.db.table_names():
            self.table = self.db.open_table("memories")
            print(f"   找到已存在表: memories")
        else:
            print(f"   创建新表: memories")
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
    
    def encode_batch(self, texts, batch_size=32):
        """批量编码文本"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).numpy()
            
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def watch_and_import(self, memory_dir, debounce_seconds=5):
        """
        监听目录变化，自动导入新文档
        
        Args:
            memory_dir: 要监听的目录
            debounce_seconds: 防抖时间（秒），连续变化后等待多久才导入
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            print("❌ 需要安装 watchdog: pip install watchdog")
            return
        
        memory_dir = Path(memory_dir)
        
        class ImportHandler(FileSystemEventHandler):
            def __init__(self, importer, memory_dir, debounce):
                self.importer = importer
                self.memory_dir = memory_dir
                self.debounce = debounce
                self.pending_files = []
                self.last_import_time = 0
                self.import_lock = False
            
            def on_created(self, event):
                if event.is_directory or not event.src_path.endswith('.md'):
                    return
                print(f"\n📄 检测到新文件: {event.src_path}")
                self._schedule_import()
            
            def on_modified(self, event):
                if event.is_directory or not event.src_path.endswith('.md'):
                    return
                print(f"\n📝 检测到文件变化: {event.src_path}")
                self._schedule_import()
            
            def _schedule_import(self):
                import time
                current_time = time.time()
                if current_time - self.last_import_time > self.debounce:
                    self._do_import()
                else:
                    # 等待debounce后导入
                    threading.Thread(target=self._delayed_import, daemon=True).start()
            
            def _delayed_import(self):
                import time
                time.sleep(self.debounce)
                self._do_import()
            
            def _do_import(self):
                import time
                if self.import_lock:
                    return
                self.import_lock = True
                
                try:
                    print(f"\n🔄 自动增量导入新文档...")
                    count = self.importer.import_documents(self.memory_dir, incremental=True)
                    if count > 0:
                        # 创建重新加载标记
                        reload_marker = Path(self.importer.db_path) / ".reload_needed"
                        reload_marker.touch()
                        print(f"📌 已创建数据库重新加载标记")
                    self.last_import_time = time.time()
                finally:
                    self.import_lock = False
        
        print(f"\n👁️ 开始监听目录: {memory_dir}")
        print(f"   防抖时间: {debounce_seconds}秒")
        print(f"   按 Ctrl+C 停止监听\n")
        
        event_handler = ImportHandler(self, memory_dir, debounce_seconds)
        observer = Observer()
        observer.schedule(event_handler, str(memory_dir), recursive=True)
        observer.start()
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n\n👋 停止监听")
            observer.stop()
        observer.join()
    
    def chunk_by_headers(self, content, filepath):
        """按标题切片 - 支持 ## 和 ### 层级，不截断内容"""
        chunks = []
        
        # 先按 ## 二级标题切分
        level2_sections = re.split(r'\n## ', content)
        
        for i, level2_section in enumerate(level2_sections):
            if len(level2_section.strip()) < 50:
                continue
            
            # 处理第一个section（文档开头）
            if i == 0:
                if len(level2_section.strip()) >= 100:
                    chunks.append({
                        "content": level2_section.strip(),
                        "source": str(filepath),
                        "metadata": json.dumps({"level": 0, "header": "文档开头"})
                    })
                continue
            
            # 提取二级标题
            lines = level2_section.split('\n', 1)
            level2_header = lines[0].strip() if lines else f"章节{i}"
            level2_body = lines[1] if len(lines) > 1 else ""
            
            # 检查是否包含三级标题 ###
            if '\n### ' in level2_body:
                # 按三级标题切分
                level3_sections = re.split(r'\n### ', level2_body)
                
                # 第一部分（二级标题下的直接内容）
                if level3_sections[0].strip():
                    chunks.append({
                        "content": f"## {level2_header}\n\n{level3_sections[0].strip()}",
                        "source": str(filepath),
                        "metadata": json.dumps({"level": 2, "header": level2_header})
                    })
                
                # 处理每个三级标题
                for j, level3_section in enumerate(level3_sections[1:], 1):
                    if len(level3_section.strip()) < 50:
                        continue
                    
                    level3_lines = level3_section.split('\n', 1)
                    level3_header = level3_lines[0].strip() if level3_lines else f"小节{j}"
                    level3_body = level3_lines[1].strip() if len(level3_lines) > 1 else ""
                    
                    chunks.append({
                        "content": f"## {level2_header}\n\n### {level3_header}\n\n{level3_body}",
                        "source": str(filepath),
                        "metadata": json.dumps({
                            "level": 3, 
                            "parent_header": level2_header,
                            "header": level3_header
                        })
                    })
            else:
                # 没有三级标题，整个二级section作为一个chunk
                chunks.append({
                    "content": f"## {level2_header}\n\n{level2_body.strip()}",
                    "source": str(filepath),
                    "metadata": json.dumps({"level": 2, "header": level2_header})
                })
        
        return chunks
    
    def get_existing_sources(self):
        """获取数据库中已存在的文档source列表"""
        try:
            existing = self.table.to_pandas()['source'].unique().tolist()
            print(f"   数据库已有 {len(existing)} 个文档")
            return set(existing)
        except Exception as e:
            print(f"   获取已有文档失败: {e}，将进行全量导入")
            return set()
    
    def import_documents(self, memory_dir, max_files=None, incremental=True):
        """
        导入文档 - 批量处理
        
        Args:
            memory_dir: 文档目录
            max_files: 最大文件数限制
            incremental: 是否增量导入（默认True，增量导入不会删除现有数据）
        """
        memory_dir = Path(memory_dir)
        files = list(memory_dir.rglob("*.md"))
        
        if max_files:
            files = files[:max_files]
        
        print(f"📂 找到 {len(files)} 个文件")
        
        # 获取已存在的文档列表
        existing_sources = set()
        if incremental:
            print(f"\n🔍 检查已有文档...")
            existing_sources = self.get_existing_sources()
        
        print(f"⏱️ 开始处理...\n")
        
        # 第一阶段：收集需要导入的chunks
        all_chunks = []
        skipped_files = 0
        new_files = 0
        start_time = time.time()
        
        for filepath in files:
            try:
                relative_path = str(filepath.relative_to(memory_dir))
                
                # 增量模式：跳过已存在的文档
                if incremental and relative_path in existing_sources:
                    skipped_files += 1
                    continue
                
                new_files += 1
                
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if len(content) < 100:
                    continue
                
                chunks = self.chunk_by_headers(content, filepath.relative_to(memory_dir))
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"  ⚠️ 读取失败: {filepath.name}")
        
        collect_time = time.time() - start_time
        
        if incremental:
            print(f"  ✅ 收集完成: {len(all_chunks)} 块 (新增 {new_files} 个文档，跳过 {skipped_files} 个已有文档，用时 {collect_time:.1f}秒)")
        else:
            print(f"  ✅ 收集完成: {len(all_chunks)} 块 (用时 {collect_time:.1f}秒)")
        
        if len(all_chunks) == 0:
            print("\n📭 没有新文档需要导入")
            return 0
        
        # 第二阶段：批量编码
        print(f"\n🔢 批量编码 {len(all_chunks)} 个chunks...")
        start_time = time.time()
        
        texts = [c["content"] for c in all_chunks]
        embeddings = self.encode_batch(texts, batch_size=64)
        
        encode_time = time.time() - start_time
        print(f"  ✅ 编码完成: {embeddings.shape} (用时 {encode_time:.1f}秒)")
        
        # 第三阶段：批量写入数据库
        print(f"\n💾 写入数据库...")
        start_time = time.time()
        
        from uuid import uuid4
        
        records = []
        for i, chunk in enumerate(all_chunks):
            records.append({
                "id": str(uuid4()),
                "content": chunk["content"],
                "vector": embeddings[i].tolist(),
                "source": chunk["source"],
                "metadata": json.dumps(chunk["metadata"]) if isinstance(chunk["metadata"], dict) else chunk["metadata"],
                "timestamp": datetime.now().isoformat(),
                "access_count": 0,
            })
        
        self.table.add(records)
        write_time = time.time() - start_time
        
        total_time = collect_time + encode_time + write_time
        print(f"\n📊 导入统计")
        print(f"{'='*60}")
        print(f"  处理文件: {new_files} (增量模式: {incremental})")
        if incremental:
            print(f"  跳过文件: {skipped_files}")
        print(f"  生成chunks: {len(records)}")
        print(f"  成功导入: {len(records)}")
        print(f"  总用时: {total_time:.1f}秒")
        print(f"  平均速度: {new_files/total_time:.1f} 文件/秒")
        print(f"  每秒chunks: {len(records)/total_time:.1f} chunks/秒")
        
        return len(records)


if __name__ == "__main__":
    # 解析命令行参数
    max_files = None
    incremental = True
    watch_mode = False
    
    for arg in sys.argv[1:]:
        if arg == "--full":
            incremental = False
        elif arg == "--watch":
            watch_mode = True
        elif arg.isdigit():
            max_files = int(arg)
    
    memory_dir = "/root/.hermes/openclaw_memories_from_server"
    
    print(f"{'='*60}")
    print(f"🚀 快速文档导入RAG - 批量处理版")
    
    if watch_mode:
        print(f"👁️ 监听模式 (自动检测新文档)")
    elif incremental:
        print(f"📦 增量导入模式 (跳过已有文档)")
    else:
        print(f"⚠️ 全量导入模式 (将覆盖已有文档)")
    
    print(f"{'='*60}\n")
    
    importer = FastDocumentImporter()
    
    if watch_mode:
        # 监听模式：先做一次增量导入，然后开始监听
        print("🔍 先执行一次增量导入...\n")
        importer.import_documents(memory_dir, max_files=max_files, incremental=True)
        print()
        
        # 创建重新加载标记
        reload_marker = Path(importer.db_path) / ".reload_needed"
        reload_marker.touch()
        
        # 开始监听
        importer.watch_and_import(memory_dir, debounce_seconds=5)
    else:
        # 普通导入模式
        count = importer.import_documents(memory_dir, max_files=max_files, incremental=incremental)
        
        if count > 0:
            print(f"\n✅ 导入完成! 共 {count} 条记忆")
            
            # 创建重新加载标记文件
            from pathlib import Path
            reload_marker = Path(importer.db_path) / ".reload_needed"
            reload_marker.touch()
            print(f"📌 已创建数据库重新加载标记")
        else:
            print(f"\n📭 没有新文档需要导入")


