#!/usr/bin/env python3
"""
文档导入RAG工具 - 支持多种切片策略
用法: python import_docs_to_rag.py [策略] [限制]

策略:
  simple    - 简单切片（按行数，最快）
  header    - 按标题切片（推荐）
  semantic  - 语义切片（保留句子完整，较慢）
  sliding   - 滑动窗口（带重叠，最慢但最完整）

示例:
  python import_docs_to_rag.py header 100    # 用header策略，导入前100个文件
  python import_docs_to_rag.py simple        # 用simple策略，导入所有文件
"""

import sys
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入CPURAG
from plugins.memory.cpu_rag import CPURAGMemoryProvider


class DocumentChunker:
    """文档切片器"""
    
    def __init__(self, strategy="header", max_size=800, overlap=200):
        self.strategy = strategy
        self.max_size = max_size
        self.overlap = overlap
    
    def chunk(self, content, filepath):
        """根据策略切片"""
        if self.strategy == "simple":
            return self._simple_chunk(content, filepath)
        elif self.strategy == "header":
            return self._header_chunk(content, filepath)
        elif self.strategy == "semantic":
            return self._semantic_chunk(content, filepath)
        elif self.strategy == "sliding":
            return self._sliding_chunk(content, filepath)
        else:
            return self._header_chunk(content, filepath)
    
    def _simple_chunk(self, content, filepath):
        """简单切片：按固定行数"""
        lines = content.split('\n')
        chunks = []
        chunk_size = 50  # 每50行一块
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i+chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            if len(chunk_content) > 100:
                chunks.append({
                    "content": chunk_content[:1000],
                    "metadata": {"strategy": "simple", "lines": f"{i}-{i+len(chunk_lines)}"}
                })
        
        return chunks
    
    def _header_chunk(self, content, filepath):
        """按标题切片：按##分割"""
        # 按二级标题分割
        sections = re.split(r'\n## ', content)
        chunks = []
        
        for i, section in enumerate(sections):
            if len(section) < 100:
                continue
            
            # 处理标题
            if i == 0:
                header = "文档开头"
                body = section
            else:
                lines = section.split('\n', 1)
                header = lines[0].strip() if lines else f"章节{i}"
                body = lines[1] if len(lines) > 1 else ""
            
            # 如果内容太长，截断
            if len(body) > self.max_size:
                body = body[:self.max_size] + "..."
            
            chunks.append({
                "content": f"## {header}\n\n{body}",
                "metadata": {
                    "strategy": "header",
                    "section": i,
                    "header": header
                }
            })
        
        return chunks
    
    def _semantic_chunk(self, content, filepath):
        """语义切片：按句子分割"""
        # 按句子分割（中英文标点）
        sentences = re.split(r'(?<=[。！？.!?])\s+', content)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent_size = len(sent)
            
            if current_size + sent_size > self.max_size and current_chunk:
                # 保存当前块
                chunks.append({
                    "content": ''.join(current_chunk),
                    "metadata": {"strategy": "semantic", "sentences": len(current_chunk)}
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sent)
            current_size += sent_size
        
        # 最后一块
        if current_chunk:
            chunks.append({
                "content": ''.join(current_chunk),
                "metadata": {"strategy": "semantic", "sentences": len(current_chunk)}
            })
        
        return chunks
    
    def _sliding_chunk(self, content, filepath):
        """滑动窗口：带重叠"""
        lines = content.split('\n')
        chunks = []
        step = self.max_size - self.overlap
        
        for i in range(0, len(lines), step):
            chunk_lines = lines[i:i+self.max_size]
            if len('\n'.join(chunk_lines)) > 100:
                chunks.append({
                    "content": '\n'.join(chunk_lines),
                    "metadata": {
                        "strategy": "sliding",
                        "window": f"{i}-{i+len(chunk_lines)}",
                        "overlap": self.overlap
                    }
                })
        
        return chunks


def import_documents(strategy="header", max_files=None, memory_dir=None):
    """导入文档"""
    if memory_dir is None:
        memory_dir = Path("/root/.hermes/openclaw_memories_from_server")
    else:
        memory_dir = Path(memory_dir)
    
    # 初始化
    print(f"\n{'='*60}")
    print(f"🚀 文档导入RAG - 策略: {strategy}")
    print(f"{'='*60}\n")
    
    provider = CPURAGMemoryProvider()
    provider.initialize(session_id=f"import-{strategy}-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    
    # 获取文件列表
    files = list(memory_dir.rglob("*.md"))
    if max_files:
        files = files[:max_files]
    
    print(f"📂 找到 {len(files)} 个 Markdown 文件")
    print(f"📁 目录: {memory_dir}")
    print(f"🔧 切片策略: {strategy}")
    print(f"⏱️ 开始导入...\n")
    
    # 初始化切片器
    chunker = DocumentChunker(strategy=strategy)
    
    # 统计
    stats = {
        "files_processed": 0,
        "files_success": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "chunks_added": 0,
        "errors": []
    }
    
    start_time = time.time()
    
    # 处理每个文件
    for i, filepath in enumerate(files, 1):
        try:
            # 读取文件
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if len(content) < 100:
                continue
            
            # 切片
            chunks = chunker.chunk(content, filepath)
            stats["chunks_created"] += len(chunks)
            
            # 导入每个块
            file_chunks_added = 0
            for chunk in chunks:
                result = provider.handle_tool_call("rag_add_memory", {
                    "content": chunk["content"],
                    "source": str(filepath.relative_to(memory_dir)),
                    "metadata": chunk["metadata"]
                })
                
                result_data = json.loads(result)
                if result_data.get("success"):
                    stats["chunks_added"] += 1
                    file_chunks_added += 1
            
            stats["files_success"] += 1
            
            # 进度显示
            if i % 10 == 0 or i == len(files):
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                print(f"  进度: {i}/{len(files)} | 块: {stats['chunks_added']} | 速度: {speed:.1f}文件/秒")
            
        except Exception as e:
            stats["files_failed"] += 1
            stats["errors"].append(f"{filepath.name}: {str(e)}")
        
        stats["files_processed"] += 1
    
    # 最终统计
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✅ 导入完成!")
    print(f"{'='*60}")
    print(f"  处理文件: {stats['files_processed']}")
    print(f"  成功文件: {stats['files_success']}")
    print(f"  失败文件: {stats['files_failed']}")
    print(f"  创建块数: {stats['chunks_created']}")
    print(f"  成功导入: {stats['chunks_added']}")
    print(f"  用时: {elapsed:.1f}秒")
    print(f"  平均速度: {stats['files_success']/elapsed:.1f} 文件/秒")
    
    # 查询最终统计
    result = provider.handle_tool_call("rag_stats", {})
    final_stats = json.loads(result)
    print(f"  数据库总记忆: {final_stats.get('total_records', 0)}")
    
    if stats["errors"]:
        print(f"\n⚠️ 错误 ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"    - {err}")
    
    print(f"\n💡 测试检索:")
    test_queries = ["AI视频", "Seedance", "Divorce Relationship"]
    for query in test_queries:
        result = provider.handle_tool_call("rag_search", {
            "query": query,
            "top_k": 1
        })
        data = json.loads(result)
        count = data.get('count', 0)
        print(f"  '{query}': {count} 条结果")


if __name__ == "__main__":
    # 解析参数
    strategy = sys.argv[1] if len(sys.argv) > 1 else "header"
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # 验证策略
    valid_strategies = ["simple", "header", "semantic", "sliding"]
    if strategy not in valid_strategies:
        print(f"❌ 无效策略: {strategy}")
        print(f"可用策略: {', '.join(valid_strategies)}")
        sys.exit(1)
    
    # 执行导入
    import_documents(strategy=strategy, max_files=max_files)
