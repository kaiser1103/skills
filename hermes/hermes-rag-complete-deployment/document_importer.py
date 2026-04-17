#!/usr/bin/env python3
"""
智能文档导入器 - 将 Markdown 文档导入向量数据库

支持多种分块策略:
1. 按标题层级分割
2. 滑动窗口重叠
3. 语义分块（不切断句子）
4. 保留元数据
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class DocumentChunker:
    """智能文档分块器"""
    
    def __init__(
        self,
        max_chunk_size: int = 800,
        overlap_size: int = 200,
        preserve_sentences: bool = True
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.preserve_sentences = preserve_sentences
    
    def chunk_document(self, content: str, filepath: str) -> List[Dict]:
        """
        主分块函数
        返回: [{"content": str, "header": str, "position": int, "metadata": {}}]
        """
        chunks = []
        
        # 第一步：按一级标题分割
        sections = self._split_by_headers(content, level=1)
        
        for section_idx, (header, section_content) in enumerate(sections):
            # 第二步：处理每个章节
            section_chunks = self._process_section(
                header=header,
                content=section_content,
                section_idx=section_idx,
                filepath=filepath
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_by_headers(self, content: str, level: int = 1) -> List[Tuple[str, str]]:
        """按标题分割内容"""
        # 匹配指定级别的标题 (# 或 ## 等)
        pattern = rf'^(#{{{level}}}\s+.+)$'
        
        lines = content.split('\n')
        sections = []
        current_header = "无标题"
        current_content = []
        
        for line in lines:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # 保存之前的章节
                if current_content:
                    sections.append((
                        current_header,
                        '\n'.join(current_content).strip()
                    ))
                # 开始新章节
                current_header = match.group(1).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 添加最后一个章节
        if current_content:
            sections.append((
                current_header,
                '\n'.join(current_content).strip()
            ))
        
        return sections if sections else [("全文", content)]
    
    def _process_section(
        self,
        header: str,
        content: str,
        section_idx: int,
        filepath: str
    ) -> List[Dict]:
        """处理单个章节"""
        chunks = []
        
        # 如果章节太短，直接作为一个块
        if len(content) <= self.max_chunk_size:
            return [{
                "content": f"{header}\n\n{content}".strip(),
                "header": header,
                "position": section_idx,
                "metadata": {
                    "type": "section",
                    "filepath": filepath,
                    "chunk_index": 0,
                    "total_chunks": 1
                }
            }]
        
        # 按二级标题进一步分割
        sub_sections = self._split_by_headers(content, level=2)
        
        for sub_idx, (sub_header, sub_content) in enumerate(sub_sections):
            # 合并标题
            full_header = f"{header} > {sub_header}" if sub_header != "无标题" else header
            
            # 如果子章节仍然很长，使用滑动窗口
            if len(sub_content) > self.max_chunk_size:
                window_chunks = self._sliding_window_chunk(
                    content=sub_content,
                    header=full_header,
                    filepath=filepath,
                    section_idx=section_idx,
                    sub_idx=sub_idx
                )
                chunks.extend(window_chunks)
            else:
                chunks.append({
                    "content": f"{full_header}\n\n{sub_content}".strip(),
                    "header": full_header,
                    "position": section_idx,
                    "metadata": {
                        "type": "subsection",
                        "filepath": filepath,
                        "chunk_index": sub_idx,
                        "total_chunks": len(sub_sections)
                    }
                })
        
        return chunks
    
    def _sliding_window_chunk(
        self,
        content: str,
        header: str,
        filepath: str,
        section_idx: int,
        sub_idx: int
    ) -> List[Dict]:
        """滑动窗口分块"""
        chunks = []
        
        # 按句子分割
        sentences = self._split_to_sentences(content)
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # 如果添加这句子会超出限制
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # 保存当前块
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    "content": f"{header}\n\n{chunk_content}".strip(),
                    "header": header,
                    "position": section_idx,
                    "metadata": {
                        "type": "chunk",
                        "filepath": filepath,
                        "chunk_index": chunk_index,
                        "total_chunks": None,  # 之后填充
                        "sentence_range": f"{i-len(current_chunk)}-{i-1}"
                    }
                })
                
                # 保留重叠内容
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 处理最后一个块
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                "content": f"{header}\n\n{chunk_content}".strip(),
                "header": header,
                "position": section_idx,
                "metadata": {
                    "type": "chunk",
                    "filepath": filepath,
                    "chunk_index": chunk_index,
                    "total_chunks": chunk_index + 1,
                    "sentence_range": f"{len(sentences)-len(current_chunk)}-{len(sentences)-1}"
                }
            })
        
        # 更新 total_chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
    
    def _split_to_sentences(self, text: str) -> List[str]:
        """按句子分割（保留标点）"""
        # 支持中英文句子结束符
        pattern = r'(?<=[。！？；]|\.|\!|\?|\;)\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """获取需要重叠的句子"""
        overlap_text = ''
        overlap_sentences = []
        
        # 从后往前取句子，直到达到重叠大小
        for sent in reversed(sentences):
            if len(overlap_text) + len(sent) <= self.overlap_size:
                overlap_sentences.insert(0, sent)
                overlap_text += sent
            else:
                break
        
        return overlap_sentences


def import_documents_to_rag(
    directory: str,
    provider,
    file_pattern: str = "*.md",
    max_file_size: int = 10 * 1024 * 1024  # 10MB
) -> Dict:
    """
    将目录中的文档导入到 RAG
    
    Args:
        directory: 文档目录
        provider: CPURAGMemoryProvider 实例
        file_pattern: 文件匹配模式
        max_file_size: 最大文件大小
    
    Returns:
        {
            "total_files": int,
            "success_files": int,
            "failed_files": int,
            "total_chunks": int,
            "errors": List[str]
        }
    """
    directory = Path(directory)
    chunker = DocumentChunker(
        max_chunk_size=800,
        overlap_size=200,
        preserve_sentences=True
    )
    
    stats = {
        "total_files": 0,
        "success_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "errors": []
    }
    
    # 遍历所有文件
    files = list(directory.rglob(file_pattern))
    stats["total_files"] = len(files)
    
    print(f"\n📂 找到 {len(files)} 个 {file_pattern} 文件")
    print("开始导入...\n")
    
    for i, filepath in enumerate(files, 1):
        try:
            # 检查文件大小
            if filepath.stat().st_size > max_file_size:
                stats["errors"].append(f"{filepath}: 文件超过大小限制")
                stats["failed_files"] += 1
                continue
            
            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            # 分块
            chunks = chunker.chunk_document(content, str(filepath))
            
            # 导入每个块
            for chunk in chunks:
                result = provider.handle_tool_call("rag_add_memory", {
                    "content": chunk["content"],
                    "source": str(filepath.relative_to(directory)),
                    "metadata": chunk["metadata"]
                })
                
                result_data = json.loads(result)
                if result_data.get("success"):
                    stats["total_chunks"] += 1
            
            stats["success_files"] += 1
            
            if i % 10 == 0 or i == len(files):
                print(f"  进度: {i}/{len(files)} 文件, {stats['total_chunks']} 块")
                
        except Exception as e:
            stats["failed_files"] += 1
            stats["errors"].append(f"{filepath}: {str(e)}")
            print(f"  ❌ 失败: {filepath.name} - {e}")
    
    return stats


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, '/root/.hermes/hermes-agent')
    
    from plugins.memory.cpu_rag import CPURAGMemoryProvider
    
    # 初始化 provider
    provider = CPURAGMemoryProvider()
    provider.initialize(session_id="document-import")
    
    # 测试单个文件
    test_content = """
# 测试文档

这是一个测试文档。用于验证分块算法是否正常工作。

## 第一节

这是第一节的内容。包含了一些重要信息。
这里有很多文字。用于测试分块逻辑。

## 第二节

这是第二节的内容。
同样包含了很多信息。
"""
    
    chunker = DocumentChunker(max_chunk_size=100, overlap_size=30)
    chunks = chunker.chunk_document(test_content, "test.md")
    
    print(f"测试分块结果: {len(chunks)} 块")
    for i, chunk in enumerate(chunks):
        print(f"\n块 {i+1}:")
        print(f"  标题: {chunk['header']}")
        print(f"  长度: {len(chunk['content'])}")
        print(f"  内容预览: {chunk['content'][:50]}...")
