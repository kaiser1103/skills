#!/usr/bin/env python3
"""
RAG 文档增量导入脚本 v2.0
- 支持文档哈希校验（检测内容变化）
- 增量导入（只导入新文档或修改过的文档）
- 禁止清空现有数据库
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from datetime import datetime

# 添加 Hermes 路径
hermes_path = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(hermes_path))


def compute_file_hash(filepath: Path) -> str:
    """计算文件的 MD5 哈希值"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def get_existing_sources_with_hash(db_path: str) -> dict:
    """
    获取已有文档的 source 和 hash 映射
    返回: {source: hash}
    """
    try:
        import lancedb
        db = lancedb.connect(db_path)
        table = db.open_table("memories")
        
        # 查询所有记录的 source 和 hash
        results = table.to_pandas()
        
        source_hash_map = {}
        for _, row in results.iterrows():
            metadata = row.get('metadata', '{}')
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            source = metadata.get('source', '')
            file_hash = metadata.get('file_hash', '')
            
            if source and file_hash:
                # 只保留最新的哈希（如果同一个 source 有多条记录）
                source_hash_map[source] = file_hash
        
        return source_hash_map
    
    except Exception as e:
        print(f"⚠️  获取已有文档列表失败: {e}")
        return {}


def should_import_file(filepath: Path, existing_sources: dict) -> tuple:
    """
    判断文件是否需要导入
    返回: (是否需要导入, 原因)
    """
    source = str(filepath)
    current_hash = compute_file_hash(filepath)
    
    if source not in existing_sources:
        return True, "新文档"
    
    existing_hash = existing_sources[source]
    if current_hash != existing_hash:
        return True, f"内容已变化 (旧: {existing_hash[:8]}... 新: {current_hash[:8]}...)"
    
    return False, "未变化"


def delete_old_records(db_path: str, source: str):
    """删除指定 source 的旧记录"""
    try:
        import lancedb
        db = lancedb.connect(db_path)
        table = db.open_table("memories")
        
        # LanceDB 使用 delete 方法
        # 需要先查询出所有匹配的记录，然后删除
        df = table.to_pandas()
        to_delete = []
        for idx, row in df.iterrows():
            metadata = row.get('metadata', '{}')
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if metadata.get('source') == source:
                to_delete.append(row['id'])
        
        if to_delete:
            # LanceDB 使用 SQL 语法删除
            for record_id in to_delete:
                table.delete(f"id = '{record_id}'")
            print(f"   🗑️  已删除 {len(to_delete)} 条旧记录")
    
    except Exception as e:
        print(f"   ⚠️  删除旧记录失败: {e}")


def split_document(content: str, max_chunk_size: int = 800) -> list:
    """
    按标题切分文档
    支持 ## 和 ### 两级标题
    """
    import re
    
    chunks = []
    
    # 按 ## 标题切分
    sections = re.split(r'\n(##\s+.+)', content)
    
    current_chunk = ""
    for i, section in enumerate(sections):
        if i == 0:
            # 文档开头部分
            if section.strip():
                current_chunk = section.strip()
        elif section.startswith('##'):
            # 保存上一个 chunk
            if current_chunk:
                chunks.append(current_chunk)
            # 开始新的 chunk
            current_chunk = section
        else:
            # 内容部分
            current_chunk += "\n" + section
    
    # 保存最后一个 chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def import_documents(docs_dir: str, db_path: str, model_name: str, incremental: bool = True):
    """
    导入文档到 RAG 向量数据库
    
    Args:
        docs_dir: 文档目录路径
        db_path: 向量数据库路径
        model_name: Embedding 模型名称
        incremental: 是否增量导入
    """
    docs_path = Path(docs_dir).expanduser()
    
    if not docs_path.exists():
        print(f"❌ 文档目录不存在: {docs_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"📚 RAG 文档导入 - {'增量模式' if incremental else '全量模式'}")
    print(f"{'='*60}")
    print(f"📂 文档目录: {docs_path}")
    print(f"💾 数据库路径: {db_path}")
    print(f"🤖 模型: {model_name}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 初始化 Embedding Service 和 Vector Store
    try:
        from plugins.memory.cpu_rag import _EmbeddingService, _VectorStore
        
        hermes_home = str(Path.home() / ".hermes")
        model_path = f"{hermes_home}/models/embeddings/{model_name.split('/')[-1]}"
        
        print("🔧 正在初始化 Embedding Service...")
        embedding_service = _EmbeddingService(
            model_path=model_path,
            max_memory_mb=512
        )
        
        print("🔧 正在初始化 Vector Store...")
        vector_store = _VectorStore(db_path=db_path)
        
        print("✅ 初始化完成\n")
    
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取已有文档列表（带哈希）
    existing_sources = {}
    if incremental:
        print("🔍 正在检查已有文档...")
        existing_sources = get_existing_sources_with_hash(db_path)
        print(f"   已有文档数: {len(existing_sources)}\n")
    
    # 扫描所有 Markdown 文件
    md_files = list(docs_path.rglob("*.md"))
    print(f"📄 发现 {len(md_files)} 个 Markdown 文件\n")
    
    if not md_files:
        print("⚠️  没有找到任何 Markdown 文件")
        return
    
    # 统计
    imported_count = 0
    skipped_count = 0
    updated_count = 0
    failed_count = 0
    
    for idx, filepath in enumerate(md_files, 1):
        relative_path = filepath.relative_to(docs_path)
        print(f"[{idx}/{len(md_files)}] {relative_path}")
        
        try:
            # 检查是否需要导入
            if incremental:
                should_import, reason = should_import_file(filepath, existing_sources)
                
                if not should_import:
                    print(f"   ⏭️  跳过: {reason}\n")
                    skipped_count += 1
                    continue
                
                # 如果是内容变化，先删除旧记录
                if "内容已变化" in reason:
                    delete_old_records(db_path, str(filepath))
                    updated_count += 1
                    print(f"   🔄 {reason}")
            
            # 读取文档内容
            content = filepath.read_text(encoding='utf-8')
            
            # 计算文档哈希
            file_hash = compute_file_hash(filepath)
            
            # 切分文档
            chunks = split_document(content)
            
            # 编码向量
            vectors = embedding_service.encode(chunks)
            
            # 准备 metadata
            metadata = [{
                "source": str(filepath),
                "filename": filepath.name,
                "relative_path": str(relative_path),
                "file_hash": file_hash,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "imported_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
            
            # 添加到向量数据库
            vector_store.add(
                contents=chunks,
                vectors=vectors,
                source=str(filepath),
                metadata=metadata
            )
            
            imported_count += 1
            print(f"   ✅ 已导入 {len(chunks)} 个切片 (hash: {file_hash[:8]}...)\n")
        
        except Exception as e:
            failed_count += 1
            print(f"   ❌ 导入失败: {e}\n")
    
    # 输出统计
    print(f"\n{'='*60}")
    print(f"📊 导入完成")
    print(f"{'='*60}")
    print(f"✅ 新增导入: {imported_count - updated_count}")
    print(f"🔄 更新导入: {updated_count}")
    print(f"⏭️  跳过: {skipped_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"📦 总计处理: {len(md_files)}")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 文档增量导入工具")
    parser.add_argument(
        "--docs-dir",
        default="~/.hermes/docs",
        help="文档目录路径"
    )
    parser.add_argument(
        "--db-path",
        default="~/.hermes/rag_vector_db",
        help="向量数据库路径"
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-zh-v1.5",
        help="Embedding 模型名称"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量导入模式（只导入新文档或变化文档）"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量导入模式（导入所有文档，但不清空数据库）"
    )
    
    args = parser.parse_args()
    
    # 默认使用增量模式
    incremental = not args.full
    
    import_documents(
        docs_dir=args.docs_dir,
        db_path=os.path.expanduser(args.db_path),
        model_name=args.model,
        incremental=incremental
    )
