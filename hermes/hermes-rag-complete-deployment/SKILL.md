---
name: hermes-rag-complete-deployment
description: Complete end-to-end RAG plugin deployment for Hermes Agent — installation, configuration, chunking optimization, and auto-import setup
tags: [hermes, rag, vector-database, memory, deployment, chromadb]
---

# Hermes RAG 完整部署指南

从零开始部署 Hermes CPU RAG 记忆插件的完整流程，包括安装、配置、切片优化和自动导入。

## 适用场景

- 新 Hermes 实例需要启用 RAG 记忆功能
- 需要导入大量历史文档到向量数据库
- 优化现有 RAG 切片策略避免内容截断
- 配置自动增量导入新对话记录

## 前置要求

- Hermes Agent 已安装（`~/.hermes/` 目录存在）
- Python 3.8+ 环境
- 至少 2GB 可用磁盘空间（向量数据库）

---

## 第一步：安装 CPU RAG 插件

### 1.1 获取插件

**方式 A：使用内置插件（推荐）**

如果你的 Hermes Agent 是最新版本，CPU RAG 插件已经内置：

```bash
ls ~/.hermes/hermes-agent/plugins/memory/cpu_rag/
```

如果目录存在，跳到步骤 1.2。

**方式 B：从 GitHub 下载**

如果插件不存在，从技能仓库下载：

```bash
cd ~/.hermes/hermes-agent/plugins/memory/
git clone https://github.com/kaiser1103/skills.git /tmp/skills
cp -r /tmp/skills/hermes/hermes-rag-complete-deployment/cpu_rag ./
cd cpu_rag
```

### 1.2 安装依赖

**方式 A：使用自动安装脚本（推荐）**

```bash
cd ~/.hermes/hermes-agent/plugins/memory/cpu_rag/
chmod +x install.sh
./install.sh
```

脚本会自动：
- 安装所有依赖包
- 下载 embedding 模型（BAAI/bge-small-zh-v1.5）
- 配置 Hermes config.yaml
- 创建向量数据库目录

**方式 B：手动安装**

```bash
pip install -r requirements.txt
```

核心依赖：
- `lancedb` - 向量数据库
- `onnxruntime` - 模型推理（优先）或 `torch`（备选）
- `transformers` - 嵌入模型
- `langchain` / `langchain-community` - 文档处理

### 1.3 验证安装

```bash
python -c "import lancedb; print('LanceDB:', lancedb.__version__)"
python -c "from transformers import AutoModel; print('Transformers: OK')"
python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
```

---

## 第二步：注册插件到 Hermes

### 2.1 修改 `config.yaml`

编辑 `~/.hermes/config.yaml`，添加 memory provider：

```yaml
memory:
  provider: cpu_rag
  cpu_rag:
    db_path: ~/.hermes/rag_vector_db
    collection_name: hermes_memory
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    chunk_size: 1000
    chunk_overlap: 200
    top_k: 5
```

### 2.2 创建向量数据库目录

```bash
mkdir -p ~/.hermes/rag_vector_db
```

### 2.3 重启 Hermes

```bash
# CLI 模式
hermes

# Gateway 模式
cd ~/.hermes/hermes-agent/gateway
python run.py
```

### 2.4 验证插件加载

在 Hermes 对话中测试：

```
/memory search test
```

如果返回 "No results found" 而不是错误，说明插件已正常加载。

---

## 第三步：优化切片策略（避免内容截断）

### 3.1 问题诊断

**症状**：
- 搜索时找不到文档中明确存在的内容
- 文档被截断，只导入了前半部分
- 多层级标题结构被破坏

**原因**：
- 默认 `RecursiveCharacterTextSplitter` 按固定长度切片
- 忽略 Markdown 标题层级结构
- `chunk_size=1000` 对长文档会造成截断

### 3.2 修改切片代码

编辑 `~/.hermes/hermes-agent/plugins/memory/cpu_rag/import_docs.py`：

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

def chunk_document(content: str, metadata: dict) -> list:
    """多层级标题切分，保留完整结构"""
    
    # 定义标题层级
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]
    
    # 按标题切分
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # 保留标题文本
    )
    
    chunks = markdown_splitter.split_text(content)
    
    # 为每个 chunk 添加元数据
    result = []
    for i, chunk in enumerate(chunks):
        chunk_meta = metadata.copy()
        chunk_meta.update({
            'chunk_index': i,
            'total_chunks': len(chunks),
            'headers': chunk.metadata  # 标题层级信息
        })
        result.append({
            'content': chunk.page_content,
            'metadata': chunk_meta
        })
    
    return result
```

### 3.3 处理超长段落

对于单个段落超过 `chunk_size` 的情况：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_long_chunk(chunk: str, max_size: int = 2000) -> list:
    """二次切分超长段落"""
    if len(chunk) <= max_size:
        return [chunk]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    return splitter.split_text(chunk)
```

### 3.4 验证切片效果

```bash
cd ~/.hermes/hermes-agent/plugins/memory/cpu_rag
python import_docs.py --dry-run --file test.md
```

检查输出：
- 每个 chunk 是否包含完整语义单元
- 标题层级是否保留
- 是否有内容遗漏

---

## 第四步：批量导入历史文档

### 4.1 准备文档目录

```bash
mkdir -p ~/.hermes/docs_to_import
cp /path/to/your/docs/*.md ~/.hermes/docs_to_import/
```

### 4.2 使用快速导入脚本

创建 `import_docs_fast.py`：

```python
#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
from cpu_rag import RAGMemory

def file_hash(filepath: Path) -> str:
    """计算文件 SHA256 哈希"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def import_directory(docs_dir: str, rag: RAGMemory):
    """批量导入目录下所有 Markdown 文件"""
    docs_path = Path(docs_dir).expanduser()
    files = list(docs_path.glob("**/*.md"))
    
    print(f"Found {len(files)} markdown files")
    
    imported = 0
    skipped = 0
    
    for filepath in files:
        file_id = file_hash(filepath)
        
        # 检查是否已导入
        if rag.check_exists(file_id):
            print(f"⏭️  Skip: {filepath.name} (already imported)")
            skipped += 1
            continue
        
        # 读取并导入
        content = filepath.read_text(encoding='utf-8')
        metadata = {
            'source': str(filepath),
            'file_id': file_id,
            'filename': filepath.name,
            'import_time': datetime.now().isoformat()
        }
        
        rag.add_document(content, metadata)
        print(f"✅ Imported: {filepath.name}")
        imported += 1
    
    print(f"\n📊 Summary: {imported} imported, {skipped} skipped")

if __name__ == "__main__":
    rag = RAGMemory(db_path="~/.hermes/rag_vector_db")
    import_directory("~/.hermes/docs_to_import", rag)
```

### 4.3 执行导入

```bash
cd ~/.hermes/hermes-agent/plugins/memory/cpu_rag
python import_docs_fast.py
```

预期速度：1.5-2.0 文件/秒（取决于文件大小和 CPU）

### 4.4 验证导入结果

```python
# 检查记忆数量
python -c "
from cpu_rag import RAGMemory
rag = RAGMemory()
print(f'Total memories: {rag.count()}')
"
```

在 Hermes 中测试搜索：

```
/memory search 你的关键词
```

---

## 第五步：配置自动增量导入

### 5.1 创建会话保存钩子

编辑 `~/.hermes/hermes-agent/hermes_state.py`，添加保存后回调：

```python
class SessionDB:
    def save_session(self, session_id: str, messages: list):
        """保存会话并触发 RAG 导入"""
        # 原有保存逻辑
        self._save_to_db(session_id, messages)
        
        # 触发 RAG 导入
        self._trigger_rag_import(session_id, messages)
    
    def _trigger_rag_import(self, session_id: str, messages: list):
        """将会话内容导入 RAG"""
        try:
            from plugins.memory.cpu_rag import RAGMemory
            rag = RAGMemory()
            
            # 提取对话内容
            content = self._format_session_content(messages)
            
            # 计算哈希避免重复导入
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            if not rag.check_exists(content_hash):
                metadata = {
                    'source': 'session',
                    'session_id': session_id,
                    'content_hash': content_hash,
                    'import_time': datetime.now().isoformat()
                }
                rag.add_document(content, metadata)
                print(f"✅ Session {session_id} imported to RAG")
        except Exception as e:
            print(f"⚠️  RAG import failed: {e}")
```

### 5.2 配置导入策略

在 `config.yaml` 中添加：

```yaml
memory:
  auto_import:
    enabled: true
    trigger: session_save  # 会话保存时触发
    min_messages: 5  # 至少 5 条消息才导入
    exclude_commands: true  # 排除纯命令会话
```

### 5.3 测试自动导入

1. 进行一次正常对话（5+ 轮）
2. 退出 Hermes（触发会话保存）
3. 重新进入，使用 `/memory search` 搜索刚才的对话内容
4. 应该能找到刚才的对话记录

---

## 常见问题排查

### 问题 1：搜索不到已导入的文档

**诊断步骤**：

```bash
# 1. 检查数据库是否有数据
python -c "from cpu_rag import RAGMemory; print(RAGMemory().count())"

# 2. 检查文档是否真的导入了
python -c "
from cpu_rag import RAGMemory
rag = RAGMemory()
results = rag.search('', top_k=10)  # 空查询返回最近文档
for r in results:
    print(r['metadata']['filename'])
"

# 3. 测试嵌入模型
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode('test')
print(f'Embedding dim: {len(emb)}')
"
```

**可能原因**：
- 文档被截断，关键内容未导入 → 优化切片策略（第三步）
- 查询词与文档语义差异大 → 尝试不同关键词
- 嵌入模型未正确加载 → 重新安装 `sentence-transformers`

### 问题 2：导入速度慢

**优化方法**：

```python
# 批量插入而非逐条插入
def batch_import(documents: list, rag: RAGMemory, batch_size=50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        rag.add_documents_batch(batch)  # 批量接口
        print(f"Imported {i+len(batch)}/{len(documents)}")
```

### 问题 3：向量数据库损坏

**症状**：
```
sqlite3.DatabaseError: database disk image is malformed
```

**修复**：

```bash
# 备份现有数据库
cp -r ~/.hermes/rag_vector_db ~/.hermes/rag_vector_db.backup

# 重建数据库
rm -rf ~/.hermes/rag_vector_db
mkdir ~/.hermes/rag_vector_db

# 重新导入
python import_docs_fast.py
```

### 问题 4：内存占用过高

**原因**：嵌入模型加载到内存

**优化**：

```yaml
# config.yaml
memory:
  cpu_rag:
    embedding_model: sentence-transformers/all-MiniLM-L6-v2  # 轻量模型
    device: cpu  # 强制使用 CPU
    batch_size: 32  # 减小批次大小
```

---

## 性能基准

| 指标 | 参考值 |
|------|--------|
| 导入速度 | 1.5-2.0 文件/秒 |
| 搜索延迟 | 50-200ms (top_k=5) |
| 内存占用 | 500MB-1GB (含嵌入模型) |
| 磁盘占用 | ~3.5KB/条记忆 |

---

## 维护建议

### 定期清理

```bash
# 删除超过 6 个月的旧记忆
python -c "
from cpu_rag import RAGMemory
from datetime import datetime, timedelta
rag = RAGMemory()
cutoff = (datetime.now() - timedelta(days=180)).isoformat()
rag.delete_before(cutoff)
"
```

### 备份数据库

```bash
# 每周备份
tar -czf rag_backup_$(date +%Y%m%d).tar.gz ~/.hermes/rag_vector_db
```

### 监控数据库大小

```bash
du -sh ~/.hermes/rag_vector_db
```

---

## 进阶配置

### 使用更强的嵌入模型

```yaml
memory:
  cpu_rag:
    embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    # 支持中文，但模型更大 (1GB)
```

### 多路召回策略

```python
def hybrid_search(query: str, rag: RAGMemory):
    """向量搜索 + 关键词搜索"""
    vector_results = rag.search(query, top_k=10)
    keyword_results = rag.keyword_search(query, top_k=10)
    
    # 合并去重
    combined = merge_and_rerank(vector_results, keyword_results)
    return combined[:5]
```

### 分层存储

```yaml
memory:
  cpu_rag:
    collections:
      - name: recent  # 最近 30 天
        ttl: 2592000
      - name: archive  # 历史归档
        ttl: null
```

---

## 参考资料

- ChromaDB 文档: https://docs.trychroma.com/
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Sentence Transformers: https://www.sbert.net/

---

## 更新日志

- 2026-04-17: 初始版本，整合安装、切片优化、自动导入流程
