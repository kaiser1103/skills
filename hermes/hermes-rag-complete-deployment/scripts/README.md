# 批量导入脚本

这个目录包含用于批量导入文档到 RAG 向量数据库的工具脚本。

## 脚本说明

### import_docs_fast.py（推荐）

**最快的批量导入工具**，使用批量编码和批量写入优化。

**特点：**
- 批量编码文档（减少模型调用次数）
- 批量写入数据库（减少 I/O 操作）
- 进度条显示
- 速度：~1.8 文件/秒

**使用方法：**
```bash
python scripts/import_docs_fast.py /path/to/docs/
```

**适用场景：**
- 首次导入大量文档（100+ 文件）
- 需要最快导入速度
- 文档格式统一（Markdown）

---

### import_docs_onnx_simple.py

**ONNX 加速版本**，使用 ONNX Runtime 进行模型推理。

**特点：**
- 使用 ONNX Runtime（比 PyTorch 快 2-3x）
- 需要预先转换模型为 ONNX 格式
- 内存占用更低

**使用方法：**
```bash
# 1. 先转换模型（如果还没转换）
optimum-cli export onnx \
  --model ~/.hermes/models/embeddings/bge-small-zh-v1.5 \
  --task feature-extraction \
  ~/.hermes/models/embeddings/bge-small-zh-v1.5-onnx

# 2. 运行导入
python scripts/import_docs_onnx_simple.py /path/to/docs/
```

**适用场景：**
- CPU 环境下需要最快推理速度
- 已有 ONNX 模型
- 大规模文档导入（1000+ 文件）

---

### import_docs_to_rag.py

**基础版本**，逐文件处理。

**特点：**
- 简单直接
- 逐文件编码和写入
- 适合小规模导入

**使用方法：**
```bash
python scripts/import_docs_to_rag.py /path/to/docs/
```

**适用场景：**
- 少量文档导入（< 50 文件）
- 调试和测试
- 学习 RAG 导入流程

---

## 性能对比

| 脚本 | 速度 | 内存占用 | 依赖 |
|------|------|----------|------|
| import_docs_fast.py | ⭐⭐⭐⭐⭐ | 中等 | PyTorch/ONNX |
| import_docs_onnx_simple.py | ⭐⭐⭐⭐ | 低 | ONNX Runtime |
| import_docs_to_rag.py | ⭐⭐⭐ | 低 | PyTorch/ONNX |

## 通用参数

所有脚本都支持以下环境变量：

```bash
# 指定 Hermes 主目录
export HERMES_HOME=~/.hermes

# 指定向量数据库路径
export RAG_DB_PATH=~/.hermes/rag_vector_db

# 指定模型路径
export MODEL_PATH=~/.hermes/models/embeddings/bge-small-zh-v1.5
```

## 故障排查

### 导入速度慢

1. 使用 `import_docs_fast.py`（批量处理）
2. 如果有 ONNX Runtime，使用 `import_docs_onnx_simple.py`
3. 检查磁盘 I/O 性能

### 内存不足

1. 使用 `import_docs_onnx_simple.py`（内存占用最低）
2. 减少批量大小（修改脚本中的 `batch_size`）
3. 分批导入文档

### 模型加载失败

```bash
# 检查模型是否存在
ls ~/.hermes/models/embeddings/bge-small-zh-v1.5/

# 重新下载模型
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('BAAI/bge-small-zh-v1.5')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-zh-v1.5')
model.save_pretrained('~/.hermes/models/embeddings/bge-small-zh-v1.5')
tokenizer.save_pretrained('~/.hermes/models/embeddings/bge-small-zh-v1.5')
"
```

## 自定义导入

如果需要自定义导入逻辑（如特殊文档格式、自定义切片策略），参考 `import_docs_fast.py` 的实现：

1. 继承 `DocumentImporter` 类
2. 重写 `chunk_document()` 方法
3. 调用 `import_documents()` 执行导入

示例见 SKILL.md 第四步。
