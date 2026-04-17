# Hermes RAG 完整部署指南

从零开始部署 Hermes CPU RAG 记忆插件的完整流程。

## 📦 包含文件

```
hermes-rag-complete-deployment/
├── SKILL.md              # 完整部署指南（主文档）
├── __init__.py           # CPU RAG 插件核心代码
├── document_importer.py  # 文档批量导入工具
├── plugin.yaml           # 插件配置文件
├── requirements.txt      # Python 依赖清单
├── install.sh            # 自动安装脚本
├── test_plugin.py        # 插件测试脚本
└── README.md             # 本文件
```

## 🚀 快速开始

### 方法 1：自动安装（推荐）

```bash
cd ~/.hermes/hermes-agent/plugins/memory/cpu_rag/
chmod +x install.sh
./install.sh
```

### 方法 2：手动安装

```bash
pip install -r requirements.txt
```

然后按照 [SKILL.md](SKILL.md) 中的步骤配置。

## 📋 依赖说明

### 核心依赖
- **lancedb** - 向量数据库（轻量级，无需服务器）
- **numpy, pyarrow** - 数据处理
- **transformers** - HuggingFace 模型加载
- **onnxruntime** - 模型推理（优先，CPU 友好）
- **torch** - PyTorch（ONNX 不可用时的备选）

### 可选依赖
- **langchain** - 文档批量导入
- **optimum** - ONNX 模型转换

## 🤖 模型

默认使用 **BAAI/bge-small-zh-v1.5**：
- 中文优化的嵌入模型
- 模型大小：~100MB
- 推理速度：~50ms/query（CPU）
- 自动下载到 `~/.hermes/models/embeddings/`

## 📖 完整文档

查看 [SKILL.md](SKILL.md) 获取：
- 详细安装步骤
- 配置说明
- 切片策略优化
- 批量导入历史文档
- 自动增量导入设置
- 常见问题排查
- 性能优化建议

## ⚙️ 系统要求

- Python 3.8+
- 至少 2GB 可用磁盘空间
- 512MB+ 可用内存
- Hermes Agent 已安装

## 🔧 故障排查

### ONNX Runtime 安装失败

```bash
# 使用 PyTorch 代替
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 模型下载慢

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
```

### 向量搜索无结果

检查切片策略配置，参考 SKILL.md 第三步。

## 📝 许可

MIT License
