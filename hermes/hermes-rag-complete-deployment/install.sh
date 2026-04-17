#!/bin/bash
#
# CPU RAG 插件安装脚本
# 适用于 Hermes Agent 记忆插件部署
#

set -e

echo "🚀 CPU RAG 插件安装"
echo "========================"

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📋 Python 版本: $python_version"

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  未检测到虚拟环境"
    echo ""
    echo "Python 3.11+ 系统不允许直接安装到系统 Python。"
    echo ""
    echo "推荐方案："
    echo "  1. 创建虚拟环境（推荐）："
    echo "     python3 -m venv ~/.hermes/venv"
    echo "     source ~/.hermes/venv/bin/activate"
    echo "     ./install.sh"
    echo ""
    echo "  2. 强制安装到系统（不推荐，可能破坏系统包）"
    echo ""
    read -p "选择: [1] 虚拟环境 / [2] 强制安装 / [Q] 退出: " -n 1 -r
    echo
    
    case $REPLY in
        1)
            echo "请先创建并激活虚拟环境，然后重新运行此脚本"
            exit 0
            ;;
        2)
            echo "⚠️  使用 --break-system-packages 强制安装"
            PIP_FLAGS="--break-system-packages"
            ;;
        *)
            echo "已取消安装"
            exit 0
            ;;
    esac
else
    echo "✓ 检测到虚拟环境: $VIRTUAL_ENV"
    PIP_FLAGS=""
fi

# 安装核心依赖
echo ""
echo "📦 安装核心依赖..."
pip install $PIP_FLAGS -q lancedb numpy pyarrow

# 安装模型推理依赖
echo "📦 安装模型推理依赖..."

# 尝试安装 ONNX Runtime
if pip install $PIP_FLAGS -q onnxruntime 2>/dev/null; then
    echo "  ✓ ONNX Runtime 安装成功"
    USE_ONNX=true
else
    echo "  ⚠️  ONNX Runtime 安装失败，将使用 PyTorch"
    USE_ONNX=false
fi

# 安装 Transformers
pip install $PIP_FLAGS -q transformers sentencepiece protobuf

# 如果 ONNX 不可用，安装 PyTorch
if [ "$USE_ONNX" = false ]; then
    echo "📦 安装 PyTorch (CPU 版本)..."
    pip install $PIP_FLAGS -q torch --index-url https://download.pytorch.org/whl/cpu
fi

# 确定模型路径
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
MODEL_DIR="$HERMES_HOME/models/embeddings/bge-small-zh-v1.5"

echo ""
echo "🤖 下载 Embedding 模型..."
echo "  模型: BAAI/bge-small-zh-v1.5"
echo "  路径: $MODEL_DIR"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "  ✓ 模型已存在，跳过下载"
else
    mkdir -p "$MODEL_DIR"
    python3 << 'EOF'
import os
from transformers import AutoTokenizer, AutoModel

model_name = "BAAI/bge-small-zh-v1.5"
model_path = os.path.expanduser("~/.hermes/models/embeddings/bge-small-zh-v1.5")

print("下载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_path)

print("下载模型...")
model = AutoModel.from_pretrained(model_name)
model.save_pretrained(model_path)

print("✓ 完成")
EOF
fi

# 如果安装了 ONNX，转换模型
if [ "$USE_ONNX" = true ]; then
    echo ""
    echo "🔄 转换为 ONNX 格式..."
    
    ONNX_DIR="$HERMES_HOME/models/embeddings/bge-small-zh-v1.5-onnx"
    if [ -f "$ONNX_DIR/model.onnx" ]; then
        echo "  ✓ ONNX 模型已存在，跳过转换"
    else
        pip install $PIP_FLAGS -q optimum[exporters] 2>/dev/null || {
            echo "  ⚠️  optimum 安装失败，跳过 ONNX 转换（使用 PyTorch 模式）"
        }
        
        if command -v optimum-cli &> /dev/null; then
            optimum-cli export onnx \
                --model "$MODEL_DIR" \
                --task feature-extraction \
                "$ONNX_DIR" \
                2>/dev/null || echo "  ⚠️  ONNX 转换失败（使用 PyTorch 模式）"
        fi
    fi
fi

# 配置 Hermes
echo ""
echo "⚙️  配置 Hermes..."

CONFIG_FILE="$HERMES_HOME/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    # 检查是否已配置
    if grep -q "provider: cpu_rag" "$CONFIG_FILE" 2>/dev/null; then
        echo "  ✓ 已配置 cpu_rag 提供者"
    else
        echo "  添加配置到 $CONFIG_FILE"
        # 备份原配置
        cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
        
        # 添加配置
        python3 << EOF
import yaml

config_path = "$CONFIG_FILE"
with open(config_path) as f:
    config = yaml.safe_load(f) or {}

# 添加 memory 配置
config.setdefault("memory", {})
config["memory"]["provider"] = "cpu_rag"

# 添加插件配置
config.setdefault("plugins", {})
config["plugins"]["cpu_rag"] = {
    "db_path": "$HERMES_HOME/rag_vector_db",
    "model_name": "BAAI/bge-small-zh-v1.5",
    "max_memory_mb": 512,
    "top_k": 5,
    "min_score": 0.6
}

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("  ✓ 配置已更新")
EOF
    fi
else
    echo "  创建新配置文件: $CONFIG_FILE"
    cat > "$CONFIG_FILE" << EOF
memory:
  provider: cpu_rag

plugins:
  cpu_rag:
    db_path: $HERMES_HOME/rag_vector_db
    model_name: BAAI/bge-small-zh-v1.5
    max_memory_mb: 512
    top_k: 5
    min_score: 0.6
EOF
fi

# 创建向量数据库目录
mkdir -p "$HERMES_HOME/rag_vector_db"

echo ""
echo "✅ 安装完成！"
echo ""
echo "💡 使用提示:"
echo "  1. 重新启动 Hermes"
echo "  2. 使用 rag_search 检索记忆"
echo "  3. 使用 rag_add_memory 添加记忆"
echo ""
echo "📋 常用命令:"
echo "  hermes                 # 启动 CLI"
echo "  rag_search('关键词')   # 搜索记忆"
echo "  rag_add_memory('内容') # 添加记忆"
echo ""
