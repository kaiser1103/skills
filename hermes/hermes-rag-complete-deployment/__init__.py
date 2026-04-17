"""CPU + 内存 RAG 记忆插件

纯 CPU 运行的向量检索记忆系统，无需 GPU，无需外部服务。
使用 LanceDB 存储向量，ONNX Runtime 进行 CPU 推理。

配置在 $HERMES_HOME/config.yaml:
  memory:
    provider: cpu_rag
  plugins:
    cpu_rag:
      db_path: $HERMES_HOME/rag_vector_db
      model_name: BAAI/bge-small-zh-v1.5
      max_memory_mb: 512
      top_k: 5
      min_score: 0.6
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 工具模式定义
# ---------------------------------------------------------------------------

RAG_SEARCH_SCHEMA = {
    "name": "rag_search",
    "description": (
        "【优先使用】检索历史记忆和知识库。"
        "使用向量语义搜索相关的历史对话、文档内容。\n\n"
        "优先级: 当需要检索历史信息时，优先使用 rag_search（语义搜索），"
        "只有在需要精确关键词匹配或时间范围查询时才使用 session_search。\n\n"
        "用法场景:\n"
        "- 用户问'我们之前讨论的...'时检索历史（模糊回忆）\n"
        "- 需要回忆具体的项目细节、技术方案时（主题检索）\n"
        "- 补充当前会话缺失的上下文（语义关联）\n"
        "- 查找相关概念和同义词（如'部署'能匹配'安装'）\n\n"
        "何时回退到 session_search:\n"
        "- 需要精确匹配命令或代码片段（如 'git clone'）\n"
        "- 需要按时间范围查询（如'最近3天的对话'）\n"
        "- 需要精确字符串匹配（如 IP 地址、URL）\n\n"
        "示例:\n"
        '- rag_search(query="项目架构设计")  # 语义搜索\n'
        '- rag_search(query="用户需求文档", top_k=3)  # 主题检索'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询词，支持自然语言"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量（默认: 5）",
                "default": 5
            },
            "filter_source": {
                "type": "string",
                "description": "过滤来源（如 conversation, document）"
            }
        },
        "required": ["query"]
    }
}

RAG_ADD_MEMORY_SCHEMA = {
    "name": "rag_add_memory",
    "description": (
        "主动添加记忆到知识库。"
        "用于保存重要信息，方便未来检索。\n\n"
        "用法场景:\n"
        "- 用户明确说'请记住...'时\n"
        "- 做出重要决定、设计方案时\n"
        "- 收集用户偏好、项目信息时\n\n"
        "示例:\n"
        '- rag_add_memory(content="用户偏好使用 PostgreSQL 数据库")\n'
        '- rag_add_memory(content="HuanForge 架构设计", source="project")'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要保存的记忆内容"
            },
            "source": {
                "type": "string",
                "description": "来源标记（默认: conversation）",
                "default": "conversation"
            },
            "metadata": {
                "type": "object",
                "description": "额外元数据（可选）"
            }
        },
        "required": ["content"]
    }
}

RAG_STATS_SCHEMA = {
    "name": "rag_stats",
    "description": "查看 RAG 记忆库统计信息",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """从 config.yaml 加载插件配置"""
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("cpu_rag", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 核心组件（内嵌实现，避免额外依赖）
# ---------------------------------------------------------------------------

class _LazyImport:
    """延迟导入，避免插件加载时就导入所有依赖"""
    
    _np = None
    _ort = None
    _lancedb = None
    _pa = None
    _AutoTokenizer = None
    _AutoModel = None
    
    @classmethod
    def numpy(cls):
        if cls._np is None:
            import numpy as np
            cls._np = np
        return cls._np
    
    @classmethod
    def onnxruntime(cls):
        if cls._ort is None:
            import onnxruntime as ort
            cls._ort = ort
        return cls._ort
    
    @classmethod
    def lancedb(cls):
        if cls._lancedb is None:
            import lancedb
            cls._lancedb = lancedb
        return cls._lancedb
    
    @classmethod
    def pyarrow(cls):
        if cls._pa is None:
            import pyarrow as pa
            cls._pa = pa
        return cls._pa
    
    @classmethod
    def transformers(cls):
        if cls._AutoTokenizer is None:
            from transformers import AutoTokenizer, AutoModel
            cls._AutoTokenizer = AutoTokenizer
            cls._AutoModel = AutoModel
        return cls._AutoTokenizer, cls._AutoModel


class _EmbeddingService:
    """嵌入服务 - 使用 ONNX Runtime 进行 CPU 推理"""
    
    def __init__(self, model_path: str = None, max_memory_mb: int = 512):
        from hermes_constants import get_hermes_home
        
        self.max_memory_mb = max_memory_mb
        self.tokenizer = None
        self.session = None
        self._lock = threading.Lock()
        
        # 默认模型路径
        if model_path is None:
            model_path = str(get_hermes_home() / "models" / "embeddings" / "bge-small-zh-v1.5")
        
        self.model_path = model_path
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            ort = _LazyImport.onnxruntime()
            AutoTokenizer, _ = _LazyImport.transformers()
            
            logger.info("正在初始化 CPU Embedding 服务...")
            
            # 加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # ONNX 模型路径
            onnx_path = Path(self.model_path).parent / "bge-small-zh-v1.5-onnx"
            if not onnx_path.exists():
                onnx_path = Path(self.model_path)  # 可能已经是 ONNX 格式
            
            model_file = onnx_path / "model.onnx"
            
            # 如果没有 ONNX 模型，使用 PyTorch 回退
            if not model_file.exists():
                logger.warning("未找到 ONNX 模型，使用 PyTorch 模式")
                self._use_pytorch = True
                self._init_pytorch()
                return
            
            self._use_pytorch = False
            
            # 配置 ONNX Runtime
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            
            # 设置线程数
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            sess_options.intra_op_num_threads = min(4, cpu_count)
            sess_options.inter_op_num_threads = min(2, cpu_count)
            
            # 加载模型
            self.session = ort.InferenceSession(
                str(model_file),
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"✓ Embedding 服务初始化完成（ONNX模式，{cpu_count}核CPU）")
            
        except Exception as e:
            logger.warning(f"ONNX 初始化失败: {e}，尝试 PyTorch 模式")
            self._use_pytorch = True
            self._init_pytorch()
    
    def _init_pytorch(self):
        """使用 PyTorch 作为回退"""
        try:
            AutoTokenizer, AutoModel = _LazyImport.transformers()
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.torch_model = AutoModel.from_pretrained(self.model_path)
            self.torch_model.eval()
            
            logger.info("✓ Embedding 服务初始化完成（PyTorch模式）")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 8, normalize: bool = True):
        """编码文本为向量"""
        np = _LazyImport.numpy()
        
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        with self._lock:  # 线程安全
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if self._use_pytorch:
                    embeddings = self._encode_pytorch(batch)
                else:
                    embeddings = self._encode_onnx(batch)
                
                all_embeddings.append(embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings
    
    def _encode_onnx(self, texts: List[str]):
        """ONNX 推理"""
        np = _LazyImport.numpy()
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs['input_ids']}
        )
        
        # Mean pooling
        return self._mean_pooling(outputs[0], inputs['attention_mask'])
    
    def _encode_pytorch(self, texts: List[str]):
        """PyTorch 推理"""
        import torch
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.torch_model(**inputs)
            # BGE 模型的输出: last_hidden_state 是 (batch, seq_len, hidden_size)
            # 我们需要取 [CLS] token 的表示或做 mean pooling
            # BGE 推荐使用 [CLS] token 或 mean pooling
            
            # Mean pooling
            embeddings = self._mean_pooling_torch(outputs.last_hidden_state, inputs['attention_mask'])
        
        return embeddings
    
    def _mean_pooling_torch(self, token_embeddings, attention_mask):
        """平均池化 (PyTorch)"""
        import torch
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).numpy()
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    
    def _mean_pooling_torch(self, token_embeddings, attention_mask):
        """平均池化 (PyTorch)"""
        import torch
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).numpy()
    
    def _normalize(self, embeddings):
        """归一化向量"""
        np = _LazyImport.numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)


class _VectorStore:
    """向量数据库 - 基于 LanceDB"""
    
    def __init__(self, db_path: str = None, max_memory_mb: int = 1024):
        from hermes_constants import get_hermes_home
        
        if db_path is None:
            db_path = str(get_hermes_home() / "rag_vector_db")
        
        self.db_path = db_path
        self.max_memory_mb = max_memory_mb
        self.db = None
        self.table = None
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        try:
            lancedb = _LazyImport.lancedb()
            pa = _LazyImport.pyarrow()
            
            logger.info(f"正在初始化向量数据库: {self.db_path}")
            
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)
            
            # 检查或创建表
            if "memories" not in self.db.table_names():
                self._create_table()
            else:
                self.table = self.db.open_table("memories")
            
            logger.info("✓ 数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    def reload(self):
        """重新加载数据库（用于导入后刷新）"""
        try:
            logger.info("正在重新加载向量数据库...")
            # 重新连接数据库
            lancedb = _LazyImport.lancedb()
            self.db = lancedb.connect(self.db_path)
            # 重新打开表
            if "memories" in self.db.table_names():
                self.table = self.db.open_table("memories")
                logger.info("✓ 数据库重新加载完成")
            else:
                logger.warning("表 'memories' 不存在")
        except Exception as e:
            logger.error(f"重新加载失败: {e}")
    
    def _create_table(self):
        """创建新表"""
        pa = _LazyImport.pyarrow()
        
        schema = pa.schema([
            ("id", pa.string()),
            ("content", pa.string()),
            ("vector", pa.list_(pa.float32(), 512)),  # BGE-small-zh-v1.5 是 512维
            ("source", pa.string()),
            ("metadata", pa.string()),  # JSON 字符串
            ("timestamp", pa.string()),
            ("access_count", pa.int32()),
        ])
        
        self.table = self.db.create_table("memories", schema=schema)
        logger.info("创建了新的 memories 表")
    
    def add(self, contents: List[str], vectors, source: str = "", metadata: List[Dict] = None):
        """添加向量数据"""
        import uuid
        from datetime import datetime
        
        if metadata is None:
            metadata = [{} for _ in contents]
        
        timestamp = datetime.now().isoformat()
        
        # 准备数据
        data = []
        for i, (content, vector) in enumerate(zip(contents, vectors)):
            data.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "vector": vector.tolist() if hasattr(vector, 'tolist') else list(vector),
                "source": source,
                "metadata": json.dumps(metadata[i]),
                "timestamp": timestamp,
                "access_count": 0,
            })
        
        # 批量插入
        self.table.add(data)
        logger.info(f"添加了 {len(data)} 条记忆")
        
        # 如果数据量大，创建索引
        if len(data) > 100:
            self._create_index()
    
    def search(self, query_vector, top_k: int = 5, filter_expr: str = None) -> List[Dict]:
        """向量搜索"""
        # 检查是否需要重新加载
        reload_marker = Path(self.db_path) / ".reload_needed"
        if reload_marker.exists():
            logger.info("检测到数据库更新标记，正在重新加载...")
            self.reload()
            try:
                reload_marker.unlink()
            except Exception:
                pass
        
        # 转换向量为列表
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()
        elif not isinstance(query_vector, list):
            query_vector = list(query_vector)
        
        # 搜索
        results = self.table.search(query_vector).metric("cosine").limit(top_k)
        
        if filter_expr:
            results = results.where(filter_expr)
        
        # 执行搜索
        df = results.to_df()
        
        # 转换为字典列表
        matches = []
        for _, row in df.iterrows():
            matches.append({
                "id": row["id"],
                "content": row["content"],
                "score": 1.0 - row["_distance"],  # 转换为相似度
                "source": row["source"],
                "metadata": row["metadata"],
                "timestamp": row["timestamp"],
            })
        
        return matches
    
    def _create_index(self):
        """创建向量索引"""
        try:
            self.table.create_index(
                metric="cosine",
                num_partitions=256,
                num_sub_vectors=16
            )
            logger.info("创建了向量索引 (IVF)")
        except Exception as e:
            logger.debug(f"创建索引失败（可能已存在）: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        try:
            return {
                "total_records": len(self.table),
                "table_names": self.db.table_names(),
                "db_path": self.db_path,
            }
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# MemoryProvider 实现
# ---------------------------------------------------------------------------

class CPURAGMemoryProvider(MemoryProvider):
    """纯 CPU RAG 记忆提供者
    
    使用 LanceDB + ONNX Runtime 实现纯 CPU 的向量检索记忆。
    适合无 GPU 环境，成本低、部署简单。
    """
    
    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._embedding_service = None
        self._vector_store = None
        self._session_id = ""
        self._lock = threading.Lock()
        
        # 配置参数
        self._top_k = int(self._config.get("top_k", 5))
        self._min_score = float(self._config.get("min_score", 0.6))
        self._max_memory_mb = int(self._config.get("max_memory_mb", 512))
    
    @property
    def name(self) -> str:
        return "cpu_rag"
    
    def is_available(self) -> bool:
        """检查依赖是否可用"""
        try:
            import lancedb
            import numpy
            import pyarrow
            from transformers import AutoTokenizer
            return True
        except ImportError as e:
            logger.debug(f"CPU RAG 依赖缺失: {e}")
            return False
    
    def get_config_schema(self) -> List[Dict[str, Any]]:
        """返回配置字段"""
        from hermes_constants import display_hermes_home
        return [
            {
                "key": "db_path",
                "description": "向量数据库路径",
                "default": f"{display_hermes_home()}/rag_vector_db"
            },
            {
                "key": "model_name",
                "description": "Embedding 模型名称",
                "default": "BAAI/bge-small-zh-v1.5"
            },
            {
                "key": "max_memory_mb",
                "description": "最大内存限制(MB)",
                "default": "512"
            },
            {
                "key": "top_k",
                "description": "默认检索结果数量",
                "default": "5"
            },
            {
                "key": "min_score",
                "description": "最小相似度阈值",
                "default": "0.6"
            },
        ]
    
    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """保存配置到 config.yaml"""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["cpu_rag"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.warning(f"保存配置失败: {e}")
    
    def initialize(self, session_id: str, **kwargs) -> None:
        """初始化"""
        from hermes_constants import get_hermes_home
        
        self._session_id = session_id
        hermes_home = str(get_hermes_home())
        
        # 解析配置路径
        db_path = self._config.get("db_path", f"{hermes_home}/rag_vector_db")
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        
        model_name = self._config.get("model_name", "BAAI/bge-small-zh-v1.5")
        model_path = f"{hermes_home}/models/embeddings/{model_name.split('/')[-1]}"
        
        # 初始化组件
        try:
            self._vector_store = _VectorStore(
                db_path=db_path,
                max_memory_mb=self._max_memory_mb
            )
            
            self._embedding_service = _EmbeddingService(
                model_path=model_path,
                max_memory_mb=self._max_memory_mb
            )
            
            logger.info("✓ CPU RAG 记忆提供者初始化完成")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def system_prompt_block(self) -> str:
        """系统提示词块"""
        if not self._vector_store:
            # 未初始化，返回基本提示
            return (
                "# CPU RAG 记忆\n"
                "已配置但未初始化。等待初始化完成后可使用向量检索功能。"
            )
        
        try:
            stats = self._vector_store.get_stats()
            total = stats.get("total_records", 0)
        except Exception:
            total = 0
        
        if total == 0:
            return (
                "# CPU RAG 记忆\n"
                "已激活但数据库为空。"
                "使用 rag_add_memory 添加重要信息，或调用 rag_search 检索历史记忆。"
            )
        
        return (
            f"# CPU RAG 记忆\n"
            f"已存储 {total} 条记忆。"
            f"使用 rag_search 检索相关内容，"
            f"rag_add_memory 添加新记忆。"
        )
    
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """预获取相关上下文"""
        if not self._embedding_service or not self._vector_store or not query:
            return ""
        
        try:
            # 编码查询
            query_vector = self._embedding_service.encode([query])[0]
            
            # 搜索
            results = self._vector_store.search(
                query_vector=query_vector,
                top_k=self._top_k
            )
            
            # 过滤低分结果
            results = [r for r in results if r["score"] >= self._min_score]
            
            if not results:
                return ""
            
            # 组装上下文
            lines = ["## 相关记忆"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. [{r['score']:.2f}] {r['content'][:200]}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.debug(f"预获取失败: {e}")
            return ""
    
    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """同步转到后端（自动保存重要内容）"""
        # 可以在这里添加自动提取逻辑
        # 目前使用显式工具调用，不自动保存
        pass
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回工具模式"""
        return [RAG_SEARCH_SCHEMA, RAG_ADD_MEMORY_SCHEMA, RAG_STATS_SCHEMA]
    
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理工具调用"""
        if tool_name == "rag_search":
            return self._handle_search(args)
        elif tool_name == "rag_add_memory":
            return self._handle_add_memory(args)
        elif tool_name == "rag_stats":
            return self._handle_stats()
        return tool_error(f"未知工具: {tool_name}")
    
    def _handle_search(self, args: dict) -> str:
        """处理搜索请求"""
        try:
            query = args.get("query", "")
            top_k = int(args.get("top_k", self._top_k))
            filter_source = args.get("filter_source")
            
            if not query:
                return tool_error("查询不能为空")
            
            # 编码查询
            query_vector = self._embedding_service.encode([query])[0]
            
            # 搜索
            results = self._vector_store.search(
                query_vector=query_vector,
                top_k=top_k * 2  # 多搜一些用于过滤
            )
            
            # 过滤
            filtered = [r for r in results if r["score"] >= self._min_score]
            if filter_source:
                filtered = [r for r in filtered if r.get("source") == filter_source]
            
            results = filtered[:top_k]
            
            return json.dumps({
                "query": query,
                "results": results,
                "count": len(results),
                "min_score": self._min_score
            }, ensure_ascii=False)
            
        except Exception as e:
            return tool_error(f"搜索失败: {e}")
    
    def _handle_add_memory(self, args: dict) -> str:
        """处理添加记忆请求"""
        try:
            content = args.get("content", "")
            source = args.get("source", "conversation")
            metadata = args.get("metadata", {})
            
            if not content:
                return tool_error("内容不能为空")
            
            # 编码
            vector = self._embedding_service.encode([content])
            
            # 添加到数据库
            self._vector_store.add(
                contents=[content],
                vectors=vector,
                source=source,
                metadata=[metadata]
            )
            
            return json.dumps({
                "status": "success",
                "content": content[:100] + "..." if len(content) > 100 else content,
                "source": source
            }, ensure_ascii=False)
            
        except Exception as e:
            return tool_error(f"添加失败: {e}")
    
    def _handle_stats(self) -> str:
        """处理统计请求"""
        try:
            stats = self._vector_store.get_stats()
            return json.dumps({
                "provider": "cpu_rag",
                "status": "active",
                **stats
            }, ensure_ascii=False)
        except Exception as e:
            return tool_error(f"获取统计失败: {e}")
    
    def shutdown(self) -> None:
        """关闭"""
        self._embedding_service = None
        self._vector_store = None
        logger.info("CPU RAG 记忆提供者已关闭")


# ---------------------------------------------------------------------------
# 插件入口点
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """注册 CPU RAG 记忆提供者"""
    config = _load_plugin_config()
    provider = CPURAGMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
