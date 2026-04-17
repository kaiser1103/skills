# RAG 搜索优先级配置

## 概述

CPU RAG 插件注册了 `rag_search` 工具到 Hermes，用于语义搜索历史记忆和文档。

## 搜索优先级策略

### 默认行为

当 CPU RAG 插件启用后，Hermes 会同时拥有两个搜索工具：

1. **rag_search** - 向量语义搜索（优先）
2. **session_search** - 关键词全文搜索（备用）

### 工具描述中的优先级提示

`rag_search` 的工具描述中包含 `【优先使用】` 标记，并明确说明：

> "当需要检索历史信息时，优先使用 rag_search（语义搜索），只有在需要精确关键词匹配或时间范围查询时才使用 session_search。"

这会引导 AI 模型在大多数情况下优先选择 `rag_search`。

## 使用场景对比

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| "我们之前讨论的架构方案" | rag_search | 模糊回忆，语义理解 |
| "关于性能优化的建议" | rag_search | 主题检索，同义词匹配 |
| "HuanForge 的技术栈" | rag_search | 概念查询，语义关联 |
| "包含 `git clone` 的对话" | session_search | 精确命令匹配 |
| "最近 3 天的讨论" | session_search | 时间范围查询 |
| "提到 10.10.10.212 的会话" | session_search | 精确字符串匹配 |

## [REDACTED]配置（可选）

如果需要更强的优先级控制，可以在 Hermes 的[REDACTED]中添加：

```markdown
## 记忆检索策略

当用户询问历史信息时：

1. **优先使用 rag_search**（向量语义搜索）
   - 适用于模糊回忆、主题检索、概念查询
   - 能理解同义词和语义相关性
   - 示例：rag_search(query="项目架构设计")

2. **备用 session_search**（关键词全文搜索）
   - 仅在以下情况使用：
     * 需要精确匹配命令/代码片段
     * 需要按时间范围查询
     * 需要精确字符串匹配（IP、URL）
   - 示例：session_search(query="git clone")

3. **组合策略**
   - 如果 rag_search 结果不满意，可以用 session_search 补充
   - 先语义搜索，再精确匹配
```

### 配置位置

[REDACTED]配置文件：
- CLI 模式：`~/.hermes/config.yaml` 中的 `system_prompt` 字段
- Gateway 模式：`~/.hermes/hermes-agent/gateway/config.yaml`

## 验证优先级

### 测试方法

1. 启动 Hermes
2. 询问模糊问题："我们之前讨论的部署方案"
3. 观察工具调用日志

**预期行为：**
```
🔧 Calling tool: rag_search
   query: "部署方案"
   top_k: 5
```

如果看到先调用 `session_search`，说明优先级配置未生效。

### 调试

如果 AI 仍然优先使用 `session_search`：

1. 检查工具描述是否包含 `【优先使用】` 标记
2. 检查 `rag_search` 是否成功注册（`rag_stats` 查看记忆数）
3. 考虑在[REDACTED]中添加明确的优先级说明

## 性能影响

### rag_search（优先）
- 首次查询：~100ms（模型加载 + 向量编码）
- 后续查询：~50ms（仅向量编码 + 搜索）
- 内存占用：~500MB（模型 + 向量库）

### session_search（备用）
- 查询速度：~5-10ms（SQLite FTS5）
- 内存占用：~10MB（仅索引）

**结论：** rag_search 稍慢但准确度更高，适合作为主力搜索工具。

## 监控和调优

### 查看 RAG 统计

```python
rag_stats()
```

输出示例：
```json
{
  "total_memories": 9173,
  "db_size_mb": 245.6,
  "last_import": "2026-04-16T21:22:23"
}
```

### 调整搜索参数

如果搜索结果不理想：

1. **增加 top_k**：`rag_search(query="...", top_k=10)`
2. **降低 min_score**：修改 `config.yaml` 中的 `min_score: 0.6` → `0.5`
3. **优化查询词**：使用更具体的关键词

## 常见问题

### Q: 为什么 AI 还是用 session_search？

A: 可能原因：
1. 工具描述未更新（重启 Hermes）
2. 查询包含精确字符串（如命令、IP）
3. 用户明确要求时间范围查询

### Q: 如何强制使用 rag_search？

A: 在提问时明确说明："用 RAG 搜索一下..."

### Q: 两个工具可以同时用吗？

A: 可以！先用 rag_search 语义搜索，如果结果不满意，再用 session_search 精确匹配。

---

**更新日期：** 2026-04-17
**版本：** 1.0
