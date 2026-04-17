# Hermes Agent 技能集

这个目录包含 Hermes Agent 相关的技能文档。

## 可用技能

### 1. hermes-rag-complete-deployment

**Hermes RAG 完整部署指南**

从零开始部署 Hermes CPU RAG 记忆插件的完整流程。

**包含内容：**
- 第一步：安装 CPU RAG 插件
- 第二步：注册插件到 Hermes
- 第三步：优化切片策略（避免内容截断）
- 第四步：批量导入历史文档
- 第五步：配置自动增量导入
- 常见问题排查
- 性能基准与维护建议
- 进阶配置

**适用场景：**
- 新 Hermes 实例需要启用 RAG 记忆功能
- 需要导入大量历史文档到向量数据库
- 优化现有 RAG 切片策略避免内容截断
- 配置自动增量导入新对话记录

**前置要求：**
- Hermes Agent 已安装
- Python 3.8+ 环境
- 至少 2GB 可用磁盘空间

**查看详情：** [hermes-rag-complete-deployment/SKILL.md](hermes-rag-complete-deployment/SKILL.md)

---

## 如何使用这些技能

### 方法 1: 复制到 Hermes skills 目录

```bash
# 复制单个技能
cp -r hermes-rag-complete-deployment ~/.hermes/skills/mlops/

# 或复制整个 hermes 目录
cp -r hermes ~/.hermes/skills/
```

### 方法 2: 在 Hermes 中加载

```bash
# 启动 Hermes
hermes

# 在对话中
/skill view hermes-rag-complete-deployment
```

---

## 技能开发指南

如果你想贡献新的 Hermes 技能，请遵循以下格式：

### 文件结构

```
skill-name/
├── SKILL.md          # 主文档（必需）
├── references/       # 参考资料（可选）
├── templates/        # 配置模板（可选）
└── scripts/          # 辅助脚本（可选）
```

### SKILL.md 格式

```markdown
---
name: skill-name
description: 简短描述（一句话）
tags: [tag1, tag2, tag3]
---

# 技能标题

简介段落

## 适用场景

- 场景 1
- 场景 2

## 前置要求

- 要求 1
- 要求 2

---

## 步骤 1: ...

### 1.1 子步骤

具体操作...

## 常见问题排查

...
```

---

## 更多资源

- [Hermes Agent 官方文档](https://github.com/anthropics/hermes-agent)
- [Hermes Skills 开发指南](https://github.com/anthropics/hermes-agent/blob/main/docs/skills.md)
