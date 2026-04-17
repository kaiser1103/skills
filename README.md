# Skills Repository

这是一个技能库仓库，用于存储和分享各种可复用的技术技能文档。

## 目录结构

```
skills/
├── hermes/                    # Hermes Agent 相关技能
│   └── hermes-rag-complete-deployment/   # RAG 完整部署指南
└── README.md
```

## Hermes 技能

### hermes-rag-complete-deployment

**完整的 Hermes RAG 插件部署指南**

从零开始部署 Hermes CPU RAG 记忆插件的端到端流程，包括：

- 插件安装与配置
- 向量数据库设置
- 切片策略优化（避免内容截断）
- 批量文档导入
- 自动增量导入配置
- 常见问题排查
- 性能优化建议

**适用场景：**
- 新 Hermes 实例需要启用 RAG 记忆功能
- 需要导入大量历史文档到向量数据库
- 优化现有 RAG 切片策略
- 配置自动增量导入新对话记录

**查看详情：** [hermes/hermes-rag-complete-deployment/SKILL.md](hermes/hermes-rag-complete-deployment/SKILL.md)

---

## 如何使用

### 在 Hermes Agent 中加载技能

```bash
# 方法 1: 直接复制到 skills 目录
cp -r hermes/hermes-rag-complete-deployment ~/.hermes/skills/mlops/

# 方法 2: 在 Hermes 对话中使用
/skill load hermes-rag-complete-deployment
```

### 查看技能内容

```bash
# 在 Hermes 中
/skill view hermes-rag-complete-deployment

# 或直接查看文件
cat hermes/hermes-rag-complete-deployment/SKILL.md
```

---

## 贡献

欢迎提交新的技能文档！请确保：

1. 遵循 Hermes Skill 格式（YAML frontmatter + Markdown）
2. 包含清晰的适用场景和前置要求
3. 提供可执行的步骤和命令
4. 添加常见问题排查部分
5. 不包含敏感信息（密钥、IP、个人信息等）

---

## 许可

本仓库中的技能文档采用 MIT 许可证。
