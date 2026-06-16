# 📚 LangRAG：智能文档问答助手

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.0+-green)](https://github.com/langchain-ai/langgraph)

**LangRAG** 是一个基于 **LangGraph**、**RAG（检索增强生成）** 和 **Streamlit** 构建的智能文档问答系统。它允许用户上传 TXT 或 PDF 文档，通过自然语言提问，系统会从文档中精准检索最相关的信息，并结合大语言模型生成高质量的回答。

- 🧠 **智能决策**：采用 Agentic RAG 模式，LLM 自主判断是否需要检索知识库。
- 💾 **状态持久化**：支持多会话隔离，对话历史和知识库跨会话持久化。
- 🎨 **开箱即用**：提供友好的 Web 界面，无需任何前端知识即可上手。

> 本项目旨在展示如何将 **LangGraph** 的高级编排能力与 **RAG** 结合，构建一个生产级可用的智能问答应用。欢迎 Star、Fork 和提 Issue！

---

## ✨ 核心功能

| 功能 | 描述 |
|------|------|
| 📄 **文档上传** | 支持 TXT、PDF 格式，自动分块并生成向量存储。 |
| 🔍 **智能检索** | 基于 ChromaDB 向量数据库，快速定位相关文档片段。 |
| 🧠 **Agent 决策** | 通过 LangGraph 实现 ReAct 模式，LLM 自主决定是否调用检索工具。 |
| 💬 **多轮对话** | 支持对话上下文记忆，可连续追问，模型能结合历史对话回答。 |
| 🔁 **多会话管理** | 每个会话拥有独立的知识库和对话历史，互不干扰。 |
| 💾 **状态持久化** | 使用 SQLite Checkpointer 持久化会话状态，重启服务不丢失。 |
| 🌐 **Web 界面** | 基于 Streamlit 构建，交互直观，支持上传和聊天。 |

---

## 🛠️ 技术栈

| 组件 | 技术选型 |
|------|----------|
| **智能体编排** | [LangGraph](https://github.com/langchain-ai/langgraph)（有状态图、工具调用、检查点） |
| **LLM 基础** | [LangChain](https://github.com/langchain-ai/langchain) + 智谱 AI (`glm-4-flash`) |
| **向量存储** | [ChromaDB](https://www.trychroma.com/)（本地持久化） |
| **嵌入模型** | 智谱 AI `embedding-2`（1024 维） |
| **前端框架** | [Streamlit](https://streamlit.io/)（快速构建 Web UI） |
| **状态持久化** | SQLite（LangGraph Checkpointer） |
| **开发语言** | Python 3.11+ |

---

## 🚀 快速开始

### 前置条件
- Python 3.11 或更高版本
- 智谱 AI API Key（[免费申请](https://open.bigmodel.cn/)）
- Git（可选，用于克隆代码）

### 1. 克隆代码
```bash
git clone https://github.com/你的用户名/LangRAG.git
cd LangRAG