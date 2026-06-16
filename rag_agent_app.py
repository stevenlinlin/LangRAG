"""
智能文档问答助手 - 改进版
- Agent 状态中显式传递 knowledge_base
- 增加最大迭代步数限制
- 全局共享 SQLite checkpointer
- 支持清空知识库
- 优化中文分块分隔符
- 工具返回结果长度限制
"""

import os
import uuid
import tempfile
from typing import Annotated, List, Literal, TypedDict

import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage,SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, add_messages
from zhipuai import ZhipuAI

# ---------- 1. 配置 ----------
st.set_page_config(page_title="智能文档问答助手", layout="wide")

ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY", "ec05d14797c247c59bf2c39b492a8ca5.VAnWH70LZFycf29Q")
if ZHIPU_API_KEY == "你的API_KEY":
    st.error("请先在代码中设置你的智谱 API Key，或配置环境变量 ZHIPUAI_API_KEY")
    st.stop()

# 模型
llm = ChatZhipuAI(api_key=ZHIPU_API_KEY, model="glm-4-flash", temperature=0.2)
embed_client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 持久化目录
DB_DIR = "./data"
os.makedirs(DB_DIR, exist_ok=True)
CHECKPOINT_DB = os.path.join(DB_DIR, "checkpoints.db")
CHROMA_DIR = os.path.join(DB_DIR, "chroma_db")

# 最大迭代步数（防止无限循环）
MAX_STEPS = 5

# ---------- 2. 向量数据库操作 ----------
import chromadb
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client

def get_retriever(collection_name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def add_documents_to_knowledge(collection_name: str, docs: List[str]):
    """将文档分块、嵌入并存入指定 collection"""
    if not docs:
        return
    collection = get_retriever(collection_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))
    if not chunks:
        return
    # 批量 embedding
    resp = embed_client.embeddings.create(model="embedding-2", input=chunks)
    embeddings = [item.embedding for item in resp.data]
    ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)

def retrieve(collection_name: str, query: str, top_k: int = 3) -> str:
    """检索最相关的文档片段，返回限制长度的字符串"""
    collection = get_retriever(collection_name)
    if collection.count() == 0:
        return ""
    resp = embed_client.embeddings.create(model="embedding-2", input=[query])
    q_emb = resp.data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    if results and results['documents']:
        combined = "\n\n".join(results['documents'][0])
        if len(combined) > 500:
            combined = combined[:500] + "...(内容过长，已截断)"
        return combined
    return ""

def clear_knowledge_base(collection_name: str):
    """删除整个知识库集合"""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass  # 集合不存在

# ---------- 3. 内部检索函数（供工具调用）----------
def _search_knowledge(kb_name: str, query: str) -> str:
    if not kb_name:
        return "当前会话没有关联的知识库，请先上传文档。"
    result = retrieve(kb_name, query)
    if not result:
        return "未找到相关信息。"
    return result

# ---------- 4. Agent 状态定义 ----------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    knowledge_base: str   # 当前会话的知识库集合名
    step: int             # 当前迭代步数

# ---------- 5. 节点函数 ----------
# 在文件开头添加系统消息模板
SYSTEM_PROMPT = """你是文档问答助手。你必须严格遵循以下规则：
1. 对于用户的每个问题，**必须先调用 search_knowledge 工具**从知识库中检索相关信息。
2. 根据工具返回的内容生成最终回答。如果返回“未找到相关信息”或空字符串，则回答“知识库中没有相关内容，请先上传文档或换一个问题”。
3. 绝对不要使用你自己记忆中的知识来回答，只能基于工具返回的检索结果。
4. 如果知识库未关联（工具返回“没有关联的知识库”），请提示用户先上传文档。
"""

def llm_node(state: AgentState):
    """调用 LLM，绑定工具，并强制使用系统提示"""
    # 构建消息列表：系统消息 + 历史消息
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "从当前会话的知识库中检索与问题相关的信息。当需要查找文档内容时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "step": state.get("step", 0) + 1}

def tool_node(state: AgentState):
    """执行工具调用（显式传递 knowledge_base）"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    results = []
    for tc in tool_calls:
        if tc["name"] == "search_knowledge":
            query = tc["args"]["query"]
            kb_name = state.get("knowledge_base", "")
            result = _search_knowledge(kb_name, query)
            results.append((tc["id"], result))
        else:
            results.append((tc["id"], f"未知工具: {tc['name']}"))
    tool_messages = [ToolMessage(content=res, tool_call_id=tid) for tid, res in results]
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """条件边：如果需要工具调用且未超过最大步数，则继续，否则结束"""
    if state.get("step", 0) >= MAX_STEPS:
        return "__end__"
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "__end__"

# ---------- 6. 构建图（带 checkpointer）----------
def build_agent(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)

# ---------- 7. 全局 checkpointer（单例）----------
import atexit
_global_checkpointer = None

def get_global_checkpointer():
    global _global_checkpointer
    if _global_checkpointer is None:
        import sqlite3
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        _global_checkpointer = SqliteSaver(conn)
        atexit.register(lambda: conn.close())
    return _global_checkpointer

# ---------- 8. Streamlit UI 状态管理 ----------
def init_session():
    """初始化会话状态"""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.active_collection = f"kb_{st.session_state.thread_id}"
        st.session_state.messages = []
        st.session_state.checkpointer = get_global_checkpointer()
        st.session_state.graph = build_agent(st.session_state.checkpointer)
        st.session_state.uploaded = False

def add_user_message(content):
    st.session_state.messages.append({"role": "user", "content": content})

def add_ai_message(content):
    st.session_state.messages.append({"role": "assistant", "content": content})

def process_question(question):
    """处理用户问题，调用 LangGraph"""
    add_user_message(question)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "knowledge_base": st.session_state.active_collection,
        "step": 0,
    }
    try:
        final_state = st.session_state.graph.invoke(initial_state, config=config)
        last_msg = final_state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            answer = last_msg.content
        else:
            answer = "抱歉，无法生成回答。"
    except Exception as e:
        answer = f"处理出错: {e}"
    add_ai_message(answer)
    st.rerun()

def handle_upload(uploaded_file):
    """处理上传的文件，存储到知识库"""
    if uploaded_file is None:
        return
    # 读取文本
    if uploaded_file.type == "text/plain":
        content = uploaded_file.read().decode("utf-8")
        docs = [content]
    elif uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        docs = [p.page_content for p in pages]
        os.unlink(tmp_path)
    else:
        st.error("不支持的文件格式，请上传 TXT 或 PDF")
        return
    # 添加到知识库
    with st.spinner("正在处理文档，请稍候..."):
        add_documents_to_knowledge(st.session_state.active_collection, docs)
    st.session_state.uploaded = True
    st.success("文档已成功添加到知识库！")

# ---------- 9. 主界面 ----------
def main():
    st.title("📚 智能文档问答助手")
    st.markdown("上传文档，然后向我提问，我会从文档中找到答案。")

    init_session()

    with st.sidebar:
        st.header("🗂️ 会话")
        st.caption(f"会话ID: {st.session_state.thread_id[:8]}...")
        if st.button("➕ 新建会话"):
            # 清空当前 session_state，刷新页面
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.divider()
        st.header("📄 知识库")
        uploaded_file = st.file_uploader("上传文档（TXT/PDF）", type=["txt", "pdf"])
        if uploaded_file:
            handle_upload(uploaded_file)
        if st.button("🗑️ 清空知识库"):
            clear_knowledge_base(st.session_state.active_collection)
            st.session_state.uploaded = False
            st.success("知识库已清空")
            st.rerun()
        if st.session_state.get("uploaded", False):
            st.success("知识库已准备就绪")
        else:
            st.info("请先上传文档")

    # 聊天界面
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入问题"):
        process_question(prompt)

if __name__ == "__main__":
    main()