import streamlit as st
from src.ingestion import load_files_from_repo
from src.chunking import chunk_all_files
from src.embeddings import build_index
from src.llm import QAEngine
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import get_reranker

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Code Q&A Bot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1f35 50%, #0E1117 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #4F8BF9, #7B68EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .sub-header {
        color: #8892b0;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1E2130, #252a40);
        border: 1px solid #4F8BF9;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }

    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4F8BF9;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Chat messages */
    .stChatMessage {
        background: #1E2130 !important;
        border-radius: 12px !important;
        border: 1px solid #2d3250 !important;
        margin: 8px 0 !important;
    }

    /* Source expander */
    .streamlit-expanderHeader {
        background: #1E2130 !important;
        border-radius: 8px !important;
        color: #4F8BF9 !important;
    }

    /* Input box */
    .stChatInput input {
        background: #1E2130 !important;
        border: 1px solid #4F8BF9 !important;
        border-radius: 12px !important;
        color: #FAFAFA !important;
    }

    /* Sidebar */
    .css-1d391kg {
        background: #1E2130 !important;
    }

    /* Success/info boxes */
    .stSuccess {
        background: #1a2e1a !important;
        border-left: 4px solid #00ff88 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("engine", None),
    ("repo_name", None),
    ("chunks", None),
    ("indexed", False),
    ("files_count", 0),
    ("chunks_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────
def extract_repo_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def index_repo(github_url: str):
    repo_name = extract_repo_name(github_url)

    with st.spinner(f"📥 Cloning {repo_name}..."):
        files = load_files_from_repo(github_url)
    st.session_state.files_count = len(files)
    st.sidebar.success(f"✅ Loaded {len(files)} files")

    with st.spinner(f"✂️ Chunking {len(files)} files..."):
        chunks = chunk_all_files(files)
    st.session_state.chunks_count = len(chunks)
    st.sidebar.success(f"✅ Created {len(chunks)} chunks")

    with st.spinner("🔢 Building vector index..."):
        build_index(chunks, repo_name)
    st.sidebar.success("✅ Index built!")

    with st.spinner("🤖 Initializing QA engine..."):
        engine = QAEngine(repo_name=repo_name, all_chunks=chunks)

    st.session_state.engine = engine
    st.session_state.repo_name = repo_name
    st.session_state.chunks = chunks
    st.session_state.indexed = True
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <span style='font-size:2.5rem'>🔍</span>
        <h2 style='color:#4F8BF9; margin:4px 0;'>Code Q&A Bot</h2>
        <p style='color:#8892b0; font-size:0.8rem;'>Powered by RAG + Groq LLaMA 3.1</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📦 Index a Repository")
    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/pallets/flask",
        label_visibility="collapsed",
    )

    if st.button("🚀 Index Repository", type="primary", use_container_width=True):
        if repo_url.strip():
            try:
                index_repo(repo_url.strip())
                st.success(f"✅ Ready! Ask questions about **{st.session_state.repo_name}**")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Please enter a GitHub URL")

    st.divider()

    if st.session_state.indexed:
        st.markdown("### 📊 Index Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-number'>{st.session_state.files_count}</div>
                <div class='stat-label'>Files</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-number'>{st.session_state.chunks_count}</div>
                <div class='stat-label'>Chunks</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='stat-card' style='margin-top:8px'>
            <div class='stat-number' style='font-size:1rem'>
                {st.session_state.repo_name}
            </div>
            <div class='stat-label'>Active Repository</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.engine:
                    st.session_state.engine.clear_memory()
                st.rerun()
        with col2:
            if st.button("🔄 New Repo", use_container_width=True):
                for key in ["indexed", "engine", "repo_name", "chunks", "messages"]:
                    st.session_state[key] = None if key != "messages" and key != "indexed" else ([] if key == "messages" else False)
                st.rerun()

    st.divider()
    st.markdown("""
    <div style='color:#8892b0; font-size:0.75rem; text-align:center;'>
        <b>Tech Stack</b><br>
        LangChain · ChromaDB · Groq<br>
        AST Chunking · Hybrid Search<br>
        Cross-Encoder Re-ranking
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────
if not st.session_state.indexed:
    st.markdown("<h1 class='main-header'>🔍 Code Q&A Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Ask natural language questions about any GitHub codebase</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stat-card'>
            <div style='font-size:2rem'>🧠</div>
            <div style='color:#4F8BF9; font-weight:600; margin:8px 0;'>AST-Aware Chunking</div>
            <div style='color:#8892b0; font-size:0.85rem;'>
                Splits code at function and class boundaries for precise retrieval
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='stat-card'>
            <div style='font-size:2rem'>🔎</div>
            <div style='color:#4F8BF9; font-weight:600; margin:8px 0;'>Hybrid Search</div>
            <div style='color:#8892b0; font-size:0.85rem;'>
                Combines vector similarity and BM25 keyword search for best results
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='stat-card'>
            <div style='font-size:2rem'>⚡</div>
            <div style='color:#4F8BF9; font-weight:600; margin:8px 0;'>Re-Ranking</div>
            <div style='color:#8892b0; font-size:0.85rem;'>
                Cross-encoder re-ranks top results for maximum answer accuracy
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Example Questions")

    examples = [
        "How does Flask handle URL routing?",
        "Where is the login_required decorator defined?",
        "How do blueprints work in Flask?",
        "What is the application context used for?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

else:
    st.markdown(
        f"<h2 style='color:#4F8BF9'>💬 Chatting with <span style='color:#7B68EE'>{st.session_state.repo_name}</span></h2>",
        unsafe_allow_html=True
    )

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📁 View Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f"📄 `{src['file']}` · lines **{src['lines']}** · "
                            f"{src['type']}: `{src['name']}`"
                        )

    # Chat input
    if prompt := st.chat_input("Ask anything about the codebase..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                for token in st.session_state.engine.answer_stream(
                    prompt, retrieval_k=10, rerank_k=3
                ):
                    full_response += token
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                # Get sources
                retrieved = hybrid_search(
                    prompt,
                    st.session_state.repo_name,
                    st.session_state.chunks,
                    top_k=10,
                )
                reranker = get_reranker()
                reranked = reranker.rerank_chunks_only(prompt, retrieved, top_k=3)
                sources = []
                for chunk in reranked:
                    name = chunk.function_name or chunk.class_name or "block"
                    sources.append({
                        "file": chunk.file_path,
                        "lines": f"{chunk.start_line}-{chunk.end_line}",
                        "name": name,
                        "type": chunk.chunk_type,
                    })

                if sources:
                    with st.expander("📁 View Sources"):
                        for src in sources:
                            st.markdown(
                                f"📄 `{src['file']}` · lines **{src['lines']}** · "
                                f"{src['type']}: `{src['name']}`"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

            except Exception as e:
                err = f"❌ Error: {e}"
                placeholder.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                    "sources": [],
                })