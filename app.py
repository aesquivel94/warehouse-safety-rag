import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from google import genai

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
PAGE_TITLE = "Warehouse Safety & OSHA Assistant"
PAGE_ICON = "🛡️"
MODEL_NAME = "gemini-2.5-flash"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
SIMILARITY_SEARCH_K = 1

FALLBACK_RESPONSE = "Information not found in the provided SOP."

SYSTEM_PROMPT_TEMPLATE = """
You are a General Safety & OSHA AI Assistant. Answer the user's question ONLY using the provided context below.
If the information is not explicitly stated in the context, respond with: "{fallback_msg}"

Context:
{context}

Question:
{question}
"""

RAW_SOP_DATASET = """
EQUIPMENT INSPECTION: Perform daily forklift pre-operation checks. Verify brakes, steering, and data plate legibility. Follow Lockout/Tagout (LOTO) to de-energize machinery before service.
HAZARDOUS MATERIALS: Anhydrous ammonia systems over 10,000 lbs require Process Safety Management (PSM). Use transfer chutes to contain dust and prevent explosions.
RECORDKEEPING: Employers must maintain OSHA 300 logs, 300A summaries, and 301 incident reports for 3 calendar years.
"""

SAMPLE_QUERIES = [
    "Custom query...",
    "What safety checks are required before operating a forklift?",
    "What are the requirements for handling anhydrous ammonia?",
    "How long must OSHA 300 logs be maintained?",
    "What is the maximum penalty for a late safety audit report?"
]

# ==============================================================================
# CORE LOGIC
# ==============================================================================
@st.cache_resource
def initialize_vector_store() -> Chroma:
    """Chunk dataset and initialize vector store once in session memory."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(RAW_SOP_DATASET)
    embeddings = FastEmbedEmbeddings()
    return Chroma.from_texts(chunks, embeddings)


def generate_rag_response(user_query: str, vector_store: Chroma, api_key: str) -> str:
    """Perform similarity search and pass context to Gemini."""
    client = genai.Client(api_key=api_key)

    search_results = vector_store.similarity_search(user_query, k=SIMILARITY_SEARCH_K)
    retrieved_context = search_results[0].page_content if search_results else "No context available."

    formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        fallback_msg=FALLBACK_RESPONSE,
        context=retrieved_context,
        question=user_query
    )

    llm_response = client.models.generate_content(
        model=MODEL_NAME,
        contents=formatted_prompt,
    )
    return llm_response.text

# ==============================================================================
# USER INTERFACE
# ==============================================================================
def set_query(query_text: str):
    """Callback function to safely update text input value in session state."""
    st.session_state["user_query_input"] = query_text


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Grounded RAG Assistant for Equipment, Chemical Hazards, and OSHA Logs.")

    # Look for the API key in Streamlit Secrets (Cloud) or os.environ (Local/Colab)
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    vector_store = initialize_vector_store()

    # Initialize Session State Key for the Text Input
    if "user_query_input" not in st.session_state:
        st.session_state["user_query_input"] = ""

    # Sidebar Buttons
    st.sidebar.header("Sample Queries")
    st.sidebar.caption("Click a question below to auto-fill:")

    sample_questions = [
        "What safety checks are required before operating a forklift?",
        "What are the requirements for handling anhydrous ammonia?",
        "How long must OSHA 300 logs be maintained?",
        "What is the maximum penalty for a late safety audit report?"
    ]

    for idx, question in enumerate(sample_questions):
        st.sidebar.button(
            label=question,
            key=f"sample_btn_{idx}",
            on_click=set_query,
            args=(question,),
            use_container_width=True
        )

    # Main Input Field (Bound directly to session state key)
    user_query = st.text_input(
        "Ask a safety question:",
        key="user_query_input",
        placeholder="e.g., What checks are needed for forklifts?"
    )

    # Action Button
    if st.button("Submit Question", type="primary") and user_query:
        if not api_key:
            st.error("Error: GEMINI_API_KEY environment variable is missing.")
            return

        with st.spinner("Searching safety documents..."):
            response_text = generate_rag_response(user_query, vector_store, api_key)
            st.subheader("Response:")
            st.info(response_text)


if __name__ == "__main__":
    main()
