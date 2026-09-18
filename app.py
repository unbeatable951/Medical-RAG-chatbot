import streamlit as st
import time
from generation.rag_chain import MedicalRAGChain

# Page Configuration
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical Assistant")
st.caption("Powered by RAG (MedlinePlus + MedQuAD), FAISS, and Groq LLM")

# Cache RAG Chain initialization to load vectors & embeddings only once
@st.cache_resource
def load_rag_chain():
    return MedicalRAGChain()

with st.spinner("Loading medical knowledge base and models..."):
    rag_chain = load_rag_chain()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant response has sources attached, display them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Medical Sources"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**[{idx}] {src['title']}** (`{src['source']}` | Distance Score: `{src['score']:.4f}`)")
                    st.caption(f'"{src["text"][:300]}..."')
                    st.divider()

# User Input
if prompt := st.chat_input("Ask a health-related question..."):
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching medical database & generating response..."):
            start_time = time.time()
            result = rag_chain.generate_response(prompt)
            latency = time.time() - start_time

            st.markdown(result["answer"])
            st.caption(f"⚡ Generated in {latency:.2f} seconds using `{result.get('model_used', 'LLM')}`")

            # Display Citations / Sources
            if result["sources"]:
                with st.expander("📚 View Retrieved Medical Sources"):
                    for idx, src in enumerate(result["sources"], 1):
                        st.markdown(f"**[{idx}] {src['title']}** (`{src['source']}` | Distance Score: `{src['score']:.4f}`)")
                        st.caption(f'"{src["text"][:300]}..."')
                        st.divider()

    # Save to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })