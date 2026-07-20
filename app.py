import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# 1. Page Configuration
st.set_page_config(page_title="Medical RAG Assistant", page_icon="⚕️", layout="centered")
st.title("⚕️ Medical Healthcare RAG Assistant")
st.write("Ask any medical question. The AI will answer based strictly on verified documentation.")

# 2. Setup Groq Client
# 2. Setup Groq Client (SAFE VERSION)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Please configure your GROQ_API_KEY in the environment variables or Streamlit secrets.")
client = Groq(api_key=GROQ_API_KEY)

# 3. Load Resources (Cached)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index("data/vector_store/medical_index.faiss")
    with open("data/vector_store/chunk_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

try:
    model, index, metadata = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")

# 4. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your healthcare assistant. How can I help you today?"}
    ]

# 5. Display Past Messages from History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input Window at the Bottom
if user_query := st.chat_input("Type your medical question here..."):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Reviewing medical documentation..."):
            
            # Step A: Retrieve Context Chunks
            query_vector = model.encode([user_query])
            scores, indices = index.search(query_vector, k=3)
            
            retrieved_context = ""
            sources = []
            for idx in indices[0]:
                if idx < len(metadata):
                    chunk_data = metadata[idx]
                    retrieved_context += f"\n{chunk_data['text']}\n"
                    source_name = chunk_data.get('source', 'Unknown Documentation')
                    sources.append(source_name)

            # Step B: LLM Generation
            system_prompt = (
                "You are an expert medical AI assistant. Answer the user's question accurately "
                "based ONLY on the provided context chunks. If the context doesn't contain the answer, "
                "honestly state that you don't know, but provide general safe guidance. Always maintain a professional tone."
            )
            user_prompt = f"Context:\n{retrieved_context}\n\nQuestion: {user_query}"
            
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                )
                
                answer = chat_completion.choices[0].message.content
                
                # Append sources cleanly to the answer string
                unique_sources = ", ".join(set(sources))
                full_response = f"{answer}\n\n---\n*Sources consulted: {unique_sources}*"
                
                st.markdown(full_response)
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)