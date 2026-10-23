
import time
import gradio as gr
import spaces

from generation.rag_chain import MedicalRAGChain


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PAGE_TITLE = "Medical RAG Chatbot"

rag_chain = None


# ---------------------------------------------------------
# ZEROGPU COMPATIBILITY
# ---------------------------------------------------------

@spaces.GPU(duration=10)
def zerogpu_check():
    """
    Minimal ZeroGPU compatibility handler.
    This function does not perform medical inference.
    """
    return "ZeroGPU handler registered."


# ---------------------------------------------------------
# RAG INITIALIZATION
# ---------------------------------------------------------

def load_rag_chain():
    global rag_chain

    if rag_chain is None:
        rag_chain = MedicalRAGChain()

    return rag_chain


# ---------------------------------------------------------
# SOURCE FORMATTING
# ---------------------------------------------------------

def format_sources(sources):
    """Format retrieved documents for the source panel."""

    if not sources:
        return "No source documents were returned for this response."

    sections = []

    for idx, src in enumerate(sources, 1):
        title = src.get("title", "Unknown")
        source = src.get("source", "Unknown")
        score = src.get("score")

        score_text = (
            f"{score:.4f}"
            if isinstance(score, (int, float))
            else "Unavailable"
        )

        excerpt = (src.get("text") or "").strip()

        if len(excerpt) > 300:
            excerpt = excerpt[:300] + "..."

        sections.append(
            f"### [{idx}] {title}\n"
            f"- **Source:** {source}\n"
            f"- **Distance score:** `{score_text}`\n\n"
            f"> {excerpt or 'No excerpt available.'}"
        )

    return "\n\n---\n\n".join(sections)


def format_answer(answer, sources, elapsed, model_used):
    """Format the answer and its source citations."""

    result = answer or "No answer was returned."

    if sources:
        result += "\n\n---\n\n### 📚 Retrieved Medical Sources\n"

        for idx, src in enumerate(sources, 1):
            result += (
                f"\n**[{idx}] {src.get('title', 'Unknown')}** "
                f"(`{src.get('source', 'Unknown')}`)\n\n"
            )

    result += f"\n\n*Generated in {elapsed:.2f} seconds*"

    if model_used:
        result += f" · Model: `{model_used}`"

    return result


# ---------------------------------------------------------
# CHAT HANDLER
# ---------------------------------------------------------

def respond(message, history):
    """
    Process the current question through the existing RAG pipeline.
    Conversation history is displayed but is not passed to the RAG chain.
    """

    message = (message or "").strip()
    updated_history = list(history or [])

    if not message:
        return (
            updated_history,
            "Enter a question to begin.",
            ""
        )

    updated_history.append({
        "role": "user",
        "content": message
    })

    start_time = time.time()

    try:
        chain = load_rag_chain()
        result = chain.generate_response(message)

        elapsed = time.time() - start_time
        sources = result.get("sources") or []

        answer = format_answer(
            answer=result.get("answer", ""),
            sources=sources,
            elapsed=elapsed,
            model_used=result.get("model_used", "")
        )

        source_panel = format_sources(sources)

    except Exception:
        answer = (
            "Sorry, I couldn't process this question. "
            "Please try again later. If the problem persists, "
            "the Space owner should inspect the application logs "
            "for configuration or dependency errors."
        )

        source_panel = (
            "Sources are unavailable because the request failed."
        )

    updated_history.append({
        "role": "assistant",
        "content": answer
    })

    return updated_history, source_panel, ""


# ---------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------

with gr.Blocks(title=PAGE_TITLE) as demo:

    gr.Markdown(
        "# 🩺 AI Medical Assistant\n"
        "Powered by RAG, MedQuAD, MedlinePlus, FAISS and Groq.\n\n"
        "**Educational information only.** This chatbot does not "
        "provide a diagnosis or replace professional medical care."
    )

    # Register a ZeroGPU-decorated event handler.
    # The hidden button is not triggered automatically.
    zerogpu_button = gr.Button(
        "ZeroGPU compatibility handler",
        visible=False
    )

    zerogpu_status = gr.Textbox(visible=False)

    zerogpu_button.click(
        fn=zerogpu_check,
        inputs=[],
        outputs=[zerogpu_status]
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=480
    )

    with gr.Row():
        message = gr.Textbox(
            placeholder="Ask a health-related question...",
            label="Your question",
            scale=8,
            lines=2
        )

        send = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )

    clear = gr.Button("Clear conversation")

    with gr.Accordion(
        "📚 View Retrieved Medical Sources for the Latest Answer",
        open=False
    ):
        sources_panel = gr.Markdown(
            "Retrieved sources will appear here after your first question."
        )

    gr.Examples(
        examples=[
            "What are common warning signs of type 2 diabetes?",
            "What is high blood pressure?",
            "What are common symptoms of asthma?"
        ],
        inputs=message
    )

    message.submit(
        fn=respond,
        inputs=[message, chatbot],
        outputs=[chatbot, sources_panel, message]
    )

    send.click(
        fn=respond,
        inputs=[message, chatbot],
        outputs=[chatbot, sources_panel, message]
    )

    clear.click(
        fn=lambda: (
            [],
            "Sources will appear here after your next question.",
            ""
        ),
        inputs=[],
        outputs=[chatbot, sources_panel, message]
    )

    demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()


# import streamlit as st
# import time
# from generation.rag_chain import MedicalRAGChain

# # Page Configuration
# st.set_page_config(
#     page_title="Medical RAG Chatbot",
#     page_icon="🩺",
#     layout="wide"
# )

# st.title("🩺 AI Medical Assistant")
# st.caption("Powered by RAG (MedlinePlus + MedQuAD), FAISS, and Groq LLM")

# # Cache RAG Chain initialization to load vectors & embeddings only once
# @st.cache_resource
# def load_rag_chain():
#     return MedicalRAGChain()

# with st.spinner("Loading medical knowledge base and models..."):
#     rag_chain = load_rag_chain()

# # Initialize Chat History
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display Chat History
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
        
#         # If assistant response has sources attached, display them in an expander
#         if "sources" in message and message["sources"]:
#             with st.expander("📚 View Retrieved Medical Sources"):
#                 for idx, src in enumerate(message["sources"], 1):
#                     st.markdown(f"**[{idx}] {src['title']}** (`{src['source']}` | Distance Score: `{src['score']:.4f}`)")
#                     st.caption(f'"{src["text"][:300]}..."')
#                     st.divider()

# # User Input
# if prompt := st.chat_input("Ask a health-related question..."):
#     # Display User Message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate Assistant Response
#     with st.chat_message("assistant"):
#         with st.spinner("Searching medical database & generating response..."):
#             start_time = time.time()
#             result = rag_chain.generate_response(prompt)
#             latency = time.time() - start_time

#             st.markdown(result["answer"])
#             st.caption(f"⚡ Generated in {latency:.2f} seconds using `{result.get('model_used', 'LLM')}`")

#             # Display Citations / Sources
#             if result["sources"]:
#                 with st.expander("📚 View Retrieved Medical Sources"):
#                     for idx, src in enumerate(result["sources"], 1):
#                         st.markdown(f"**[{idx}] {src['title']}** (`{src['source']}` | Distance Score: `{src['score']:.4f}`)")
#                         st.caption(f'"{src["text"][:300]}..."')
#                         st.divider()

#     # Save to Session State
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result["answer"],
#         "sources": result["sources"]
#     })
