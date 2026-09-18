import os
from dotenv import load_dotenv

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

VECTOR_STORE_DIR = os.path.join(
    "data",
    "vector_store",
    "medical_index"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Current Groq production models
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b"
]


# ============================================================
# MEDICAL RAG CHAIN
# ============================================================

class MedicalRAGChain:

    def __init__(self):

        # ----------------------------------------------------
        # Initialize Embedding Model
        # ----------------------------------------------------

        print("Initializing Embedding Model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        # ----------------------------------------------------
        # Load FAISS Vector Store
        # ----------------------------------------------------

        print("Loading FAISS Vector Store...")

        self.vector_store = FAISS.load_local(
            VECTOR_STORE_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # ----------------------------------------------------
        # Initialize Groq Client
        # ----------------------------------------------------

        print("Initializing Groq Client...")

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is missing. "
                "Check your .env file!"
            )

        self.client = Groq(
            api_key=api_key
        )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3
    ):

        results = (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=top_k
            )
        )

        docs = []

        for doc, score in results:

            docs.append(
                {
                    "title": doc.metadata.get(
                        "title",
                        "Unknown"
                    ),

                    "source": doc.metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "text": doc.page_content,

                    "score": float(score)
                }
            )

        return docs


    # ========================================================
    # RESPONSE GENERATION
    # ========================================================

    def generate_response(
        self,
        query: str,
        top_k: int = 3
    ):

        # ----------------------------------------------------
        # 1. Retrieve relevant documents
        # ----------------------------------------------------

        context_docs = self.retrieve_context(
            query,
            top_k=top_k
        )

        # ----------------------------------------------------
        # 2. Format retrieved context
        # ----------------------------------------------------

        formatted_context = ""

        for idx, doc in enumerate(
            context_docs,
            1
        ):

            formatted_context += (
                f"--- Source {idx}: "
                f"{doc['title']} "
                f"({doc['source']}) ---\n"
            )

            formatted_context += (
                f"{doc['text']}\n\n"
            )

        # ----------------------------------------------------
        # 3. System Prompt
        # ----------------------------------------------------

        system_prompt = (
            "You are a helpful and accurate medical AI assistant. "

            "Answer the user's health-related question strictly "
            "using the retrieved medical context provided by the "
            "system. "

            "Do not introduce medical facts that are not supported "
            "by the retrieved context. "

            "If the retrieved context does not contain enough "
            "information to answer the question, clearly state "
            "that the available information is insufficient. "

            "Do not make a diagnosis or provide personalized "
            "medical treatment instructions. "

            "For medical concerns, advise the user to consult "
            "a qualified healthcare professional."
        )

        # ----------------------------------------------------
        # 4. User Prompt
        # ----------------------------------------------------

        user_prompt = (
            f"Context Information:\n\n"
            f"{formatted_context}"
            f"\nUser Question: {query}\n\n"
            "Answer the question using only the context above."
        )

        # ----------------------------------------------------
        # 5. Generate response
        # ----------------------------------------------------

        last_exception = None

        for model in GROQ_MODELS:

            try:

                print(
                    f"Attempting response generation "
                    f"using model: {model}..."
                )

                response = (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=model,

                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },

                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],

                        temperature=0.2,

                        max_tokens=800
                    )
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                return {
                    "query": query,
                    "answer": answer,
                    "sources": context_docs,
                    "model_used": model
                }

            except Exception as e:

                print(
                    f" -> Model {model} failed: {e}"
                )

                last_exception = e

                continue

        raise last_exception


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    rag = MedicalRAGChain()

    test_query = (
        "What are the early warning signs "
        "of type 2 diabetes?"
    )

    print(
        f"\nQuery: {test_query}\n"
    )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    output = rag.generate_response(
        test_query,
        top_k=3
    )

    # --------------------------------------------------------
    # Generated Answer
    # --------------------------------------------------------

    print("=" * 70)

    print(
        f"GENERATED ANSWER "
        f"(Model: {output['model_used']}):"
    )

    print("=" * 70)

    print(
        output["answer"]
    )

    # --------------------------------------------------------
    # Retrieved Sources
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "RETRIEVED SOURCES:"
    )

    print("=" * 70)

    for idx, src in enumerate(
        output["sources"],
        1
    ):

        print(
            f"\n[{idx}]"
        )

        print(
            f"Source : {src['source']}"
        )

        print(
            f"Title  : {src['title']}"
        )

        print(
            f"Score  : {src['score']:.4f}"
        )

        print(
            f"Text   : "
            f"{src['text'][:500]}..."
        )