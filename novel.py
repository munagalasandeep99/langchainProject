import os
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# ---------------------------
# 1. Embedding Model
# ---------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview"
)

# ---------------------------
# 2. Connect to Qdrant
# ---------------------------
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="ancient_godly_monarch",
)

# ---------------------------
# 3. Initialize LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

# ---------------------------
# 4. System Prompt for Novel
# ---------------------------
SYSTEM_PROMPT = """
You are an expert reader and analyst of the novel "Ancient Godly Monarch".

Your job is to answer questions about the novel's plot, characters, events,
power systems, and spoilers — using ONLY the provided context chunks.

Guidelines:
- Give detailed, spoiler-rich answers when asked. Don't hold back plot details.
- If a character's fate, power level, or relationship is mentioned in the context, include it.
- If multiple chunks mention the same character/event, synthesize them into one coherent answer.
- If the answer is NOT in the context, say: "This information isn't in the retrieved sections. Try rephrasing or asking about a specific character/event."
- Mention which part of the novel (early, mid, late chapters) the event occurs if inferable.
- Do NOT make up plot points. Stick strictly to the context.
"""

# ---------------------------
# 5. Conversational Loop
# ---------------------------
print("=" * 50)
print("  Ancient Godly Monarch — Spoiler Assistant")
print("  Type 'exit' or 'quit' to stop")
print("=" * 50)

chat_history = []  # stores (question, answer) pairs for context continuity

while True:
    print()
    user_query = input("Your Question: ").strip()

    if not user_query:
        continue
    if user_query.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # ---------------------------
    # 6. Retrieve Relevant Chunks
    # ---------------------------
    search_results = vector_db.similarity_search(
        query=user_query,
        k=8,  # fetch more chunks for novel — more context = better answers
    )

    if not search_results:
        print("\nNo relevant information found. Try a different question.")
        continue

    # ---------------------------
    # 7. Build Context from Chunks
    # ---------------------------
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(
            f"[Chunk {i}]\n"
            f"{result.page_content}\n"
            f"Source: {result.metadata.get('source', 'N/A')}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # ---------------------------
    # 8. Include Recent Chat History
    # ---------------------------
    history_text = ""
    if chat_history:
        recent = chat_history[-3:]  # only last 3 Q&A pairs to avoid token bloat
        history_text = "Previous conversation:\n" + "\n".join(
            [f"Q: {q}\nA: {a}" for q, a in recent]
        )

    # ---------------------------
    # 9. Build Final Prompt
    # ---------------------------
    final_prompt = f"""
{SYSTEM_PROMPT}

{history_text}

--- Retrieved Novel Context ---
{context}
--- End of Context ---

User Question: {user_query}
"""

    # ---------------------------
    # 10. Generate Answer
    # ---------------------------
    try:
        response = llm.invoke(final_prompt)
        answer = response.content

        print("\n" + "=" * 50)
        print("Answer:\n")
        print(answer)
        print("=" * 50)

        # Save to history
        chat_history.append((user_query, answer))

    except Exception as e:
        print(f"\nError generating answer: {e}")