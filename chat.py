import os
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_qdrant import QdrantVectorStore

# Load environment variables (make sure GOOGLE_API_KEY is set)
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
# 3. Initialize LLM (Gemini)
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # you can use gemini-1.5-pro also
)

# ---------------------------
# 4. User Query
# ---------------------------
user_query = input("Ask a question about the thesis draft: ")

# ---------------------------
# 5. Retrieve Relevant Docs
# ---------------------------
search_results = vector_db.similarity_search(
    query=user_query,
 # number of chunks to retrieve
)

if not search_results:
    print("No relevant information found in the document.")
    exit()

# ---------------------------
# 6. Build Context
# ---------------------------
context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('page_label', 'N/A')}\n"
    f"File Location: {result.metadata.get('source', 'N/A')}"
    for result in search_results
])

# ---------------------------
# 7. System Prompt
# ---------------------------
SYSTEM_PROMPT = f"""
You are a helpful AI assistant.

Answer the user question ONLY using the provided context.
If the answer is not in the context, say:
"I couldn't find this in the document."

Always include the page number in your answer.

Context:
{context}
"""

# ---------------------------
# 8. Final Prompt
# ---------------------------
final_prompt = f"""
{SYSTEM_PROMPT}

User Question:
{user_query}
"""

# ---------------------------
# 9. Generate Answer
# ---------------------------
response = llm.invoke(final_prompt)

# ---------------------------
# 10. Print Output
# ---------------------------
print("\n==============================")
print("Answer:\n")
print(response.content)
print("==============================")