import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

txt_path = Path(__file__).parent / "ancient_godly_monarch.txt"

if not txt_path.exists():
    print(f"Error: Could not find {txt_path}.")
else:
    loader = TextLoader(file_path=str(txt_path), encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
    )

    BATCH_SIZE = 100

    try:
        client = QdrantClient(url="http://localhost:6333")

        # Fix 1: Replace deprecated recreate_collection
        if client.collection_exists("ancient_godly_monarch"):
            client.delete_collection("ancient_godly_monarch")

        # Fix 2: Correct dimensions → 3072 for gemini-embedding-2-preview
        client.create_collection(
            collection_name="ancient_godly_monarch",
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="ancient_godly_monarch",
            embedding=embedding_model,
        )

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vector_store.add_documents(batch)
            print(f"Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")

        print("Done! Successfully indexed ancient_godly_monarch to Qdrant.")

    except Exception as e:
        print(f"An error occurred: {e}")