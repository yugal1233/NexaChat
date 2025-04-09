import argparse
import streamlit as st
import ollama
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Argument Parser
parser = argparse.ArgumentParser(description="PDF to Milvus Vector Store with AI Chat & Semantic Search")
parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
#ollama model argument
parser.add_argument("--ollama_model", type=str, default="gemma3", help="Ollama model")
parser.add_argument("--embedding_model", type=str, default="nomic-embed-text", help="Embedding model")
args = parser.parse_args()

# Connect to Milvus
connections.connect("default", host=args.host, port=args.port)

# Initialize embedding model
# get_path = os.getcwd()
# model_path = os.path.join(get_path, 'models', 'embedding_model')
# model = SentenceTransformer(model_path)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 Ollama AI Model
if args.ollama_model:
    OLLAMA_MODEL = args.ollama_model
else:
    OLLAMA_MODEL = "llama3:70b"

if args.embedding_model:
    embedding_model = args.embedding_model
else:
    embedding_model = "nomic-embed-text"

def get_all_collections():
    """Retrieve a list of all Milvus collections."""
    return utility.list_collections()

def initialize_milvus(collection_name):
    """Ensure the collection exists and initialize it if necessary."""
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, description="PDF text embeddings")
    collection = Collection(collection_name, schema)
    collection.create_index("embedding", {"metric_type": "L2"})
    collection.load()
    return collection

def chunk_text(text, chunk_size=512, overlap=50):
    """Split text into optimized chunks using LangChain RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

def upload_pdf(collection_name, file):
    """Extracts text from PDF and stores chunks in Milvus."""
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    doc = fitz.open(file_path)
    all_chunks, all_embeddings = [], []
    
    for page in doc:
        text = page.get_text("text")
        chunks = chunk_text(text)
        embeddings = ollama.embed(input = chunks, model=embedding_model)
        embeddings = embeddings["embeddings"]
        # print(embeddings)
        # embeddings = model.encode(chunks).tolist()
        
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
    print(len(all_chunks))
    print(len(all_embeddings))
    
    if all_chunks and all_embeddings:
        collection = initialize_milvus(collection_name)
        collection.insert([all_chunks, all_embeddings])
        return f"✅ {len(all_chunks)} chunks stored in Milvus!"
    
    return "⚠️ No text extracted from PDF."

def check_data(collection_name):
    """Retrieve and display stored text and embeddings from Milvus."""
    if not utility.has_collection(collection_name):
        return ["❌ Collection does not exist. Please upload a PDF first."]
    
    collection = Collection(collection_name)
    collection.load()
    
    result = collection.query(expr="", output_fields=["id", "text", "embedding"], limit=5)
    
    if result:
        return [
            f"🆔 ID: {row['id']}\n📄 Text: {row['text'][:100]}...\n🧠 Embedding: {str(row['embedding'][:5])}..."  
            for row in result
        ]  # Showing only first 5 values of the embedding for readability
    
    return ["❌ No data found in the collection."]

def delete_collection(collection_name):
    """Delete a specific Milvus collection."""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        return f"✅ Collection '{collection_name}' deleted."
    return f"❌ Collection '{collection_name}' does not exist."

def delete_all_collections():
    """Delete all collections in Milvus."""
    collections = get_all_collections()
    for coll in collections:
        utility.drop_collection(coll)
    return "✅ All collections deleted."

def semantic_search(query, top_k=5):
    """Perform semantic search across all collections."""
    collections = get_all_collections()
    results = []
    
    # query_embedding = model.encode([query]).tolist()
    query_embedding = ollama.embed(model = embedding_model, input = query)
    query_embedding = query_embedding["embeddings"]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    for coll_name in collections:
        collection = Collection(coll_name)
        collection.load()
        search_results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        for hits in search_results:
            results.extend([f"📄 {hit.entity.get('text')[:200]}... (from {coll_name})" for hit in hits])
    
    return results if results else ["❌ No relevant results found."]

def ai_chat_with_ollama(user_query):
    """Use Ollama to generate AI responses based on semantic search results."""
    search_results = semantic_search(user_query)
    
    if search_results and "❌ No relevant results found." not in search_results:
        context_text = "\n".join(search_results[:10])  # Use top 3 relevant results
    else:
        context_text = "No relevant context found. Answer based on general knowledge."
    
    ollama_prompt = f"""
    You are an AI assistant with knowledge of uploaded PDF documents.
    Context: {context_text}
    
    User Query: {user_query}
    
    Provide a relevant and concise answer.
    """
    
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": ollama_prompt}])
    
    return response.get("message", {}).get("content", "⚠️ AI could not generate a response.")

# 🔹 Streamlit UI
st.title("📄 PDF to Milvus with AI Chat & Semantic Search")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload PDF", "View Stored Data", "Delete Collection", "Semantic Search", "AI Chat"])

if page == "Upload PDF":
    st.header("📤 Upload a PDF File")
    collection_name = st.text_input("Enter collection name:")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file:
        st.write("✅ File received! Processing...")
        status = upload_pdf(collection_name, uploaded_file)
        st.success(status)

elif page == "View Stored Data":
    st.header("📂 Stored Data in Milvus")
    
    collections = get_all_collections()
    
    if collections:
        collection_name = st.selectbox("Select a collection:", collections)
        
        if collection_name:
            sample_data = check_data(collection_name)
            for item in sample_data:
                st.write(item)
    else:
        st.warning("❌ No collections found. Please upload a PDF first.")

elif page == "Delete Collection":
    st.header("🗑️ Delete Collections")
    action = st.radio("Choose an action:", ["Delete Specific Collection", "Delete All Collections"])
    
    if action == "Delete Specific Collection":
        collections = get_all_collections()
        
        if collections:
            collection_name = st.selectbox("Select a collection to delete:", collections)
            if st.button("Delete Collection"):
                result = delete_collection(collection_name)
                st.success(result)
        else:
            st.warning("❌ No collections available to delete.")

    if action == "Delete All Collections":
        if st.button("Delete All Collections"):
            result = delete_all_collections()
            st.success(result)

elif page == "Semantic Search":
    st.header("🔍 Semantic Search in PDF Content")
    query = st.text_input("Enter search query:")
    
    if query:
        st.write("🔎 Searching across all collections...")
        results = semantic_search(query)
        for res in results:
            st.write(res)

elif page == "AI Chat":
    st.header("💬 AI Chat using Ollama")
    user_input = st.text_area("Ask something based on uploaded PDFs:")
    
    if st.button("Chat"):
        ai_response = ai_chat_with_ollama(user_input)
        st.write(f"🤖 **AI Response:** {ai_response}")