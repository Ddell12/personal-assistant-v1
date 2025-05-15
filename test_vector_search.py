# test_vector_search.py
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file")
    sys.exit(1)

# Now import after environment variables are loaded
from personal_assistant.vector_stores.supabase_store import SupabaseVectorStore

# Initialize store
store = SupabaseVectorStore()

# Test basic connectivity
print("Testing basic database connection...")
response = store.cli.from_("documents").select("id, doc_id").limit(1).execute()
print(f"Basic connection test: {response.data}")

# Test direct SQL execution without vector
print("\nTesting direct SQL execution...")
sql_test = """
SELECT doc_id, content FROM documents LIMIT 1
"""
response = store.cli.rpc("exec_sql", {"sql": sql_test, "params": []}).execute()
print(f"Direct SQL result: {response.data}")

# Create a test document with embedding if needed
print("\nCreating test document...")
store.upsert(
    doc_id="test_vector_doc",
    content="This is a test document for vector search.",
    metadata={"test": True}
)
print("Test document created")

# Test search function
print("\nTesting search function...")
results = store.search("test document", top_k=3)
print(f"Search results: {results}")