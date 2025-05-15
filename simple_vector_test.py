#!/usr/bin/env python
"""
Simplified test script for vector search with lower threshold.
This script avoids the import chain issues by directly importing only what's needed.
"""
import os
import json
import sys
import typing as t
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Load environment variables first
load_dotenv()

# Constants
EMBED_MODEL = "text-embedding-3-small"  # 1536-d 
_TOPK_DEFAULT = 5

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    sys.exit(1)
    
client = OpenAI(api_key=api_key)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    sys.exit(1)
    
supabase = create_client(supabase_url, supabase_key)

@lru_cache(maxsize=100)
def embed_text(text: str) -> t.List[float]:
    """Generate an embedding for the given text."""
    if not text.strip():
        raise ValueError("Cannot embed empty text")
    
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

def test_vector_search():
    """Test vector search with a lower threshold."""
    print("Testing vector search with lower threshold...")
    
    # Generate embedding for the query
    query = "test document"
    q_emb = embed_text(query)
    
    # Use the modified stored procedure with explicit threshold parameter
    sql = "SELECT * FROM search_vectors($1, $2, $3)"
    
    # Set a very low threshold to get more results
    threshold = 0.0
    params = [q_emb, _TOPK_DEFAULT, threshold]
    
    print(f"Using search_vectors with threshold={threshold}")
    response = supabase.rpc(
        "exec_sql", 
        {"sql": sql, "params": params}
    ).execute()
    
    # Process results
    if hasattr(response, 'data') and response.data:
        rows = response.data
        print(f"\nFound {len(rows)} results with vector search")
        
        # Print the scores to see the similarity values
        for i, row in enumerate(rows):
            print(f"Result {i+1}: doc_id={row.get('doc_id')}, score={row.get('score')}")
            
        # Print full details of the first result
        if rows:
            print("\nFirst result details:")
            print(json.dumps(rows[0], indent=2))
    else:
        print("No results found with vector search")

if __name__ == "__main__":
    test_vector_search()
