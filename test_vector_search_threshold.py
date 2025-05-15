#!/usr/bin/env python
"""
Test script to debug vector search functionality with lower threshold.
"""
import os
import json
from dotenv import load_dotenv
from personal_assistant.vector_stores.supabase_store import SupabaseVectorStore, _embed_single

# Ensure environment variables are loaded
load_dotenv()

def main():
    """Run tests for the vector store with lower threshold."""
    print("Initializing vector store...")
    store = SupabaseVectorStore()

    # Test basic connectivity
    print("\nTesting basic database connection...")
    try:
        response = store.cli.from_("documents").select("id, doc_id").limit(5).execute()
        print(f"Basic connection test: {response.data}")
    except Exception as e:
        print(f"Basic connection error: {str(e)}")

    # Test direct vector search with lower threshold
    print("\nTesting vector search with lower threshold...")
    try:
        # Generate embedding for the query
        query = "test document"
        q_emb = _embed_single(query)
        
        # Use the modified stored procedure with explicit threshold parameter
        sql = "SELECT * FROM search_vectors($1, $2, $3)"
        
        # Set a very low threshold to get more results
        threshold = 0.0
        params = [q_emb, 5, threshold]
        
        print(f"Using search_vectors with threshold={threshold}")
        response = store.cli.rpc(
            "exec_sql", 
            {"sql": sql, "params": params}
        ).execute()
        
        print(f"Vector search response data: {json.dumps(response.data, indent=2) if hasattr(response, 'data') else 'No data'}")
        
        # Process results
        if hasattr(response, 'data') and response.data:
            rows = response.data
            print(f"\nFound {len(rows)} results with vector search")
            
            # Print the scores to see the similarity values
            for i, row in enumerate(rows):
                print(f"Result {i+1}: doc_id={row.get('doc_id')}, score={row.get('score')}")
        else:
            print("No results found with vector search")
            
    except Exception as e:
        print(f"Vector search error: {str(e)}")

if __name__ == "__main__":
    main()
