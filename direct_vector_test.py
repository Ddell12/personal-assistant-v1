#!/usr/bin/env python
"""
Direct test of vector search by using a document's own embedding to search.
This should definitely find at least the original document.
"""
import os
import json
import sys
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables first
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
if not supabase_url or not supabase_key:
    print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    sys.exit(1)
    
supabase = create_client(supabase_url, supabase_key)

def get_document_embedding(doc_id):
    """Get a document's embedding by ID."""
    print(f"Fetching document with ID: {doc_id}")
    
    response = supabase.from_("documents").select("*").eq("doc_id", doc_id).execute()
    
    if not response.data:
        print(f"No document found with ID: {doc_id}")
        return None
        
    doc = response.data[0]
    print(f"Found document: {doc['doc_id']} - {doc['content'][:50]}...")
    
    # Return the embedding
    return doc['embedding']

def search_with_embedding(embedding, top_k=5, threshold=0.0):
    """Search for documents using an embedding."""
    print(f"Searching with embedding (threshold={threshold})...")
    
    # Use the search_vectors function with the embedding
    sql = "SELECT * FROM search_vectors($1, $2, $3)"
    params = [embedding, top_k, threshold]
    
    response = supabase.rpc("exec_sql", {"sql": sql, "params": params}).execute()
    
    # Process results
    if hasattr(response, 'data') and response.data:
        rows = response.data
        print(f"\nFound {len(rows)} results with vector search")
        
        # Print the scores to see the similarity values
        for i, row in enumerate(rows):
            print(f"Result {i+1}: doc_id={row.get('doc_id')}, score={row.get('score')}")
    else:
        print("No results found with vector search")
        
    return response.data if hasattr(response, 'data') else []

def test_direct_vector_search():
    """Test vector search using a document's own embedding."""
    # Get the test document's embedding
    doc_id = "test_vector_doc"
    embedding = get_document_embedding(doc_id)
    
    if not embedding:
        print("Could not get embedding. Exiting.")
        return
        
    # Search using the document's own embedding
    print("\nSearching with document's own embedding:")
    results = search_with_embedding(embedding, top_k=5, threshold=0.0)
    
    # If still no results, try a raw SQL approach
    if not results:
        print("\nTrying direct SQL approach...")
        sql = """
        SELECT 
            doc_id, 
            content, 
            1 - (embedding <=> $1::vector) as score
        FROM 
            documents
        ORDER BY 
            embedding <=> $1::vector
        LIMIT 5
        """
        
        response = supabase.rpc("exec_sql", {"sql": sql, "params": [embedding]}).execute()
        
        if hasattr(response, 'data') and response.data:
            print(f"Direct SQL found {len(response.data)} results")
            for i, row in enumerate(response.data):
                print(f"Result {i+1}: doc_id={row.get('doc_id')}, score={row.get('score')}")
        else:
            print("Direct SQL found no results")

if __name__ == "__main__":
    test_direct_vector_search()
