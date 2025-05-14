# Personal_assistant_V1\personal_assistant\tools\supabase_tools.py
import json
from typing import Dict, Any, List

from personal_assistant.vector_stores.supabase_store import SupabaseVectorStore

_store = SupabaseVectorStore(auto_setup=True)  # Auto-setup option to check database

def supabase_upsert(doc_id: str, content: str,
                    metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Insert or update a document in Supabase pgvector.
    """
    try:
        print(f"[supabase_upsert] Storing document: id={doc_id}, content={content[:50]}...")
        row = _store.upsert(doc_id, content, metadata)
        return {"status": "success", "row": row}
    except Exception as e:
        print(f"[supabase_upsert] Error: {str(e)}")
        return {"status": "error", "message": str(e)}

def supabase_upsert_batch(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Insert or update multiple documents in a single operation.
    
    Each document should have: doc_id, content, and metadata.
    """
    try:
        print(f"[supabase_upsert_batch] Storing {len(documents)} documents...")
        results = _store.upsert_batch(documents)
        return {
            "status": "success", 
            "message": f"Successfully processed {len(results)} documents",
            "count": len(results)
        }
    except Exception as e:
        print(f"[supabase_upsert_batch] Error: {str(e)}")
        return {"status": "error", "message": str(e)}

def supabase_search(query: str, top_k: int = 8) -> Dict[str, Any]:
    """Semantic search against Supabase pgvector index."""
    try:
        print(f"[supabase_search] Searching for: {query}")
        hits = _store.search(query, top_k)
        print(f"[supabase_search] Found {len(hits)} results")
        return {"status": "success", "hits": hits}
    except Exception as e:
        print(f"[supabase_search] Error: {str(e)}")
        return {"status": "error", "message": str(e)}

def supabase_delete(doc_id: str) -> Dict[str, Any]:
    """Delete a document from Supabase by ID."""
    try:
        print(f"[supabase_delete] Deleting document: id={doc_id}")
        success = _store.delete(doc_id)
        if success:
            return {"status": "success", "message": f"Document {doc_id} deleted"}
        else:
            return {"status": "warning", "message": f"Document {doc_id} not found"}
    except Exception as e:
        print(f"[supabase_delete] Error: {str(e)}")
        return {"status": "error", "message": str(e)}

def supabase_get_document(doc_id: str) -> Dict[str, Any]:
    """Retrieve a specific document by ID."""
    try:
        print(f"[supabase_get_document] Retrieving document: id={doc_id}")
        doc = _store.get_document(doc_id)
        if doc:
            return {"status": "success", "document": doc}
        else:
            return {"status": "not_found", "message": f"Document {doc_id} not found"}
    except Exception as e:
        print(f"[supabase_get_document] Error: {str(e)}")
        return {"status": "error", "message": str(e)}

def supabase_count_documents() -> Dict[str, Any]:
    """Count total documents in the store."""
    try:
        count = _store.count_documents()
        return {"status": "success", "count": count}
    except Exception as e:
        print(f"[supabase_count_documents] Error: {str(e)}")
        return {"status": "error", "message": str(e)}