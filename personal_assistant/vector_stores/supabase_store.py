# ──────────────────────────────────────────────────────────────
# personal_assistant/sub_agents/rag_agent/vector_stores/supabase_store.py
# Supabase pgvector store that uses OpenAI's text‑embedding‑3‑small model
# ──────────────────────────────────────────────────────────────
import os
import json
import time
import typing as t
from functools import lru_cache
from supabase import create_client, Client
from openai import OpenAI  # Updated import

# ---------- Constants and Configuration ------------------------------------------
EMBED_MODEL = "text-embedding-3-small"  # 1536‑d 
_TOPK_DEFAULT = 8  # default search limit
_BATCH_SIZE = 20  # number of documents to embed in a single API call
_MAX_RETRIES = 3  # maximum number of retries for API calls
_RETRY_DELAY = 2  # seconds to wait between retries

# ---------- OpenAI client initialization ------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Set OPENAI_API_KEY in your env / .env file"
client = OpenAI(api_key=api_key)  # Initialize client

# ---------- OpenAI embedding helper ------------------------------------------
@lru_cache(maxsize=100)  # Cache recent embeddings to reduce API calls
def _embed_single(text: str) -> t.List[float]:
    """
    Get a single 1536‑dim embedding from OpenAI (floats list).
    Uses caching to avoid re-embedding the same text.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text")
        
    retries = 0
    while retries < _MAX_RETRIES:
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            retries += 1
            if retries >= _MAX_RETRIES:
                print(f"Failed to generate embedding after {_MAX_RETRIES} attempts. Error: {str(e)}")
                raise
            print(f"Error generating embedding (attempt {retries}/{_MAX_RETRIES}): {str(e)}")
            time.sleep(_RETRY_DELAY)

def _embed_batch(texts: t.List[str]) -> t.List[t.List[float]]:
    """
    Get embeddings for multiple texts in a single API call.
    Returns a list of embedding vectors.
    """
    if not texts:
        return []
        
    # Filter out empty texts
    valid_texts = [text for text in texts if text.strip()]
    if not valid_texts:
        return []
        
    retries = 0
    while retries < _MAX_RETRIES:
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=valid_texts,
                encoding_format="float"
            )
            # Return embeddings in the same order as input texts
            return [item.embedding for item in response.data]
        except Exception as e:
            retries += 1
            if retries >= _MAX_RETRIES:
                print(f"Failed to generate batch embeddings after {_MAX_RETRIES} attempts. Error: {str(e)}")
                raise
            print(f"Error generating batch embeddings (attempt {retries}/{_MAX_RETRIES}): {str(e)}")
            time.sleep(_RETRY_DELAY * retries)  # Increasing backoff

# ---------- Vector store class ----------------------------------------------
class SupabaseVectorStore:
    def __init__(self, auto_setup: bool = False) -> None:
        """
        Initialize the Supabase Vector Store.
        
        Args:
            auto_setup: If True, will attempt to set up the database schema if it doesn't exist
        """
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        assert url and key, "Set SUPABASE_URL and SUPABASE_SERVICE_KEY"
        self.cli: Client = create_client(url, key)
        
        # Check if database is set up properly
        if auto_setup:
            is_setup = self.check_database_setup()
            if not is_setup:
                print("Database not set up properly. Attempting to create schema...")
                self.setup_database()

    def check_database_setup(self) -> bool:
        """
        Verify that the database has the necessary tables and extensions.
        Returns True if the database is properly set up, False otherwise.
        """
        try:
            # Check if the documents table exists
            response = self.cli.table("documents").select("count(*)", count="exact").execute()
            if hasattr(response, 'error') and response.error:
                print(f"Error checking documents table: {response.error}")
                return False
                
            # Check if pgvector extension is enabled (indirectly)
            # We try a simple vector similarity query to see if it works
            try:
                dummy_vector = [0.0] * 1536  # Create a dummy vector of the right size
                sql = """
                SELECT 1 
                FROM documents 
                WHERE embedding <=> %(vec)s::vector 
                LIMIT 1;
                """
                self.cli.postgrest.rpc("exec_sql", {"sql": sql, "params": {"vec": dummy_vector}}).execute()
                # If we get here without exception, the vector extension is working
                return True
            except Exception as e:
                print(f"PGVector extension might not be enabled: {str(e)}")
                return False
                
        except Exception as e:
            print(f"Error checking database setup: {str(e)}")
            return False

    def setup_database(self) -> bool:
        """
        Set up the database schema with proper tables and extensions.
        Returns True if successful, False otherwise.
        """
        try:
            # Note: In production, this would need administrative privileges
            # and might need to be run through a migration script
            
            # For now, just print instructions
            print("""
            To set up your Supabase database properly, run these SQL commands in the SQL editor of your Supabase dashboard:
            
            -- Enable the pgvector extension
            CREATE EXTENSION IF NOT EXISTS vector;
            
            -- Create the documents table
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                doc_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(1536) NOT NULL, 
                metadata JSONB DEFAULT '{}'::JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- Create an index for vector similarity search
            CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
            return False  # Return False since we didn't actually set it up
        except Exception as e:
            print(f"Error setting up database: {str(e)}")
            return False

    # -------- public API used by agent tools --------
    def upsert(self, doc_id: str, content: str, metadata: dict) -> dict:
        """
        Insert or update a document in the vector store.
        
        Args:
            doc_id: Unique identifier for the document
            content: The text content to store and embed
            metadata: Additional data about the document
            
        Returns:
            The created or updated document record
        """
        try:
            if not content.strip():
                raise ValueError("Document content cannot be empty")
                
            if not doc_id:
                raise ValueError("Document ID is required")
                
            # Generate embedding for the content
            embedding = _embed_single(content)
            
            # Prepare row data
            row = {
                "doc_id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": json.dumps(metadata) if isinstance(metadata, dict) else metadata,
            }
            
            # Insert or update the document
            response = self.cli.table("documents").upsert(row).execute()
            
            if hasattr(response, 'error') and response.error:
                raise RuntimeError(f"Upsert error: {response.error}")
                
            if not response.data:
                raise RuntimeError("Upsert returned no data")
                
            return response.data[0]
            
        except Exception as e:
            print(f"Error in upsert operation: {str(e)}")
            raise

    def upsert_batch(self, documents: t.List[t.Dict[str, t.Any]]) -> t.List[dict]:
        """
        Insert or update multiple documents in a single transaction.
        
        Args:
            documents: List of document dicts, each with 'doc_id', 'content', and 'metadata'
            
        Returns:
            List of created or updated document records
        """
        if not documents:
            return []
            
        try:
            # Group into batches to avoid API rate limits
            batches = [documents[i:i + _BATCH_SIZE] for i in range(0, len(documents), _BATCH_SIZE)]
            results = []
            
            for batch in batches:
                # Extract content for batch embedding
                contents = [doc["content"] for doc in batch if "content" in doc and doc["content"]]
                
                # Get embeddings for the batch
                if contents:
                    embeddings = _embed_batch(contents)
                    
                    # Map embeddings back to documents
                    rows = []
                    for i, doc in enumerate(batch):
                        if i < len(embeddings) and "content" in doc and doc["content"]:
                            rows.append({
                                "doc_id": doc["doc_id"],
                                "content": doc["content"],
                                "embedding": embeddings[i],
                                "metadata": json.dumps(doc.get("metadata", {})) if isinstance(doc.get("metadata", {}), dict) else doc.get("metadata", "{}"),
                            })
                    
                    # Insert or update in a single transaction
                    if rows:
                        response = self.cli.table("documents").upsert(rows).execute()
                        if hasattr(response, 'error') and response.error:
                            raise RuntimeError(f"Batch upsert error: {response.error}")
                        
                        results.extend(response.data)
            
            return results
            
        except Exception as e:
            print(f"Error in batch upsert operation: {str(e)}")
            raise

    def search(self, query: str, top_k: int = _TOPK_DEFAULT) -> t.List[dict]:
        """
        Search for documents similar to the query.
        """
        try:
            if not query.strip():
                raise ValueError("Search query cannot be empty")
                
            # Generate embedding for the query
            q_emb = _embed_single(query)
            
            # Perform vector similarity search with simpler SQL that's more likely to work
            sql = """
            SELECT  
                doc_id,
                content,
                metadata,
                1 - (embedding <=> $1::vector) as score
            FROM    
                documents
            ORDER BY 
                embedding <=> $1::vector
            LIMIT   
                $2
            """
            
            # This calls the function with proper parameters
            response = self.cli.rpc(
                "exec_sql", 
                {"sql": sql, "params": {"vec": q_emb, "k": top_k}}
            ).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"Vector search error: {response.error}")
                return self.search_fallback(query, top_k)
                
            # Process results
            rows = response.data or []
            
            # Parse metadata if it's a JSON string
            for row in rows:
                if isinstance(row.get("metadata"), str):
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        pass
            
            if not rows:
                # If no results from vector search, try the fallback
                return self.search_fallback(query, top_k)
                
            return rows
            
        except Exception as e:
            print(f"Error in search operation: {str(e)}")
            return self.search_fallback(query, top_k)

    def search_fallback(self, query: str, top_k: int = _TOPK_DEFAULT) -> t.List[dict]:
        """
        Improved fallback search when vector search fails:
        1. First tries to find the document by using word parts from the query
        2. If that fails, list all documents and search client-side
        """
        print(f"[supabase_search] Using fallback search for: {query}")
        
        try:
            # Try several search patterns
            query_words = query.lower().replace("?", "").replace(".", "").split()
            
            # Try various combinations of words to increase chances of a match
            for i in range(len(query_words)):
                if i > 3:  # Don't try too many combinations
                    break
                    
                # Try with increasingly fewer words
                search_term = "%" + "%".join(query_words[i:]) + "%"
                response = self.cli.table("documents").select("*").ilike("content", search_term).limit(top_k).execute()
                
                if response.data and len(response.data) > 0:
                    print(f"[supabase_search] Found {len(response.data)} results with fallback (term: {search_term})")
                    return response.data
            
            # Second fallback: try with project name directly if it might be in the query
            if "project" in query.lower():
                project_name = query.lower().split("project")[1].strip().split()[0]
                if project_name:
                    search_term = f"%project {project_name}%"
                    response = self.cli.table("documents").select("*").ilike("content", search_term).limit(top_k).execute()
                    if response.data and len(response.data) > 0:
                        return response.data
            
            # Last resort: just get all documents and filter client-side
            response = self.cli.table("documents").select("*").limit(100).execute()
            
            if response.data:
                # Simple client-side search
                results = []
                query_terms = set(query.lower().replace("?", "").replace(".", "").split())
                
                for doc in response.data:
                    content = doc.get("content", "").lower()
                    # Check if ANY of the query terms appear in the content
                    if any(term in content for term in query_terms):
                        results.append(doc)
                        if len(results) >= top_k:
                            break
                
                print(f"[supabase_search] Found {len(results)} results with client-side filtering")
                return results[:top_k]
                
            return []
                
        except Exception as e:
            print(f"Error in fallback search: {str(e)}")
            
            # Absolute last resort - just return any documents
            try:
                return self.cli.table("documents").select("*").limit(top_k).execute().data or []
            except:
                return []

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Unique identifier for the document to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            response = self.cli.table("documents").delete().eq("doc_id", doc_id).execute()
            
            if hasattr(response, 'error') and response.error:
                raise RuntimeError(f"Delete error: {response.error}")
                
            # Check if anything was deleted
            return len(response.data) > 0
            
        except Exception as e:
            print(f"Error in delete operation: {str(e)}")
            return False

    def delete_batch(self, doc_ids: t.List[str]) -> int:
        """
        Delete multiple documents in a single operation.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents successfully deleted
        """
        if not doc_ids:
            return 0
            
        try:
            # Format list for SQL IN clause
            doc_ids_str = ", ".join(f"'{doc_id}'" for doc_id in doc_ids)
            
            # Use SQL for bulk delete
            sql = f"""
            DELETE FROM documents 
            WHERE doc_id IN ({doc_ids_str})
            RETURNING doc_id;
            """
            
            response = self.cli.postgrest.rpc("exec_sql", {"sql": sql, "params": {}}).execute()
            
            if hasattr(response, 'error') and response.error:
                raise RuntimeError(f"Batch delete error: {response.error}")
                
            # Return count of deleted documents
            return len(response.data) if response.data else 0
            
        except Exception as e:
            print(f"Error in batch delete operation: {str(e)}")
            return 0

    def get_document(self, doc_id: str) -> t.Optional[dict]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Unique identifier for the document
            
        Returns:
            Document record or None if not found
        """
        try:
            response = self.cli.table("documents").select("*").eq("doc_id", doc_id).execute()
            
            if hasattr(response, 'error') and response.error:
                raise RuntimeError(f"Get document error: {response.error}")
                
            if not response.data:
                return None
                
            # Process the result
            document = response.data[0]
            
            # Parse metadata if it's a JSON string
            if isinstance(document.get("metadata"), str):
                try:
                    document["metadata"] = json.loads(document["metadata"])
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass
            
            return document
            
        except Exception as e:
            print(f"Error retrieving document: {str(e)}")
            return None

    def count_documents(self) -> int:
        """
        Count the total number of documents in the store.
        
        Returns:
            The document count
        """
        try:
            response = self.cli.table("documents").select("count(*)", count="exact").execute()
            
            if hasattr(response, 'error') and response.error:
                raise RuntimeError(f"Count error: {response.error}")
                
            return response.count if hasattr(response, 'count') else 0
            
        except Exception as e:
            print(f"Error counting documents: {str(e)}")
            return 0