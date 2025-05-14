"""
Personal_assistant_V1\personal_assistant\memory_service.py
───────
Provides a memory service implementation that integrates with Supabase
for Retrieval Augmented Generation (RAG) capabilities.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from google.adk.memory import MemoryService, MemoryStoreQuery, MemoryRecord

# Import the Supabase vector store for integration
from personal_assistant.sub_agents.rag_agent.vector_stores.supabase_store import SupabaseVectorStore

class SupabaseMemoryService(MemoryService):
    """Memory service implementation using Supabase for RAG capabilities."""

    def __init__(self):
        """Initialize the memory service with Supabase vector store."""
        self._store = SupabaseVectorStore()
        # In-memory storage for recent conversations and sessions
        self._session_records: Dict[str, List[MemoryRecord]] = {}
        print("[MemoryService] Initialized with Supabase vector store")

    def create(self, record: MemoryRecord) -> MemoryRecord:
        """Create a new memory record."""
        # Generate a unique ID if one isn't provided
        if not record.id:
            record.id = str(uuid.uuid4())
        
        # Add timestamp if not present
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat()
        
        # Store in session memory
        session_key = f"{record.app_id}:{record.user_id}:{record.session_id}"
        if session_key not in self._session_records:
            self._session_records[session_key] = []
        
        self._session_records[session_key].append(record)
        
        # For "important" memories that should be retrievable long-term,
        # store them in the vector database
        if record.metadata and record.metadata.get("store_in_vector_db", False):
            try:
                self._store.upsert(
                    doc_id=record.id,
                    content=record.text,
                    metadata={
                        "app_id": record.app_id,
                        "user_id": record.user_id,
                        "session_id": record.session_id,
                        "timestamp": record.timestamp,
                        "type": record.metadata.get("type", "memory"),
                        **record.metadata  # Include other metadata
                    }
                )
                print(f"[MemoryService] Stored record {record.id} in vector DB")
            except Exception as e:
                print(f"[MemoryService] Error storing in vector DB: {e}")
        
        return record

    def get(self, app_id: str, user_id: str, session_id: str, 
            record_id: str) -> Optional[MemoryRecord]:
        """Retrieve a specific memory record."""
        session_key = f"{app_id}:{user_id}:{session_id}"
        
        # Check in-memory records first
        if session_key in self._session_records:
            for record in self._session_records[session_key]:
                if record.id == record_id:
                    return record
        
        return None  # Not found

    def list(self, app_id: str, user_id: str, session_id: str) -> List[MemoryRecord]:
        """List all memory records for a session."""
        session_key = f"{app_id}:{user_id}:{session_id}"
        return self._session_records.get(session_key, [])

    def search(self, query: MemoryStoreQuery) -> List[MemoryRecord]:
        """Search for memory records based on semantic similarity."""
        if not query.query_text:
            return []
        
        try:
            # Use the vector store to perform semantic search
            results = self._store.search(query.query_text, top_k=query.limit or 5)
            
            # Convert results back to MemoryRecords
            records = []
            for hit in results:
                metadata = {}
                if hit.get("metadata"):
                    try:
                        if isinstance(hit["metadata"], str):
                            metadata = json.loads(hit["metadata"])
                        else:
                            metadata = hit["metadata"]
                    except:
                        metadata = {"raw_metadata": str(hit["metadata"])}
                
                # Extract app_id, user_id, session_id from metadata if available
                app_id = metadata.get("app_id", query.app_id)
                user_id = metadata.get("user_id", query.user_id)
                session_id = metadata.get("session_id", query.session_id)
                
                record = MemoryRecord(
                    id=hit.get("doc_id", str(uuid.uuid4())),
                    app_id=app_id,
                    user_id=user_id,
                    session_id=session_id,
                    text=hit.get("content", ""),
                    timestamp=metadata.get("timestamp", datetime.now().isoformat()),
                    metadata=metadata
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"[MemoryService] Error during vector search: {e}")
            return []

    def delete(self, app_id: str, user_id: str, session_id: str,
              record_id: str) -> bool:
        """Delete a specific memory record."""
        session_key = f"{app_id}:{user_id}:{session_id}"
        
        # Delete from in-memory storage
        if session_key in self._session_records:
            self._session_records[session_key] = [
                record for record in self._session_records[session_key]
                if record.id != record_id
            ]
        
        return True

    def clear(self, app_id: str, user_id: str, session_id: str) -> bool:
        """Clear all memory records for a session."""
        session_key = f"{app_id}:{user_id}:{session_id}"
        
        # Clear from in-memory storage
        if session_key in self._session_records:
            self._session_records[session_key] = []
        
        return True


# Create an instance of the memory service for use in the application
MEMORY = SupabaseMemoryService()