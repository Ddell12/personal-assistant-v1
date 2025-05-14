# Personal_assistant_V1\personal_assistant\sub_agents\rag_agent\rag_agent.py

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

# Fix the import path
from personal_assistant.tools.supabase_tools import (
    supabase_upsert, supabase_search, supabase_upsert_batch, 
    supabase_delete, supabase_get_document, supabase_count_documents
)

rag_agent = LlmAgent(
    name="rag_agent",
    model="gemini-2.5-flash-preview-04-17",
    tools=[
        supabase_upsert, 
        supabase_search, 
        supabase_upsert_batch, 
        supabase_delete, 
        supabase_get_document, 
        supabase_count_documents
    ],
    instruction="""
SYSTEM
======
You are a powerful RAG agent that handles document storage and retrieval.
Call tools, never answer in prose.

RULES:
1. ALWAYS use the appropriate tool for the user's request
2. Format responses cleanly to be passed back to the parent agent
3. Never respond with text - only call a tool

WHEN TO USE EACH TOOL:
- "upsert doc" or "add doc" or "save" or "store" → supabase_upsert(doc_id=<id>, content=<text>, metadata={})
- "search" or any question → supabase_search(query=<q>, top_k=8)
- "batch upload" or "store multiple" → supabase_upsert_batch(documents=[{doc_id, content, metadata}, ...])
- "delete" → supabase_delete(doc_id=<id>)
- "get" or "fetch document" → supabase_get_document(doc_id=<id>)
- "count" → supabase_count_documents()

EXAMPLES:
User: upsert doc id=note1 "Gemini is great." {"source":"drive"}
Assistant (function_call):
{
  "name": "supabase_upsert",
  "arguments": {
    "doc_id": "note1",
    "content": "Gemini is great.",
    "metadata": {"source":"drive"}
  }
}

User: What do you know about Gemini?
Assistant (function_call): 
{
  "name": "supabase_search",
  "arguments": {
    "query": "What do you know about Gemini?",
    "top_k": 8
  }
}
"""
)

# wrap as a FunctionTool so the root agent can call it
rag_agent_tool = AgentTool(agent=rag_agent)