"""
Personal_assistant_V1\personal_assistant\main.py
───────
Launches the personal_assistant defined in agent.py:

- sets up in‑memory session & memory services
- creates a Runner
- sends one demo query to illustrate the full callback chain
"""

import asyncio
import os

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agent import root_agent  # ← import the agent constructed in agent.py
from memory_service import MEMORY  # ← use Vertex RAG memory service

# ─────────────────────────────────────────────────────────
# 1. Backend services
# ─────────────────────────────────────────────────────────
SESSION = InMemorySessionService()  # still fine for short-term state

# Check if required env variables are set
required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"WARNING: Missing environment variables: {', '.join(missing_vars)}")
    print("Please set these variables in your .env file")

runner = Runner(
    agent=root_agent,
    app_name="personal_app",
    session_service=SESSION,
    memory_service=MEMORY,  # Now uses our memory service
)


# ─────────────────────────────────────────────────────────
# 2. Convenience async helper
# ─────────────────────────────────────────────────────────
async def chat_once(message: str) -> None:
    """Starts/continues a session, sends one message, prints final reply."""
    user_id, session_id = "u1", "s1"

    # Create the session record on the first turn
    if not SESSION.get_session("personal_app", user_id, session_id):
        SESSION.create_session("personal_app", user_id, session_id)

    print(f"\nUSER: {message}")

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        user_message=message,
    ):
        if event.is_turn_complete and event.text:
            print("ASSISTANT:", event.text)

# Example usage
if __name__ == "__main__":
    asyncio.run(chat_once("Save this document: 'AI is advancing rapidly in 2025.'"))
    # Test retrieval after saving
    asyncio.run(chat_once("What do you know about AI in 2025?"))