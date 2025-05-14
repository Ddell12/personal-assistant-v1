"""
Personal_assistant_V1\personal_assistant\agent.py
"""

from datetime import datetime
import json  # Add this import
from typing import Any, Dict, Optional

# ADK core classes
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.genai import types

# import the AgentTool that wraps the dedicated RAG agent
from .sub_agents.rag_agent import rag_agent_tool


# ─────────────────────────────────────────────────────────
# 1. AGENT‑LEVEL CALLBACKS
# ─────────────────────────────────────────────────────────
def before_agent_cb(*, callback_context: CallbackContext) -> Optional[types.Content]:
    """Executed right before the root agent starts processing the turn."""
    state = callback_context.state
    state["start_time"] = datetime.now()
    state["turn"] = state.get("turn", 0) + 1
    print(f"[before_agent] turn {state['turn']}")
    return None


def after_agent_cb(*, callback_context: CallbackContext) -> Optional[types.Content]:
    """Executed after the agent completes; logs total turn latency."""
    dt = (datetime.now() - callback_context.state["start_time"]).total_seconds()
    print(f"[after_agent] finished in {dt:.2f}s")
    return None


# ─────────────────────────────────────────────────────────
# 2. MODEL‑LEVEL CALLBACKS
# ─────────────────────────────────────────────────────────
def before_model_cb(*, callback_context: CallbackContext,
                    llm_request: LlmRequest) -> Optional[LlmResponse]:
    """
    Count prompt tokens safely (skip parts with no text).
    """
    tokens = sum(
        len(p.text.split())
        for p in llm_request.contents[-1].parts
        if getattr(p, "text", None)  # <- guard against None
    )
    print("[before_model] prompt tokens:", tokens)
    return None


def after_model_cb(
    *, callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Logs number of text tokens in the raw model reply (0 if only function_call)."""
    words = (
        sum(
            len(p.text.split())
            for p in llm_response.content.parts
            if getattr(p, "text", None)
        )
        if llm_response.content and llm_response.content.parts
        else 0
    )
    print("[after_model] raw model tokens:", words)
    return None


# ─────────────────────────────────────────────────────────
# 3. TOOL‑LEVEL CALLBACKS
# ─────────────────────────────────────────────────────────
def before_tool_cb(
    *, tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """Runs just before any tool executes; logs and stores last_tool."""
    print(f"[before_tool] {tool.name} args={json.dumps(args, indent=2)}")
    tool_context.state["last_tool"] = tool.name
    return None


def after_tool_cb(*, tool: BaseTool, args: Dict[str, Any],
                  tool_context: ToolContext, tool_response: Any) -> Optional[Any]:
    """Runs after the tool finishes; logs a brief summary regardless of type."""
    try:
        # Try to get a more detailed summary of the response
        if isinstance(tool_response, dict):
            summary = f"dict keys={list(tool_response.keys())[:5]}"
            print(f"[after_tool] {tool.name} response={json.dumps(tool_response, indent=2)[:500]}")
        elif isinstance(tool_response, str):
            summary = f"str len={len(tool_response)}"
            print(f"[after_tool] {tool.name} -> {summary}")
            print(f"Content preview: {tool_response[:200]}...")
        else:
            summary = f"{type(tool_response).__name__}"
            print(f"[after_tool] {tool.name} -> {summary}")
    except Exception as e:
        # If anything goes wrong with detailed logging, fall back to simple logging
        summary = (
            f"dict keys={list(tool_response)[:5]}" if isinstance(tool_response, dict)
            else f"str len={len(tool_response)}"   if isinstance(tool_response, str)
            else f"{type(tool_response).__name__}"
        )
        print(f"[after_tool] {tool.name} -> {summary} (Exception during logging: {e})")
    
    return None          # never mutate the real response



# ─────────────────────────────────────────────────────────
# 4. AGENT DEFINITION
# ─────────────────────────────────────────────────────────
root_agent = LlmAgent(
    name="personal_assistant",
    model="gemini-2.5-flash-preview-04-17",
    instruction="""
You are Personal Assistant.

If the user asks to **store** or **retrieve** personal knowledge, delegate to
supabase_agent_tool exactly once, passing a literal request string such as:
- "upsert doc id=… <text> { … }"
- "search <question>"
For all other queries answer directly.
""",
    tools=[rag_agent_tool],  # only one tool keeps AFC happy
    before_agent_callback=before_agent_cb,
    after_agent_callback=after_agent_cb,
    before_model_callback=before_model_cb,
    after_model_callback=after_model_cb,
    before_tool_callback=before_tool_cb,
    after_tool_callback=after_tool_cb,
)