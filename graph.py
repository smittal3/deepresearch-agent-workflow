"""LangGraph wiring.

    START ─┬─▶ rag_agent ──────┐
           ├─▶ web_agent ──────┼─▶ synthesis_agent ─▶ END
           └─▶ academic_agent ─┘

The three research agents fan out from START and run in parallel. Each writes a
different state slot, so there are no write conflicts. `synthesis_agent` is a
fan-in: LangGraph waits for all three to finish before running it.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from agents import academic_agent, rag_agent, synthesis_agent, web_agent
from state import ResearchState

# Friendly metadata for the UI status display, in the order shown.
AGENT_STEPS = [
    ("rag_agent", "📄 Document Analyst", "Extracting claims & facts from your file"),
    ("web_agent", "🌐 Web Researcher", "Searching the web for current information & sentiment"),
    ("academic_agent", "🔬 Academic Validator", "Checking academic literature on arXiv"),
    ("synthesis_agent", "🧠 Lead Analyst", "Cross-examining sources & writing the report"),
]
PARALLEL_AGENTS = ["rag_agent", "web_agent", "academic_agent"]


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("rag_agent", rag_agent)
    g.add_node("web_agent", web_agent)
    g.add_node("academic_agent", academic_agent)
    g.add_node("synthesis_agent", synthesis_agent)

    for name in PARALLEL_AGENTS:
        g.add_edge(START, name)
        g.add_edge(name, "synthesis_agent")
    g.add_edge("synthesis_agent", END)

    return g.compile()
