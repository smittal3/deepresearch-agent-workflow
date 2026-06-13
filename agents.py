"""The four agents. Each is a LangGraph node: it receives the shared
`ResearchState` and returns a partial dict updating ONLY its own slot.

Every agent is defensively wrapped so a single failure (e.g. a rate-limited
web search) degrades gracefully into a note instead of crashing the whole run.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_embed_model, get_llm
from state import ResearchReport, ResearchState


# --------------------------------------------------------------------------- #
# Agent 1 — RAG: the internal document reader (LlamaIndex)
# --------------------------------------------------------------------------- #
RAG_SYSTEM = (
    "You are a meticulous Document Analyst. Read the provided excerpts from the user's "
    "internal document(s) and extract what matters for their research query.\n\n"
    "Rules:\n"
    "1. Extract concrete facts, metrics, financials, dates, and core claims relevant to "
    "the query. Quote exact numbers and the exact wording of important claims.\n"
    "2. Label the epistemic status of each item so a downstream analyst can weigh it:\n"
    "   - [supported] backed by data, a figure, or a citation inside the document;\n"
    "   - [asserted] stated as fact but not substantiated within the document (self-reported);\n"
    "   - [projection] forward-looking, targeted, or hypothetical;\n"
    "   - [framing] opinion, positioning, or narrative spin.\n"
    "3. If MULTIPLE documents are provided, actively cross-check them against one another: "
    "flag agreements that reinforce a point AND any contradictions or inconsistencies between them.\n"
    "4. Note conspicuous ABSENCES: information a careful reader would expect for this query "
    "but that the document does not provide.\n"
    "5. Preserve the date or timeframe attached to any claim.\n"
    "6. Report ONLY what is in the text — no outside knowledge, no inference beyond what is "
    "written. If the document does not address the query, say so plainly."
)


def rag_agent(state: ResearchState) -> dict:
    if not state.get("has_document"):
        return {"rag_notes": "No internal document was provided by the user."}

    api_key = state["api_key"]
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.llms import MockLLM

        # We only use LlamaIndex for parse + embed + retrieve — never its LLM.
        # MockLLM prevents it from trying to resolve a default (OpenAI) model.
        Settings.llm = MockLLM()
        Settings.embed_model = get_embed_model(api_key)

        index = VectorStoreIndex.from_documents([Document(text=state["document_text"])])
        nodes = index.as_retriever(similarity_top_k=10).retrieve(state["user_query"])
        context = "\n\n---\n\n".join(n.get_content() for n in nodes)
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        return {"rag_notes": f"[RAG error] Could not index/retrieve from the document: {e}"}

    llm = get_llm(state["model"], api_key, temperature=0.1)
    human = (
        f"User's research query:\n{state['user_query']}\n\n"
        f"Relevant excerpts from the internal document:\n{context}"
    )
    resp = llm.invoke([SystemMessage(content=RAG_SYSTEM), HumanMessage(content=human)])
    return {"rag_notes": resp.text}


# --------------------------------------------------------------------------- #
# Agent 2 — Web search: the market pulse (DuckDuckGo)
# --------------------------------------------------------------------------- #
WEB_SYSTEM = (
    "You are a Web Researcher. Using the supplied search results, synthesize what "
    "credible, current sources say about the user's topic.\n\n"
    "Rules:\n"
    "1. Weigh sources by credibility: primary/official sources and reputable institutions "
    "> established press > blogs, forums, and marketing copy. Say which source supports each point.\n"
    "2. Weigh by recency: note dates where available and flag information that may be "
    "outdated for a fast-moving topic.\n"
    "3. Surface both points of CONSENSUS and points of DISAGREEMENT or controversy — do not "
    "flatten them into a single sentiment.\n"
    "4. Treat a conspicuous ABSENCE of expected coverage as a finding in itself (e.g. an "
    "entity that claims significant activity but has no verifiable public footprint).\n"
    "5. Separate what the sources actually say from your own inference, and label inference as such.\n"
    "6. Be concrete and grounded in the results. If results are thin or low-quality, say so "
    "rather than speculating."
)


def web_agent(state: ResearchState) -> dict:
    api_key = state["api_key"]
    llm = get_llm(state["model"], api_key, temperature=0.3)

    # Let the LLM expand the query into a few focused searches (more agentic + robust).
    try:
        q_resp = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Break the user's research goal into 3 focused web-search queries "
                        "that together cover different FACETS of the question — e.g. "
                        "background/definitions, supporting evidence, criticism or "
                        "counter-evidence, and recent developments. Make them genuinely "
                        "varied in angle and wording, not paraphrases of each other. "
                        "Output one query per line, no numbering or quotes."
                    )
                ),
                HumanMessage(content=state["user_query"]),
            ]
        )
        queries = [ln.strip("-• ").strip() for ln in q_resp.text.splitlines() if ln.strip()][:3]
    except Exception:  # noqa: BLE001
        queries = []
    if not queries:
        queries = [state["user_query"]]

    snippets: list[str] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            for q in queries:
                try:
                    for r in ddgs.text(q, max_results=4):
                        url = r.get("href", "")
                        snippets.append(
                            f"- {r.get('title', '')}: {r.get('body', '')} ({url})"
                        )
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            sources.append({"title": r.get("title", "") or url, "url": url})
                except Exception as e:  # noqa: BLE001 - one bad query shouldn't kill the rest
                    snippets.append(f"[search error for '{q}': {e}]")
    except Exception as e:  # noqa: BLE001
        return {
            "web_notes": f"[Web search unavailable: {e}] Proceeding without live market data.",
            "web_sources": [],
        }

    if not snippets:
        return {"web_notes": "No web results were found for this query.", "web_sources": []}

    blob = "\n".join(snippets[:20])
    human = f"Research goal:\n{state['user_query']}\n\nSearch results:\n{blob}"
    resp = llm.invoke([SystemMessage(content=WEB_SYSTEM), HumanMessage(content=human)])
    return {"web_notes": resp.text, "web_sources": sources[:10]}


# --------------------------------------------------------------------------- #
# Agent 3 — Academic: the technical validator (arXiv)
# --------------------------------------------------------------------------- #
ACADEMIC_SYSTEM = (
    "You are a Technical & Scientific Validator. Your ONLY material is the academic papers "
    "supplied below — do not analyze the user's internal document and do not speculate "
    "beyond what the papers establish.\n\n"
    "Rules:\n"
    "1. First judge relevance. If the papers do not genuinely bear on the user's topic, say "
    "so plainly and stop — do NOT stretch tangential papers into false validation.\n"
    "2. For relevant papers, extract baseline metrics, established results, known "
    "limitations, and the direction of scientific consensus, and state clearly whether it "
    "SUPPORTS or REFUTES the core idea behind the query.\n"
    "3. Distinguish the maturity of the evidence: peer-reviewed / replicated results vs. "
    "preprints / single studies / early claims. Note publication dates and whether the "
    "findings are still current.\n"
    "4. Be precise about what the literature does and does not establish."
)


def academic_agent(state: ResearchState) -> dict:
    api_key = state["api_key"]
    llm = get_llm(state["model"], api_key, temperature=0.2)

    try:
        keywords = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract a concise academic search phrase (3-6 keywords) capturing "
                        "the technical/scientific core of the user's query. Output only the phrase."
                    )
                ),
                HumanMessage(content=state["user_query"]),
            ]
        ).text.strip()
    except Exception:  # noqa: BLE001
        keywords = state["user_query"]

    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(
            query=keywords, max_results=3, sort_by=arxiv.SortCriterion.Relevance
        )
        entries = []
        sources: list[dict] = []
        for r in client.results(search):
            summary = (r.summary or "").strip().replace("\n", " ")[:1200]
            entries.append(
                f"Title: {r.title}\nPublished: {r.published.date()}\nAbstract: {summary}"
            )
            sources.append({"title": r.title, "url": r.entry_id})
        papers = "\n\n".join(entries)
    except Exception as e:  # noqa: BLE001
        return {
            "academic_notes": f"[Academic search unavailable: {e}] No academic validation performed.",
            "academic_sources": [],
        }

    if not papers.strip():
        return {
            "academic_notes": f"No directly relevant academic papers found for: {keywords}",
            "academic_sources": [],
        }

    human = (
        f"Topic / claims to validate:\n{state['user_query']}\n\n"
        f"Academic papers found (search: '{keywords}'):\n{papers}"
    )
    resp = llm.invoke(
        [SystemMessage(content=ACADEMIC_SYSTEM), HumanMessage(content=human)]
    )
    return {"academic_notes": resp.text, "academic_sources": sources}


# --------------------------------------------------------------------------- #
# Agent 4 — Synthesis: the orchestrating senior analyst (structured JSON out)
# --------------------------------------------------------------------------- #
SYNTHESIS_SYSTEM = (
    "You are a Senior Analyst writing a decision-ready brief. You receive up to three "
    "independent research streams: (1) notes from the user's internal document(s), "
    "(2) what credible web sources say, and (3) technical/academic evidence. One or more "
    "streams may be empty or unavailable — adapt, and lower your confidence accordingly.\n\n"
    "Method:\n"
    "1. TRIANGULATE. A claim is strong only when independent streams agree. Treat any "
    "pivotal claim that rests on a single source — especially a self-interested one — as "
    "unverified, and say so.\n"
    "2. SEPARATE epistemic levels: established fact vs. self-reported claim vs. projection "
    "vs. opinion. Do not let a confident claim inherit the authority of a verified fact.\n"
    "3. HUNT CONTRADICTIONS, adapted to what you actually have:\n"
    "   - if there is an internal document, test its claims against external evidence;\n"
    "   - if there are multiple documents, check them against each other;\n"
    "   - with or without a document, surface conflicts between sources and "
    "consensus-vs-minority disagreements.\n"
    "   A conspicuous absence of expected evidence counts as a red flag.\n"
    "4. CALIBRATE confidence on the quality, agreement, and coverage of the evidence — not "
    "on tone. If a stream was unavailable or sources were thin, confidence cannot be High.\n"
    "5. Make the UNKNOWNS explicit: state what is missing, what could not be verified, and "
    "what new information would most change the conclusion.\n\n"
    "Output rules:\n"
    "- Fill EVERY field of the required schema. Be specific and quote concrete figures or "
    "wording; avoid vague hedging.\n"
    "- Never fabricate numbers or sources to fill a field; if something is unsupported, say so.\n"
    "- For the scorecard, choose 3-5 evaluation dimensions that genuinely fit THIS query "
    "(a startup, a scientific claim, a policy question, and a product each warrant different "
    "rubrics) and score each 1-5 with a one-line, evidence-grounded rationale.\n"
    "- For each data point, set source_ids to the IDs from the provided SOURCES list that "
    "back it (only IDs that appear there); leave it empty if the fact comes from the "
    "internal document or general reasoning. Never invent source IDs."
)


def _build_sources(state: ResearchState) -> list[dict]:
    """Deterministically assemble the numbered source list from what the agents
    actually retrieved. URLs are never produced by the LLM, so they can't be faked."""
    sources: list[dict] = []
    if state.get("has_document"):
        doc_names = state.get("doc_names") or [state.get("doc_name") or "Internal document"]
        for name in doc_names:
            sources.append({"title": name, "url": "", "type": "document"})
    for s in state.get("web_sources", []) or []:
        sources.append({"title": s.get("title", ""), "url": s.get("url", ""), "type": "web"})
    for s in state.get("academic_sources", []) or []:
        sources.append({"title": s.get("title", ""), "url": s.get("url", ""), "type": "academic"})
    # Assign stable 1-based ids used both in the prompt and the rendered report.
    for i, s in enumerate(sources, start=1):
        s["id"] = i
    return sources


def synthesis_agent(state: ResearchState) -> dict:
    llm = get_llm(state["model"], state["api_key"], temperature=0.2)
    structured = llm.with_structured_output(ResearchReport)

    sources = _build_sources(state)
    if sources:
        source_block = "\n".join(
            f"[{s['id']}] ({s['type']}) {s['title']} {s['url']}".rstrip() for s in sources
        )
    else:
        source_block = "(no external sources were retrieved)"

    human = (
        f"User's research query: {state['user_query']}\n\n"
        f"=== INTERNAL DOCUMENT NOTES (RAG agent) ===\n{state.get('rag_notes', '(none)')}\n\n"
        f"=== MARKET / WEB SENTIMENT (Web agent) ===\n{state.get('web_notes', '(none)')}\n\n"
        f"=== ACADEMIC / TECHNICAL REALITY (Academic agent) ===\n"
        f"{state.get('academic_notes', '(none)')}\n\n"
        f"=== SOURCES (cite these IDs in each data point's source_ids) ===\n{source_block}\n"
    )
    report: ResearchReport = structured.invoke(
        [SystemMessage(content=SYNTHESIS_SYSTEM), HumanMessage(content=human)]
    )

    final = report.model_dump()
    # Attach the deterministic source list (not LLM-generated) and keep only valid ids.
    final["sources"] = sources
    valid_ids = {s["id"] for s in sources}
    for dp in final.get("key_data_points", []):
        dp["source_ids"] = [i for i in dp.get("source_ids", []) if i in valid_ids]
    return {"final_report": final}
