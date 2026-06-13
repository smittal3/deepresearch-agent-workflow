# 🔭 Deep Research Agent

A multi-agent research assistant. Upload a document (e.g. a startup pitch deck),
ask a question, and **four specialist agents investigate in parallel** and return
a decision-ready dashboard — _"how competitive is this, and should I invest?"_

| Agent | Role | Tech |
|---|---|---|
| 📄 **Document Analyst** | Extracts claims & metrics from your file (PDF/DOCX/PPTX/TXT) | LlamaIndex RAG (in-memory vector index) |
| 🌐 **Market Researcher** | Web sentiment, competitors, traction | DuckDuckGo search |
| 🔬 **Technical Validator** | Validates the underlying science | arXiv |
| 🧠 **Senior Analyst** | Cross-examines all sources → JSON report | LangChain structured output |

Orchestrated with **LangGraph** (the three research agents fan out in parallel,
then the analyst synthesizes). UI is **Streamlit**.

**Highlights**
- 📂 Accepts **PDF, DOCX, PPTX, TXT/MD** (pitch decks are usually PPTX).
- 🔗 **Cited output** — every key data point links to the real source it came from
  (URLs are collected from what the agents actually fetched, never invented by the LLM).
- 🎯 **Adaptive scorecard** — the analyst picks 3-5 evaluation dimensions that fit
  *your* query (a startup vs. a scientific claim get different rubrics), each scored 1-5.

> **One key for everything.** All four agents *and* the RAG embeddings run on
> **Google Gemini**, so each teammate only needs a single free `GOOGLE_API_KEY`.

---

## Quick start

### 0. Get a free API key (30 seconds)
Go to **https://aistudio.google.com/app/apikey** → **Create API key** → copy it.

### 1. Setup

**macOS / Linux**
```bash
bash setup.sh
```

**Windows**
```bat
setup.bat
```

This creates a `.venv` and installs everything from `requirements.txt`.

> Requires **Python 3.10+** (3.11 recommended). On macOS, `python3.11` from
> [python.org](https://www.python.org/downloads/) or Homebrew works great.
> Python 3.9 is **not** supported (a dependency uses 3.10+ syntax).

### 2. Add your key (optional)
Either paste the key into the app's **sidebar** at runtime, or create a `.env`:
```bash
cp .env.example .env      # then edit .env and paste your key
```

### 3. Run

**macOS / Linux:** `bash run.sh`  **Windows:** `run.bat`

…or directly:
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501.

---

## Using it

1. Click **Try sample case study** (loads the bundled startup files + a VC prompt in
   one click), or **upload** your own PDF / DOCX / PPTX / TXT — *several at once is fine,
   and documents are optional.* Image-based PDFs (e.g. pitch decks exported as images)
   are transcribed automatically via Gemini's PDF vision — no OCR install needed.
2. **Type your query.** Click **✨ Load sample** for a ready-made startup
   due-diligence prompt. Tip: name the **competitors/market** and the **core
   technology** so the web and academic agents know what to dig into.
3. Hit **🚀 Run Deep Research** and watch the agents complete live.
4. Read the dashboard — a pinned **verdict + confidence** header with tabbed sections:
   **Overview** (scorecard + cited key metrics), **Analysis** (summary, sentiment,
   technical reality), **Red flags** (contradictions + questions), and **Sources**
   (clickable citations + raw agent notes). Download the whole thing as `.md`.

---

## Project layout

```
app.py            Streamlit UI (inputs, live agent status, dashboard)
graph.py          LangGraph wiring (parallel fan-out -> synthesis)
agents.py         The four agents (each a graph node)
state.py          Shared state TypedDict + ResearchReport output schema
llm.py            Single-provider (Gemini) model + embedding factory
requirements.txt  Pinned, verified-working dependency set
setup.* / run.*   Cross-platform setup & launch scripts
```

## Notes & troubleshooting
- **Web search returns nothing / errors:** DuckDuckGo occasionally rate-limits.
  The app degrades gracefully (the analyst notes the gap); just re-run.
- **Swap the LLM provider:** everything funnels through `llm.py` — change the two
  factory functions and you've repointed all agents at once.
- **Model choice:** pick `gemini-2.0-flash` (fast/free-tier) up to `gemini-2.5-pro`
  (deeper) in the sidebar.
