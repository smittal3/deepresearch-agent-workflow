# 🔭 Deep Research Agent

A multi-agent research assistant. Upload a document (e.g. a startup pitch deck),
ask a question, and **four specialist agents investigate in parallel** and return
a decision-ready dashboard — _"how competitive is this, and should I invest?"_

| Agent | Role | Tech |
|---|---|---|
| 📄 **Document Analyst** | Extracts claims & metrics from your file (PDF/DOCX/PPTX/TXT) | Document-in-context reading |
| 🌐 **Web Researcher** | Current sources, consensus & disagreement | DuckDuckGo search |
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

> **One key for everything.** All four agents *and* the image-PDF vision OCR run
> through **OpenRouter**, so each teammate only needs a single `OPENROUTER_API_KEY`.
> Swap the model in `llm.py` to any OpenRouter model id.

---

## Quick start

### 0. Get an API key (30 seconds)
Go to **https://openrouter.ai/keys** → **Create Key** → copy it.

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

## 🪟 Windows quick start (copy-paste)

A clean top-to-bottom path for a teammate on Windows starting from nothing:

**1. Install Python 3.11** from [python.org](https://www.python.org/downloads/) — on the
first installer screen, **tick "Add python.exe to PATH"** before clicking Install.
(Python 3.9 will *not* work.)

**2. Get the code** — either `git clone <repo-url>`, or download the repo as a ZIP
(green **Code** button → *Download ZIP*) and extract it.

**3. Open the folder in a terminal.** In File Explorer, open the project folder, type
`cmd` in the address bar, and press Enter — that opens Command Prompt already in the
right directory.

**4. One-time setup:**
```bat
setup.bat
```
(or just double-click `setup.bat` in Explorer). This builds a `.venv` and installs
everything — first run takes a few minutes.

**5. Add your OpenRouter key** — copy `.env.example` to `.env` and paste your key in, *or*
skip this and paste it into the app's sidebar at runtime.

**6. Launch (now and every time after):**
```bat
run.bat
```
The app opens in your browser at http://localhost:8501.

> **If you hit an error about Python syntax / `LLM | None`:** your default `python` is
> older than 3.10. Install Python 3.11 and make sure it's on PATH, then delete the
> `.venv` folder and re-run `setup.bat`.

---

## Using it

1. Click **Try sample case study** (loads the bundled startup files + a VC prompt in
   one click), or **upload** your own PDF / DOCX / PPTX / TXT — *several at once is fine,
   and documents are optional.* Image-based PDFs (e.g. pitch decks exported as images)
   are transcribed automatically via a vision model (rasterized with PyMuPDF) — no OCR install needed.
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
llm.py            Single-provider (OpenRouter) model factory
requirements.txt  Pinned, verified-working dependency set
setup.* / run.*   Cross-platform setup & launch scripts
```

## Notes & troubleshooting
- **Web search returns nothing / errors:** DuckDuckGo occasionally rate-limits.
  The app degrades gracefully (the analyst notes the gap); just re-run.
- **Swap the model:** `MODEL` in `llm.py` takes any OpenRouter model id
  (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`). The synthesis report
  needs a tool-calling model; image-PDF OCR needs a multimodal one (the default is both).
- **No embeddings needed:** OpenRouter has no embeddings endpoint, so the Document
  Analyst passes the document straight into context (no vector index) — simpler and
  reliable for typical doc sizes; very large docs are truncated with a note.
