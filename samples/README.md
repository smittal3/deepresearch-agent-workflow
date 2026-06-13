# Sample case studies

Each **subdirectory** here is one self-contained sample case. The app shows a button
per case on the input screen; clicking it loads *every* file in that subdirectory as
the documents for the run and fills in the case's research query.

Current cases:

| Folder | Case | What it demonstrates |
|---|---|---|
| `startup_due_diligence/` | VC due-diligence on a startup | A compliance audit contradicts the pitch deck → integrity red flag |
| `smartphone_ban_policy/` | School smartphone-ban decision | A viral parent guide's pseudo-scientific claims fail web + academic fact-checking |

Cases are declared in `samples.json`:

```json
{
  "cases": [
    {
      "id": "startup",
      "label": "Startup due-diligence",
      "emoji": "🚀",
      "dir": "startup_due_diligence",
      "blurb": "Short tooltip shown on hover.",
      "query": "The research prompt that gets filled in."
    }
  ]
}
```

**To add a case:** create a new subdirectory, drop your documents in it
(`.pdf`, `.docx`, `.pptx`, `.txt`, `.md`), and add an entry to `samples.json`
pointing `dir` at the folder. No code changes needed. A case whose folder has no
readable files is automatically hidden.

**Image-based PDFs are supported:** if a PDF has no extractable text layer (e.g. a
pitch deck exported as images), the app automatically transcribes it with Gemini's
native PDF vision — no OCR install required.
