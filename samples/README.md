# Sample case study

All files here (`.pdf`, `.docx`, `.pptx`, `.txt`, `.md`) make up **one** sample
case study. In the app, the **"Try sample case study"** button loads *every* file
here as the documents for the run, and fills the research query.

The query comes from `samples.json`:

```json
{ "case_query": "Act as a Venture Capitalist. ..." }
```

Edit `case_query` to change the prompt. Replace the files with your own to build a
different case study — no code changes needed.

**Image-based PDFs are supported:** if a PDF has no extractable text layer (e.g. a
pitch deck exported as images), the app automatically transcribes it with Gemini's
native PDF vision — no OCR install required.
