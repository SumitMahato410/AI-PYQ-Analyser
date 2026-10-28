# AI PYQ Analyzer Pro

A local-first Streamlit app that extracts questions from text-based exam PDFs using a local Ollama model, classifies topics, analyzes topic frequency and year-wise trends, flags repeated questions, and exports CSV reports.

## Requirements
- Python 3.10+
- Ollama installed
- A local model such as `gemma3:1b`

## Setup (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama run gemma3:1b
```

Open another terminal in the project directory and run:
```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## Notes and limitations
- Use text-selectable PDFs. Scanned PDFs require OCR, which is not included.
- AI extraction and classification can be wrong. Verify records against original papers.
- Exact duplicate matching is conservative. Repeated question occurrences across years are retained to support frequency analysis.
- The priority score is a heuristic based on historical frequency, year coverage, and recency. It is not a probability that a topic will appear on a future exam.
- For cloud deployment, replace local Ollama with a hosted model endpoint or host Ollama on a reachable server. A cloud app cannot normally call Ollama on your laptop via `localhost`.
