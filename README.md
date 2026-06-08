# Python RAG Application

A local retrieval-augmented generation (RAG) app using Streamlit, FAISS, BM25, and Google Gemini 3.5 Flash.

## Setup

1. Open a terminal in the workspace root.
2. Create a virtual environment:
   ```powershell
   C:/Python314/python.exe -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and set your `GOOGLE_API_KEY`.
5. Add `.txt` or `.pdf` files to the `docs/` folder.

## Run

```powershell
streamlit run main.py
```

## Notes

- The app uses `sentence-transformers` with `all-MiniLM-L6-v2`.
- The app builds an ensemble retriever combining BM25 and FAISS.
- Gemini is configured from `GOOGLE_API_KEY` in `.env`.
