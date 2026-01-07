# Legal Document Summarizer (NLP)

A Streamlit-based NLP application that summarizes long legal and legislative documents using transformer-based models.  
Supports single-document and batch summarization with progress tracking and downloadable outputs.

---

## Features

- Transformer-based summarization (BART-large-CNN)
- Handles long legal documents via chunking
- Batch summarization for multiple bills (JSONL / TXT)
- Progress bar for batch processing
- Adjustable chunk size and summary length
- Download summaries as TXT or JSON
- Runs locally and deployable on Streamlit Cloud

---

## Model Used

- **facebook/bart-large-cnn**
- Fine-tuned for abstractive summarization
- Chunk-based strategy for long documents

---

## Tech Stack

- Python
- Hugging Face Transformers
- Streamlit
- PyTorch

---

## Input Formats

- `.txt` — Plain legal text
- `.jsonl` — One document per line with a `text` field

Example JSONL:
{"text": "SECTION 1. SHORT TITLE..."}
{"text": "SEC. 2. FINDINGS..."}

## How It Works

- Long documents are split into chunks
- Each chunk is summarized independently
- Chunk summaries are combined into a final summary
- Batch mode processes multiple documents with progress tracking

## Run Locally

- git clone https://github.com/rohan-crypto/legal-document-summarizer.git
- cd legal-doc-summarizer
- pip install -r requirements.txt
- streamlit run app.py

### Deployed using Streamlit Community Cloud at https://legal-document-summarizer-nnwruuquyw7vdlbygnc9xn.streamlit.app/

## Future Improvements

- Model comparison (BART vs PEGASUS)
- ROUGE-based evaluation
- PDF support
- API endpoint
