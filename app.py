import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json

#@st.cache_resource(show_spinner=True)

# Load pretrained model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1) # device=0 uses MPS on Mac

# Function to chunk text
# BART works on tokens so, instead of counting by words count with tokens. Hard limit of BART is 1024 tokens. So it will break if slider is >650.
# OLD code: def chunk_text(text, max_words=500):
def chunk_text(text, tokenizer, max_tokens=900):
    # words = text.split()
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    #for i in range(0, len(words), max_words):
    #    chunk = " ".join(words[i:i+max_words])
    #    chunks.append(chunk)
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# Function to summarize text with chunking
def summarize(text, max_words_per_chunk=500, max_summary_length=256):
    # Array of chunks of 500 words. But instead of words now using tokens.
    #chunks = chunk_text(text, max_words=max_words_per_chunk)
    # * 1.3 because it is approx word to token conversion and Keeps slider intuitive for users and Caps safely below 1024.
    chunks = chunk_text(text, tokenizer=tokenizer, max_tokens=min(int(max_words_per_chunk * 1.3), 900))
    chunk_summaries = []
    for chunk in chunks:
        # Summarizing every chunk and storing in array
        summary = summarizer(chunk, max_length=max_summary_length, min_length=50, do_sample=False)
        chunk_summaries.append(summary[0]["summary_text"])
    if len(chunk_summaries) > 1:
        # Combine chunk summaries into final summary
        final_summary = summarizer(" ".join(chunk_summaries), max_length=max_summary_length, min_length=50, do_sample=False)
        return final_summary[0]["summary_text"]
    else:
        return chunk_summaries[0]
        
MAX_DOCS = 10  # Maximum number of documents to summarize in batch
st.title("Legal Document Summarizer")
st.write("Paste a legal document below and get a concise summary.")

# Input selection
input_option = st.radio("Choose input type:", ["Paste Text", "Upload File"])

text_to_summarize = ""

if input_option == "Paste Text":
    # Text input
    input_text = st.text_area("Enter legal text here:", height=300)
    if input_text.strip() != "":
        text_to_summarize = input_text

elif input_option == "Upload File":
    # File Uplaoder
    # Removed the argument type=["txt", "jsonl"] because Streamlit’s file_uploader filters by file extension at the OS level (OS level filtering). 
    # So '.jsonl' files don’t appear when Streamlit restricts types.
    uploaded_files = st.file_uploader("Upload file(s) (.txt or .jsonl)", accept_multiple_files=True)
    if uploaded_files is not None:
        batch_mode = st.checkbox("Batch summarize multiple bills")
        text_entries = []
        for file in uploaded_files:
            file_content = file.read().decode("utf-8")
            # if uploaded_file.type == "application/jsonl": this was causing type error
            if file.name.endswith(".jsonl"):
                lines = file_content.splitlines()
                st.write(f"Found {len(lines)} records in JSONL file.")
                for line in lines:
                    obj = json.loads(line)
                    if "text" in obj:
                        text_entries.append(obj["text"])
            elif file.name.endswith(".txt"):    # Plain text
                text_entries.append(file_content)
            else:
                st.warning(f"Skipped unsupported file: {file.name}")
        if batch_mode:
            num_docs = min(len(text_entries), MAX_DOCS)
            if len(text_entries) > MAX_DOCS:
                st.warning(f"Only summarizing first {MAX_DOCS} documents (MAX_DOCS limit).")
            text_to_summarize = text_entries[:num_docs]  # List of texts
        else:
            text_to_summarize = " ".join(text_entries[:MAX_DOCS]) # single summary

# Sliders for chunk size and summary length
# Correcting the max_words_chunk slider's limits.
max_words_chunk = st.slider("Approx words per chunk (auto-token-safe)", 200, 800, 500)
max_summary_len = st.slider("Maximum summary length", 50, 512, 256)

# Summarize button
if st.button("Generate Summary"):
    if text_to_summarize == "" or text_to_summarize is None:
        st.warning("Please enter text or upload a file!")
    else:
        with st.spinner("Summarizing..."):
            if isinstance(text_to_summarize, list):  # batch mode
                all_summaries = []
                json_summaries = []
                progress_bar = st.progress(0) # Set progress bar to 0
                for i, t in enumerate(text_to_summarize):
                    summary = summarize(t, max_words_per_chunk=max_words_chunk, max_summary_length=max_summary_len)
                    # .txt format
                    all_summaries.append(f"Bill {i+1} Summary:\n{summary}\n")
                    # .json format
                    json_summaries.append({
                        "bill_id": i + 1,
                        "summary": summary
                    })
                    progress = (i + 1) / len(text_to_summarize)
                    progress_bar.progress(progress)
                summary_output = "\n\n".join(all_summaries)
                json_output = json.dumps(json_summaries, indent=2)
            else: # single document
                summary_output = summarize(text_to_summarize, max_words_per_chunk=max_words_chunk, max_summary_length=max_summary_len)
        st.success("Summary Generated!")
        st.text_area("Summary", summary_output, height=400)

        # Download buttons
        st.download_button(
            label="Download Summary as TXT",
            data=summary_output,
            file_name="legal_summaries.txt",
            mime="text/plain"
        )
        # In single document mode, json_output won’t exist so wrapping it in batch mode
        if isinstance(text_to_summarize, list): # batch mode
            st.download_button(
                label="Download Summaries as JSON",
                data=json_output,
                file_name="legal_summaries.json",
                mime="application/json"
            )
