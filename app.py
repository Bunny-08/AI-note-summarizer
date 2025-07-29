import os
import torch
import streamlit as st
import fitz  # PyMuPDF
from fpdf import FPDF
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Avoid TensorFlow/Keras import issues
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Load LED model
@st.cache_resource
def load_led_model():
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
    return tokenizer, model

tokenizer, model = load_led_model()

# Clean text: remove weird symbols & repeated lines
def clean_text(text):
    lines = text.splitlines()
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line.replace("escription", "description"))
    return " ".join(cleaned)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Split text into chunks under token limit
def split_into_chunks(text, max_tokens=16000):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(tokenizer.encode(current + sentence)) < max_tokens:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# Summarize one chunk
def summarize_chunk(chunk):
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=16384, padding="max_length")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        max_length=512,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Export summary to PDF (safe ASCII only)
def generate_summary_pdf(summary_text, filename="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    def clean_line(line):
        return ''.join(c if ord(c) < 128 else ' ' for c in line)

    for line in summary_text.split("\n"):
        pdf.multi_cell(0, 10, clean_line(line))

    pdf.output(filename)
    return filename

# Streamlit UI
st.set_page_config(page_title="AI Long-Note Summarizer", page_icon="🧠")
st.title("🧠 AI Long-Note Summarizer (LED)")
st.markdown("Upload long PDFs and summarize them chunk by chunk using `allenai/led-base-16384`. Output is a PDF!")

uploaded_file = st.file_uploader("📎 Upload a long PDF", type=["pdf"])
input_text = ""

if uploaded_file:
    try:
        input_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Text extracted!")
        st.text_area("📄 Preview (first 2000 chars):", input_text[:2000], height=200)
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")

if not input_text:
    input_text = st.text_area("✍️ Or paste your notes here:", height=300)

if st.button("✨ Summarize"):
    if input_text.strip():
        with st.spinner("⏳ Cleaning and splitting..."):
            cleaned = clean_text(input_text)
            chunks = split_into_chunks(cleaned)

        final_summary = ""
        for i, chunk in enumerate(chunks):
            with st.spinner(f"🔍 Summarizing chunk {i+1}/{len(chunks)}..."):
                summary = summarize_chunk(chunk)
                final_summary += f"--- Summary {i+1} ---\n{summary}\n\n"

        st.subheader("📝 Final Summary")
        st.success(final_summary)

        pdf_path = generate_summary_pdf(final_summary)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Summary as PDF", f, file_name="summary.pdf", mime="application/pdf")

    else:
        st.warning("⚠️ Please upload a PDF or paste some content first.")
