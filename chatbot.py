import os
import re
import PyPDF2
import spacy
import nltk
import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.tokenize import sent_tokenize
from datetime import datetime
import base64

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model_path = "fine_tuned_distilbert"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device

class RequirementsExtractor:
    def __init__(self):
        self.extracted_requirements = []
        download_nltk_data()
        self.nlp = load_spacy_model()
        self.tokenizer, self.model, self.device = load_model()

    def extract_text_from_pdf(self, pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def identify_requirements(self, text):
        sentences = sent_tokenize(text)
        requirements = []
        requirement_patterns = [
            r"(?i).*(shall|must|should|will|need to|has to|requires|required to).*",
            r"(?i).*(necessary|mandatory|essential|important|critical).*",
        ]
        
        for sentence in sentences:
            if len(sentence.split()) < 3:
                continue
            if any(re.match(pattern, sentence) for pattern in requirement_patterns):
                requirements.append(sentence)
        
        return requirements
    
    def predict_requirement(self, text):
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        return "FR" if prediction == 0 else "NFR"
    
    def create_download_link_csv(self, df, filename="classified_requirements.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Classified Requirements (.csv)</a>'
        return href

def main():
    st.set_page_config(page_title="Requirements Extraction & Classification", page_icon="📝", layout="wide")
    st.title("📝 Requirements Extraction & Classification Tool")
    
    uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    extractor = RequirementsExtractor()
    
    if uploaded_files and st.sidebar.button("Extract & Classify Requirements"):
        with st.spinner("Processing PDFs..."):
            all_requirements = []
            
            for file in uploaded_files:
                text = extractor.extract_text_from_pdf(file)
                requirements = extractor.identify_requirements(text)
                
                all_requirements.extend(
                    [{"Requirement Text": req, "Predicted Type": extractor.predict_requirement(req), "Source File": file.name} for req in requirements]
                )
            
            df = pd.DataFrame(all_requirements)
            st.success(f"Extracted and classified {len(df)} requirements from {len(uploaded_files)} files!")
            st.dataframe(df, height=400)
            st.markdown(extractor.create_download_link_csv(df), unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()