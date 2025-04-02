import os
import re
import PyPDF2
import spacy
import nltk
import pandas as pd
import streamlit as st
from nltk.tokenize import sent_tokenize
from datetime import datetime
import base64

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        # If model isn't downloaded, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

class RequirementsExtractor:
    def __init__(self):
        self.pdf_text = None
        self.extracted_requirements = []
        download_nltk_data()
        self.nlp = load_spacy_model()
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file."""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        total_pages = len(reader.pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num in range(total_pages):
            status_text.text(f"Processing page {page_num+1}/{total_pages}...")
            text += reader.pages[page_num].extract_text()
            progress_bar.progress((page_num + 1) / total_pages)
            
        status_text.text(f"Successfully extracted {len(text)} characters from {total_pages} pages.")
        return text

    def identify_requirements(self, text):
        """Identify requirements in the extracted text."""
        st.info("Analyzing text for requirements...")
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        st.text(f"Found {len(sentences)} sentences to analyze.")
        
        requirements = []
        
        # Requirement patterns and indicators
        requirement_indicators = [
            r"(?i).*\b(shall|must|should|will|need to|has to|requires|required to)\b.*",
            r"(?i).*\b(necessary|mandatory|essential|important|critical)\b.*",
            r"(?i).*\b(system|application|platform|software|solution|user)\b.*\b(shall|must|should|will)\b.*",
            r"(?i).*\b(functionality|feature|capability)\b.*",
            r"(?i).*\b(ability to|enable|allow)\b.*",
        ]
        
        # Analyze each sentence
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sentence in enumerate(sentences):
            if i % 10 == 0:  # Update status less frequently to improve performance
                status_text.text(f"Analyzed {i}/{len(sentences)} sentences...")
                progress_bar.progress(min((i + 1) / len(sentences), 1.0))
                
            # Clean and normalize sentence
            clean_sentence = sentence.strip()
            
            # Skip very short sentences
            if len(clean_sentence.split()) < 3:
                continue
                
            # Check if the sentence matches any requirement pattern
            is_requirement = False
            for pattern in requirement_indicators:
                if re.match(pattern, clean_sentence):
                    is_requirement = True
                    break
            
            if is_requirement:
                # Use spaCy for additional filtering
                doc = self.nlp(clean_sentence)
                
                # Check for key verbs that indicate requirements
                has_key_verbs = any(token.lemma_ in ["shall", "must", "should", "need", "require", "provide", "support", "allow", "enable"] 
                                for token in doc)
                
                if has_key_verbs:
                    requirements.append(clean_sentence)
        
        status_text.text(f"Found {len(requirements)} potential requirements.")
        return requirements

    def create_download_link(self, requirements, filename="requirements.txt"):
        """Create a download link for the requirements text file."""
        content = "# EXTRACTED REQUIREMENTS\n"
        content += f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, req in enumerate(requirements, 1):
            content += f"REQ-{i:03d}: {req}\n\n"
        
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Requirements (.txt)</a>'
        return href

    def create_download_link_csv(self, requirements, filename="requirements.csv"):
        """Create a download link for the requirements CSV file."""
        df = pd.DataFrame({
            'ID': [f"REQ-{i:03d}" for i in range(1, len(requirements) + 1)],
            'Requirement': requirements,
            'Priority': ['Medium'] * len(requirements),
            'Status': ['New'] * len(requirements)
        })
        
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Requirements (.csv)</a>'
        return href

def main():
    st.set_page_config(
        page_title="Requirements Extraction Tool",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Requirements Extraction Tool")
    st.markdown("""
    This application extracts requirements from PDF documents using natural language processing.
    Upload a PDF file to get started.
    """)
    
    # Initialize session state variables if they don't exist
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = None
    if 'requirements' not in st.session_state:
        st.session_state.requirements = []
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    
    # Sidebar
    st.sidebar.header("Options")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])
    
    extractor = RequirementsExtractor()
    
    col1, col2 = st.columns(2)
    
    # Process PDF button
    if uploaded_file is not None:
        if st.sidebar.button("Extract Requirements"):
            with st.spinner("Processing PDF..."):
                st.session_state.file_name = uploaded_file.name
                st.session_state.pdf_text = extractor.extract_text_from_pdf(uploaded_file)
                st.session_state.requirements = extractor.identify_requirements(st.session_state.pdf_text)
    
    # Display input text
    with col1:
        st.header("Document Text")
        if st.session_state.pdf_text:
            st.text_area("Extracted Text (Preview)", st.session_state.pdf_text[:5000] + "..." if len(st.session_state.pdf_text) > 5000 else st.session_state.pdf_text, height=400)
            st.caption(f"Total text length: {len(st.session_state.pdf_text)} characters")
        else:
            st.info("Upload a PDF and click 'Extract Requirements' to view the document text here.")
            
    # Display requirements
    with col2:
        st.header("Extracted Requirements")
        if st.session_state.requirements:
            df = pd.DataFrame({
                'ID': [f"REQ-{i:03d}" for i in range(1, len(st.session_state.requirements) + 1)],
                'Requirement': st.session_state.requirements
            })
            st.dataframe(df, height=400)
            st.success(f"Found {len(st.session_state.requirements)} potential requirements!")
            
            # Download links
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = st.session_state.file_name.replace('.pdf', '') if st.session_state.file_name else "requirements"
            txt_filename = f"{base_filename}_requirements_{timestamp}.txt"
            csv_filename = f"{base_filename}_requirements_{timestamp}.csv"
            
            st.markdown(extractor.create_download_link(st.session_state.requirements, txt_filename), unsafe_allow_html=True)
            st.markdown(extractor.create_download_link_csv(st.session_state.requirements, csv_filename), unsafe_allow_html=True)
        else:
            st.info("Extracted requirements will appear here after processing.")
    
    # Additional information
    st.markdown("---")
    st.header("How It Works")
    st.markdown("""
    This tool identifies requirements by:
    1. Extracting text from your uploaded PDF
    2. Breaking the text into sentences
    3. Analyzing each sentence using NLP techniques
    4. Identifying sentences that match requirement patterns
    5. Filtering based on linguistic indicators of requirements
    
    The tool looks for:
    - Modal verbs: shall, must, should, will
    - Action phrases: need to, required to, has to
    - Descriptors: necessary, mandatory, essential
    - Subject+modal: "system shall", "user must", etc.
    """)

if __name__ == "__main__":
    main()