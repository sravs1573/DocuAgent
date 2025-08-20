import streamlit as st
import json
import base64
from io import BytesIO
import pandas as pd
from agents.document_processor import DocumentProcessor
from agents.extraction_agent import ExtractionAgent
from utils.confidence_scoring import ConfidenceScorer
from utils.file_handlers import FileHandler

# Configure page
st.set_page_config(
    page_title="Document Processing Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_result' not in st.session_state:
        st.session_state.processed_result = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def display_confidence_bars(fields_data):
    """Display confidence bars for each field"""
    st.subheader("Field Confidence Scores")
    
    for field in fields_data:
        confidence = field.get('confidence', 0)
        name = field.get('name', 'Unknown')
        value = field.get('value', 'N/A')
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.text(f"{name}: {value}")
        
        with col2:
            # Create a progress bar for confidence
            st.progress(confidence)
        
        with col3:
            # Color code the confidence score
            if confidence >= 0.8:
                st.success(f"{confidence:.2f}")
            elif confidence >= 0.6:
                st.warning(f"{confidence:.2f}")
            else:
                st.error(f"{confidence:.2f}")

def display_validation_results(qa_data):
    """Display validation results"""
    st.subheader("Validation Results")
    
    passed_rules = qa_data.get('passed_rules', [])
    failed_rules = qa_data.get('failed_rules', [])
    notes = qa_data.get('notes', '')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"Passed Rules ({len(passed_rules)})")
        for rule in passed_rules:
            st.text(f"✅ {rule}")
    
    with col2:
        if failed_rules:
            st.error(f"Failed Rules ({len(failed_rules)})")
            for rule in failed_rules:
                st.text(f"❌ {rule}")
        else:
            st.success("All validation rules passed!")
    
    if notes:
        st.info(f"Notes: {notes}")

def main():
    initialize_session_state()
    
    st.title("📄 Intelligent Document Processing Agent")
    st.markdown("Upload a document (PDF or image) to extract structured data with confidence scoring")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Optional custom fields
        st.subheader("Custom Fields (Optional)")
        custom_fields = st.text_area(
            "Specify additional fields to extract (one per line):",
            placeholder="PatientName\nDoctorName\nTotalAmount\nDate",
            height=100
        )
        
        # Processing options
        st.subheader("Processing Options")
        enable_ocr = st.checkbox("Enable OCR for scanned documents", value=True)
        confidence_threshold = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.5, 0.1)
        
        # Model selection
        st.subheader("Model Configuration")
        model_choice = st.selectbox(
            "Select OpenAI Model:",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Display file preview
            if uploaded_file.type.startswith('image/'):
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("🚀 Process Document", disabled=st.session_state.processing):
                st.session_state.processing = True
                
                with st.spinner("Processing document..."):
                    try:
                        # Initialize processors
                        file_handler = FileHandler()
                        doc_processor = DocumentProcessor()
                        extraction_agent = ExtractionAgent(model=model_choice)
                        confidence_scorer = ConfidenceScorer()
                        
                        # Process the uploaded file
                        file_content = file_handler.process_uploaded_file(uploaded_file)
                        
                        # Parse custom fields
                        custom_field_list = [field.strip() for field in custom_fields.split('\n') if field.strip()] if custom_fields else []
                        
                        # Run the complete processing pipeline
                        result = doc_processor.process_document(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            enable_ocr=enable_ocr,
                            custom_fields=custom_field_list,
                            extraction_agent=extraction_agent,
                            confidence_scorer=confidence_scorer,
                            confidence_threshold=confidence_threshold
                        )
                        
                        st.session_state.processed_result = result
                        st.session_state.processing = False
                        st.success("Document processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.processing = False
                        st.error(f"Error processing document: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.header("Results")
        
        if st.session_state.processed_result:
            result = st.session_state.processed_result
            
            # Overall metrics
            st.subheader("Document Summary")
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Document Type", result.get('doc_type', 'Unknown'))
            with metric_cols[1]:
                st.metric("Overall Confidence", f"{result.get('overall_confidence', 0):.2f}")
            with metric_cols[2]:
                st.metric("Fields Extracted", len(result.get('fields', [])))
            
            # Confidence visualization
            fields_data = result.get('fields', [])
            if fields_data:
                display_confidence_bars(fields_data)
            
            # Validation results
            qa_data = result.get('qa', {})
            if qa_data:
                display_validation_results(qa_data)
            
            # JSON output
            st.subheader("Extracted Data (JSON)")
            
            # Format JSON for display
            formatted_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.code(formatted_json, language='json')
            
            # Download buttons
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                if uploaded_file is not None:
                    st.download_button(
                        label="📥 Download JSON",
                        data=formatted_json,
                        file_name=f"extracted_data_{uploaded_file.name}.json",
                        mime="application/json"
                    )
            
            with download_col2:
                # Create CSV for tabular view
                if fields_data and uploaded_file is not None:
                    df = pd.DataFrame(fields_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv,
                        file_name=f"extracted_fields_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
            
            # Detailed field analysis
            if fields_data:
                st.subheader("Detailed Field Analysis")
                
                for i, field in enumerate(fields_data):
                    with st.expander(f"{field.get('name', f'Field {i+1}')} - Confidence: {field.get('confidence', 0):.2f}"):
                        st.json(field)
        
        else:
            st.info("👆 Upload and process a document to see results here")

if __name__ == "__main__":
    main()
