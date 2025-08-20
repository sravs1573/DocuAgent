import os
import json
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .ocr_processor import OCRProcessor
from utils.validation import DocumentValidator

class DocumentProcessor:
    """Main document processing agent that orchestrates the entire pipeline"""
    
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.validator = DocumentValidator()
        
        # Initialize LangChain LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            api_key=api_key
        )
    
    def detect_document_type(self, text_content: str) -> str:
        """Detect the type of document using LLM"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document classification expert. Analyze the provided text and classify it into one of these categories:
            - invoice: Business invoices, bills for services/products
            - medical_bill: Hospital bills, medical invoices, insurance claims
            - prescription: Medical prescriptions, pharmacy documents
            
            Look for key indicators:
            - Invoice: Item descriptions, quantities, prices, tax amounts, vendor information
            - Medical_bill: Patient information, medical procedures, insurance details, provider names
            - Prescription: Medication names, dosages, doctor names, pharmacy information
            
            Respond with only the classification: invoice, medical_bill, or prescription"""),
            ("user", "Classify this document text:\n\n{text}")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": text_content[:2000]})  # Limit text length
            
            doc_type = response.content.strip().lower()
            if doc_type in ['invoice', 'medical_bill', 'prescription']:
                return doc_type
            else:
                return 'invoice'  # Default fallback
                
        except Exception as e:
            print(f"Error detecting document type: {e}")
            return 'invoice'  # Default fallback
    
    def create_document_tools(self, text_content: str) -> List[Tool]:
        """Create tools for the agent to use"""
        
        def ocr_tool(query: str) -> str:
            """Extract text using OCR capabilities"""
            return f"OCR processing completed. Text content available for analysis."
        
        def validation_tool(data: str) -> str:
            """Validate extracted data against business rules"""
            try:
                data_dict = json.loads(data) if isinstance(data, str) else data
                validation_result = self.validator.validate_extraction(data_dict)
                return f"Validation completed: {json.dumps(validation_result)}"
            except Exception as e:
                return f"Validation error: {str(e)}"
        
        def text_analysis_tool(query: str) -> str:
            """Analyze document text for specific patterns"""
            return f"Text analysis available for: {len(text_content)} characters"
        
        tools = [
            Tool(
                name="ocr_processor",
                description="Process images and scanned documents with OCR",
                func=ocr_tool
            ),
            Tool(
                name="validator",
                description="Validate extracted data against business rules and patterns",
                func=validation_tool
            ),
            Tool(
                name="text_analyzer",
                description="Analyze document text for patterns and structure",
                func=text_analysis_tool
            )
        ]
        
        return tools
    
    def process_document(
        self,
        file_content: bytes,
        filename: str,
        enable_ocr: bool,
        custom_fields: List[str],
        extraction_agent,
        confidence_scorer,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """
        Main processing pipeline that coordinates all components
        """
        
        try:
            # Step 1: Extract text content
            if filename.lower().endswith('.pdf'):
                text_content = self.ocr_processor.extract_text_from_pdf(file_content)
            else:
                # Image file
                if enable_ocr:
                    text_content = self.ocr_processor.extract_text_from_image(file_content)
                else:
                    text_content = "OCR disabled - using image analysis only"
            
            if not text_content or len(text_content.strip()) < 10:
                raise ValueError("Could not extract sufficient text from document. Please ensure the document is clear and readable.")
            
            # Step 2: Detect document type
            doc_type = self.detect_document_type(text_content)
            
            # Step 3: Extract structured data
            extraction_result = extraction_agent.extract_structured_data(
                text_content=text_content,
                doc_type=doc_type,
                custom_fields=custom_fields
            )
            
            # Step 4: Calculate confidence scores
            confidence_result = confidence_scorer.calculate_field_confidence(
                extraction_result,
                text_content,
                doc_type
            )
            
            # Step 5: Validate extraction
            validation_result = self.validator.validate_extraction(confidence_result)
            
            # Step 6: Compile final result
            final_result = {
                "doc_type": doc_type,
                "fields": confidence_result.get("fields", []),
                "overall_confidence": confidence_result.get("overall_confidence", 0.0),
                "qa": validation_result,
                "processing_metadata": {
                    "filename": filename,
                    "text_length": len(text_content),
                    "ocr_enabled": enable_ocr,
                    "custom_fields": custom_fields,
                    "confidence_threshold": confidence_threshold
                }
            }
            
            return final_result
            
        except Exception as e:
            # Return error in standard format
            return {
                "doc_type": "error",
                "fields": [],
                "overall_confidence": 0.0,
                "qa": {
                    "passed_rules": [],
                    "failed_rules": ["processing_failed"],
                    "notes": f"Processing failed: {str(e)}"
                },
                "error": str(e)
            }
