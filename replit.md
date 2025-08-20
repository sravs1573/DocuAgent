# Overview

An intelligent document processing system that automatically extracts structured data from PDFs and images using OpenAI's GPT-4o. The system specializes in processing invoices, medical bills, and prescriptions with advanced confidence scoring and validation capabilities. Built with Streamlit for an interactive web interface and LangChain for agent orchestration.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Agent-Based Architecture
The application follows a modular agent-based design with specialized components:

- **DocumentProcessor**: Main orchestration agent that coordinates the entire pipeline, handles document type detection using LLM classification
- **ExtractionAgent**: LLM-powered extraction using GPT-4o with structured Pydantic schemas for consistent output formatting
- **OCRProcessor**: Handles text extraction from PDFs (PyMuPDF) and images (Tesseract) with preprocessing for better accuracy

## Document Processing Pipeline
1. **File Upload & Validation**: Supports PDF and image formats with size/type validation
2. **OCR Text Extraction**: Automatic text extraction with fallback OCR for scanned documents
3. **Document Classification**: LLM-based classification into invoice, medical_bill, or prescription categories
4. **Structured Data Extraction**: Schema-based extraction using Pydantic models for consistent output
5. **Confidence Scoring**: Multi-factor scoring system considering text clarity, context strength, pattern matching, and consistency
6. **Validation Engine**: Business rule validation with cross-field consistency checks and regex pattern matching

## Data Models
Structured schemas using Pydantic for:
- **InvoiceSchema**: Invoice numbers, dates, vendor/customer info, line items, amounts
- **MedicalBillSchema**: Patient info, procedures, insurance details, charges
- **PrescriptionSchema**: Patient/doctor info, medications, dosages, pharmacy details

## Confidence Scoring System
Advanced multi-factor confidence calculation:
- Text clarity assessment based on OCR quality
- Context strength evaluation using surrounding text
- Pattern matching for structured data (dates, amounts, phone numbers)
- Cross-field consistency validation
- Weighted scoring algorithm with configurable parameters

## Frontend Architecture
Streamlit-based web interface featuring:
- File upload with drag-and-drop support
- Real-time processing with progress indicators
- Interactive confidence visualization with progress bars
- JSON/CSV export capabilities
- Responsive layout with sidebar controls

## Error Handling & Robustness
- Comprehensive input validation and sanitization
- Graceful fallback mechanisms for OCR failures
- Timeout handling for LLM requests
- Detailed error reporting and logging
- File size and format restrictions for security

# External Dependencies

## Core LLM Service
- **OpenAI GPT-4o**: Primary language model for document classification and data extraction
- **LangChain**: Agent framework for LLM orchestration and prompt management

## Document Processing
- **PyMuPDF (fitz)**: PDF text extraction and rendering
- **Tesseract OCR**: Image-to-text conversion for scanned documents
- **OpenCV**: Image preprocessing and enhancement
- **Pillow (PIL)**: Image format handling and manipulation

## Web Framework & UI
- **Streamlit**: Web application framework and user interface
- **Pandas**: Data manipulation and CSV export functionality

## Data Validation & Structure
- **Pydantic**: Schema definition and data validation
- **NumPy**: Numerical computations for confidence scoring

## Environment Requirements
- **OPENAI_API_KEY**: Required environment variable for OpenAI API access
- Python 3.8+ runtime environment
- System-level Tesseract OCR installation