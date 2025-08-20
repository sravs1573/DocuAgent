import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Any

class OCRProcessor:
    """Handles OCR processing for PDFs and images with table detection"""
    
    def __init__(self):
        # Configure tesseract if needed
        self.tesseract_config = '--oem 3 --psm 6'
        
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            text_content = ""
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                # Extract text - compatible with different PyMuPDF versions
                page_text = ""
                try:
                    page_text = page.get_text()
                except AttributeError:
                    # Fallback for older versions
                    page_text = page.getText() if hasattr(page, 'getText') else ""
                
                text_content += page_text
                
                # If no text found, try OCR on the page
                if not page_text.strip():
                    try:
                        pix = page.get_pixmap() if hasattr(page, 'get_pixmap') else page.getPixmap()
                        img_data = pix.tobytes("png") if hasattr(pix, 'tobytes') else pix.getImageData("png")
                        ocr_text = self.extract_text_from_image(img_data)
                        text_content += f"\n[OCR Page {page_num + 1}]\n{ocr_text}\n"
                    except Exception as ocr_error:
                        print(f"OCR failed for page {page_num + 1}: {ocr_error}")
            
            pdf_document.close()
            return text_content
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_from_image(self, image_content: bytes) -> str:
        """Extract text from image using Tesseract OCR"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content))
            
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess the image
            processed_image = self.preprocess_image(opencv_image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Failed to extract text from image: {str(e)}")
    
    def detect_tables(self, image_content: bytes) -> List[Dict[str, Any]]:
        """Detect and extract tables from images"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content))
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find contours (potential table regions)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:  # Filter small contours
                    table_roi = opencv_image[y:y+h, x:x+w]
                    table_text = pytesseract.image_to_string(table_roi, config=self.tesseract_config)
                    
                    tables.append({
                        'table_id': i,
                        'bbox': [x, y, x+w, y+h],
                        'text': table_text.strip()
                    })
            
            return tables
            
        except Exception as e:
            print(f"Table detection failed: {e}")
            return []
    
    def extract_structured_content(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract structured content including text and tables"""
        try:
            result = {
                'text': '',
                'tables': [],
                'metadata': {
                    'filename': filename,
                    'processing_method': ''
                }
            }
            
            if filename.lower().endswith('.pdf'):
                result['text'] = self.extract_text_from_pdf(content)
                result['metadata']['processing_method'] = 'pdf_extraction'
            else:
                result['text'] = self.extract_text_from_image(content)
                result['tables'] = self.detect_tables(content)
                result['metadata']['processing_method'] = 'ocr_image'
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to extract structured content: {str(e)}")
