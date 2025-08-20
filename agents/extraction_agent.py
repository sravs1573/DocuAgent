import os
import json
import re
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from models.schemas import InvoiceSchema, MedicalBillSchema, PrescriptionSchema
from pydantic import ValidationError

class ExtractionAgent:
    """LLM-powered extraction agent with structured output"""
    
    def __init__(self, model: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            temperature=0,
            model=model,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            openai_api_key=api_key
        )
        
        # Schema mapping
        self.schemas = {
            'invoice': InvoiceSchema,
            'medical_bill': MedicalBillSchema,
            'prescription': PrescriptionSchema
        }
    
    def get_extraction_prompt(self, doc_type: str, custom_fields: Optional[List[str]] = None) -> ChatPromptTemplate:
        """Generate extraction prompt based on document type"""
        
        base_fields = {
            'invoice': [
                'invoice_number', 'invoice_date', 'due_date', 'vendor_name', 
                'vendor_address', 'customer_name', 'customer_address',
                'subtotal', 'tax_amount', 'total_amount', 'line_items'
            ],
            'medical_bill': [
                'patient_name', 'patient_id', 'provider_name', 'provider_address',
                'service_date', 'diagnosis', 'procedures', 'insurance_company',
                'total_charges', 'insurance_paid', 'patient_responsibility'
            ],
            'prescription': [
                'patient_name', 'doctor_name', 'pharmacy_name', 'prescription_date',
                'medications', 'dosage_instructions', 'refills', 'pharmacy_phone'
            ]
        }
        
        fields_to_extract = base_fields.get(doc_type, base_fields['invoice'])
        if custom_fields:
            fields_to_extract.extend(custom_fields)
        
        fields_list = '\n'.join([f'- {field}' for field in fields_to_extract])
        
        system_prompt = f"""You are an expert document data extraction specialist. Extract structured information from {doc_type} documents.

EXTRACTION REQUIREMENTS:
1. Extract the following fields with high precision:
{fields_list}

2. For each field, provide:
   - name: field name
   - value: extracted value (null if not found)
   - confidence: confidence score (0.0-1.0) based on text clarity and context
   - source: {{page: page_number, bbox: [x1,y1,x2,y2]}} (estimate coordinates if needed)

3. CONFIDENCE SCORING GUIDELINES:
   - 0.9-1.0: Clear, unambiguous text with strong context
   - 0.7-0.9: Clear text with some ambiguity or weak context
   - 0.5-0.7: Partially clear with moderate ambiguity
   - 0.3-0.5: Unclear text or high ambiguity
   - 0.0-0.3: Very poor quality or missing

4. SPECIAL HANDLING:
   - Dates: Extract in YYYY-MM-DD format, handle various input formats
   - Amounts: Extract as numbers, handle currency symbols and formatting
   - Lists: For line_items, medications, procedures - extract as structured arrays
   - Addresses: Combine multi-line addresses into single strings

5. OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
  "fields": [
    {{
      "name": "field_name",
      "value": "extracted_value",
      "confidence": 0.85,
      "source": {{"page": 1, "bbox": [100, 200, 300, 220]}}
    }}
  ]
}}

Be extremely careful with numerical values, dates, and proper names. If unsure, lower the confidence score rather than guessing."""

        user_prompt = """Extract structured data from this {doc_type} document:

DOCUMENT TEXT:
{text}

Remember to:
- Be precise with numerical values and dates
- Provide realistic confidence scores
- Include source information for each field
- Return valid JSON only"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    
    def extract_structured_data(
        self, 
        text_content: str, 
        doc_type: str, 
        custom_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract structured data using LLM with retry mechanism"""
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Get prompt template
                prompt = self.get_extraction_prompt(doc_type, custom_fields)
                
                # Create chain
                chain = prompt | self.llm
                
                # Invoke with response format for JSON
                response = chain.invoke({
                    "doc_type": doc_type,
                    "text": text_content[:4000]  # Limit text length to avoid token limits
                })
                
                # Parse JSON response
                content = response.content if hasattr(response, 'content') else str(response)
                if hasattr(content, 'strip'):
                    content = content.strip()
                else:
                    content = str(content).strip()
                
                # Clean up potential markdown formatting
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                
                result = json.loads(content)
                
                # Validate result structure
                if 'fields' not in result:
                    raise ValueError("Response missing 'fields' key")
                
                # Post-process and validate fields
                processed_fields = self.post_process_fields(result['fields'], doc_type)
                
                return {
                    'fields': processed_fields,
                    'extraction_metadata': {
                        'model': self.llm.model_name,
                        'attempt': attempt + 1,
                        'doc_type': doc_type,
                        'custom_fields': custom_fields or []
                    }
                }
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing error: {e}"
                print(f"Attempt {attempt + 1} failed: {last_error}")
                continue
                
            except Exception as e:
                last_error = f"Extraction error: {e}"
                print(f"Attempt {attempt + 1} failed: {last_error}")
                continue
        
        # If all retries failed, return empty result
        return {
            'fields': [],
            'extraction_metadata': {
                'model': self.llm.model_name,
                'attempts': max_retries,
                'error': last_error,
                'doc_type': doc_type
            }
        }
    
    def post_process_fields(self, fields: List[Dict], doc_type: str) -> List[Dict]:
        """Post-process extracted fields for consistency and validation"""
        
        processed_fields = []
        
        for field in fields:
            processed_field = {
                'name': field.get('name', ''),
                'value': field.get('value'),
                'confidence': max(0.0, min(1.0, float(field.get('confidence', 0)))),
                'source': field.get('source', {'page': 1, 'bbox': [0, 0, 100, 20]})
            }
            
            # Type-specific post-processing
            processed_field = self.apply_field_validation(processed_field, doc_type)
            processed_fields.append(processed_field)
        
        return processed_fields
    
    def apply_field_validation(self, field: Dict, doc_type: str) -> Dict:
        """Apply field-specific validation and formatting"""
        
        field_name = field['name'].lower()
        value = field['value']
        
        if not value:
            return field
        
        try:
            # Date fields
            if 'date' in field_name:
                formatted_date = self.normalize_date(str(value))
                if formatted_date:
                    field['value'] = formatted_date
                else:
                    field['confidence'] *= 0.7  # Reduce confidence for invalid dates
            
            # Amount fields
            elif any(keyword in field_name for keyword in ['amount', 'total', 'subtotal', 'tax', 'paid', 'charges']):
                normalized_amount = self.normalize_amount(str(value))
                if normalized_amount is not None:
                    field['value'] = normalized_amount
                else:
                    field['confidence'] *= 0.6  # Reduce confidence for invalid amounts
            
            # Phone number fields
            elif 'phone' in field_name:
                normalized_phone = self.normalize_phone(str(value))
                if normalized_phone:
                    field['value'] = normalized_phone
            
        except Exception as e:
            print(f"Field validation error for {field_name}: {e}")
            field['confidence'] *= 0.5  # Significantly reduce confidence on validation errors
        
        return field
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date strings to YYYY-MM-DD format"""
        if not date_str:
            return None
        
        # Common date patterns
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
            r'(\d{1,2})\.(\d{1,2})\.(\d{4})', # MM.DD.YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if len(groups[0]) == 4:  # First group is year
                        year, month, day = groups
                    else:  # First group is month/day
                        month, day, year = groups
                    
                    try:
                        # Validate and format
                        month, day, year = int(month), int(day), int(year)
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except ValueError:
                        continue
        
        return None
    
    def normalize_amount(self, amount_str: str) -> Optional[float]:
        """Normalize amount strings to float values"""
        if not amount_str:
            return None
        
        # Remove common currency symbols and formatting
        cleaned = re.sub(r'[$€£¥,\s]', '', str(amount_str))
        
        # Handle negative amounts in parentheses
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def normalize_phone(self, phone_str: str) -> Optional[str]:
        """Normalize phone number strings"""
        if not phone_str:
            return None
        
        # Extract digits only
        digits = re.sub(r'\D', '', phone_str)
        
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        return phone_str  # Return original if can't normalize
