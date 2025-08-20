import re
import math
from typing import Dict, List, Any, Optional
import numpy as np
from collections import Counter

class ConfidenceScorer:
    """Advanced confidence scoring system for extracted document data"""
    
    def __init__(self):
        # Confidence scoring weights
        self.scoring_weights = {
            'text_clarity': 0.3,
            'context_strength': 0.25,
            'pattern_match': 0.25,
            'consistency': 0.2
        }
        
        # Pattern confidence multipliers
        self.pattern_confidence = {
            'date': 0.9,
            'amount': 0.85,
            'phone': 0.8,
            'email': 0.9,
            'zip_code': 0.85,
            'id_number': 0.75
        }
    
    def calculate_field_confidence(
        self, 
        extraction_result: Dict[str, Any], 
        source_text: str, 
        doc_type: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive confidence scores for all extracted fields"""
        
        fields = extraction_result.get('fields', [])
        if not fields:
            return {
                'fields': [],
                'overall_confidence': 0.0,
                'confidence_metadata': {
                    'method': 'advanced_scoring',
                    'source_text_length': len(source_text),
                    'field_count': 0
                }
            }
        
        # Calculate confidence for each field
        scored_fields = []
        confidence_scores = []
        
        for field in fields:
            field_confidence = self.calculate_single_field_confidence(
                field, source_text, doc_type, fields
            )
            scored_fields.append(field_confidence)
            confidence_scores.append(field_confidence['confidence'])
        
        # Calculate overall confidence
        overall_confidence = self.calculate_overall_confidence(confidence_scores, fields, doc_type)
        
        return {
            'fields': scored_fields,
            'overall_confidence': overall_confidence,
            'confidence_metadata': {
                'method': 'advanced_scoring',
                'source_text_length': len(source_text),
                'field_count': len(fields),
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'confidence_std': np.std(confidence_scores) if len(confidence_scores) > 1 else 0,
                'scoring_weights': self.scoring_weights
            }
        }
    
    def calculate_single_field_confidence(
        self, 
        field: Dict[str, Any], 
        source_text: str, 
        doc_type: str, 
        all_fields: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate confidence for a single field using multiple factors"""
        
        field_name = field.get('name', '')
        field_value = field.get('value', '')
        base_confidence = field.get('confidence', 0.5)
        
        if not field_value:
            field['confidence'] = 0.0
            return field
        
        # Component confidence scores
        text_clarity = self.score_text_clarity(field_value, source_text)
        context_strength = self.score_context_strength(field_name, field_value, source_text, doc_type)
        pattern_match = self.score_pattern_match(field_name, field_value)
        consistency_score = self.score_cross_field_consistency(field, all_fields)
        
        # Weighted final confidence
        final_confidence = (
            text_clarity * self.scoring_weights['text_clarity'] +
            context_strength * self.scoring_weights['context_strength'] +
            pattern_match * self.scoring_weights['pattern_match'] +
            consistency_score * self.scoring_weights['consistency']
        )
        
        # Blend with base confidence (if provided by LLM)
        if base_confidence > 0:
            final_confidence = (final_confidence * 0.7) + (base_confidence * 0.3)
        
        # Ensure confidence is within bounds
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Store detailed confidence breakdown
        field['confidence'] = final_confidence
        field['confidence_breakdown'] = {
            'text_clarity': text_clarity,
            'context_strength': context_strength,
            'pattern_match': pattern_match,
            'consistency': consistency_score,
            'base_confidence': base_confidence,
            'final_confidence': final_confidence
        }
        
        return field
    
    def score_text_clarity(self, field_value: str, source_text: str) -> float:
        """Score based on text clarity and OCR quality indicators"""
        
        if not field_value:
            return 0.0
        
        clarity_score = 1.0
        
        # Check for common OCR errors
        ocr_error_patterns = [
            r'[|]{2,}',  # Multiple pipes (common OCR error)
            r'[0O]{3,}', # Multiple O's or 0's
            r'[Il1]{3,}', # Multiple similar characters
            r'[@#$%^&*]{2,}', # Multiple special characters
            r'\s{3,}', # Excessive whitespace
        ]
        
        for pattern in ocr_error_patterns:
            if re.search(pattern, field_value):
                clarity_score *= 0.7
        
        # Length-based confidence (very short or very long values are suspicious)
        value_length = len(field_value.strip())
        if value_length < 2:
            clarity_score *= 0.5
        elif value_length > 200:
            clarity_score *= 0.8
        
        # Check for mixed character types that don't make sense
        if re.search(r'\d+[A-Za-z]+\d+', field_value) and 'address' not in field_value.lower():
            clarity_score *= 0.8
        
        # Presence in source text (exact or fuzzy match)
        if field_value.lower() in source_text.lower():
            clarity_score *= 1.2
        elif self.fuzzy_text_match(field_value, source_text):
            clarity_score *= 1.1
        else:
            clarity_score *= 0.7
        
        return min(1.0, clarity_score)
    
    def score_context_strength(self, field_name: str, field_value: str, source_text: str, doc_type: str) -> float:
        """Score based on contextual clues and field placement"""
        
        if not field_value:
            return 0.0
        
        context_score = 0.6  # Base score
        
        # Document type specific context keywords
        context_keywords = {
            'invoice': {
                'invoice_number': ['invoice', 'inv', '#', 'number'],
                'total_amount': ['total', 'amount', 'due', '$', 'balance'],
                'vendor_name': ['from:', 'vendor', 'company', 'bill to'],
                'invoice_date': ['date', 'issued', 'invoice date'],
                'due_date': ['due', 'payment due', 'due date']
            },
            'medical_bill': {
                'patient_name': ['patient', 'name', 'member'],
                'provider_name': ['provider', 'hospital', 'clinic', 'doctor'],
                'total_charges': ['total', 'charges', 'amount', 'balance'],
                'service_date': ['service', 'date', 'visit date'],
                'insurance_company': ['insurance', 'plan', 'coverage']
            },
            'prescription': {
                'patient_name': ['patient', 'name'],
                'doctor_name': ['doctor', 'physician', 'prescriber', 'md'],
                'pharmacy_name': ['pharmacy', 'rx', 'dispensed by'],
                'medications': ['medication', 'drug', 'rx', 'prescribed'],
                'prescription_date': ['date', 'prescribed', 'rx date']
            }
        }
        
        # Check for relevant context keywords
        field_keywords = context_keywords.get(doc_type, {}).get(field_name, [])
        
        for keyword in field_keywords:
            # Look for keyword near the field value in source text
            if self.find_keyword_near_value(keyword, field_value, source_text):
                context_score += 0.1
        
        # Field-specific context validation
        context_score += self.validate_field_context(field_name, field_value)
        
        return min(1.0, context_score)
    
    def score_pattern_match(self, field_name: str, field_value: str) -> float:
        """Score based on expected patterns for field types"""
        
        if not field_value:
            return 0.0
        
        field_name_lower = field_name.lower()
        pattern_score = 0.5  # Base score
        
        # Date patterns
        if 'date' in field_name_lower:
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
                r'^\d{1,2}-\d{1,2}-\d{4}$',  # MM-DD-YYYY
            ]
            for pattern in date_patterns:
                if re.match(pattern, field_value):
                    pattern_score = self.pattern_confidence['date']
                    break
        
        # Amount patterns
        elif any(keyword in field_name_lower for keyword in ['amount', 'total', 'subtotal', 'tax', 'charge', 'paid']):
            amount_patterns = [
                r'^\d+\.\d{2}$',  # XX.XX
                r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$',  # Currency format
                r'^\d+$',  # Integer
            ]
            for pattern in amount_patterns:
                if re.match(pattern, str(field_value)):
                    pattern_score = self.pattern_confidence['amount']
                    break
        
        # Phone patterns
        elif 'phone' in field_name_lower:
            phone_patterns = [
                r'^\(\d{3}\) \d{3}-\d{4}$',  # (XXX) XXX-XXXX
                r'^\d{10}$',  # XXXXXXXXXX
                r'^\d{3}-\d{3}-\d{4}$',  # XXX-XXX-XXXX
            ]
            for pattern in phone_patterns:
                if re.match(pattern, field_value):
                    pattern_score = self.pattern_confidence['phone']
                    break
        
        # Email patterns
        elif 'email' in field_name_lower:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, field_value):
                pattern_score = self.pattern_confidence['email']
        
        # ID/Number patterns
        elif any(keyword in field_name_lower for keyword in ['id', 'number', 'policy', 'member']):
            # Should contain some numbers or be purely alphanumeric
            if re.search(r'\d', field_value) and field_value.replace(' ', '').isalnum():
                pattern_score = self.pattern_confidence['id_number']
        
        # Name patterns (should not contain numbers or excessive special chars)
        elif 'name' in field_name_lower and 'file' not in field_name_lower:
            if not re.search(r'\d', field_value) and len(re.findall(r'[^a-zA-Z\s\-\.]', field_value)) < 3:
                pattern_score = 0.8
        
        return min(1.0, pattern_score)
    
    def score_cross_field_consistency(self, field: Dict[str, Any], all_fields: List[Dict]) -> float:
        """Score based on consistency with other extracted fields"""
        
        field_name = field.get('name', '')
        field_value = field.get('value', '')
        
        if not field_value:
            return 0.0
        
        consistency_score = 0.7  # Base score
        
        # Create a dict of other fields for easy lookup
        field_dict = {f['name']: f['value'] for f in all_fields if f['name'] != field_name}
        
        # Date consistency checks
        if 'date' in field_name.lower():
            invoice_date = field_dict.get('invoice_date')
            due_date = field_dict.get('due_date')
            
            if field_name == 'due_date' and invoice_date:
                try:
                    if field_value >= invoice_date:
                        consistency_score += 0.1
                    else:
                        consistency_score -= 0.2
                except:
                    pass
        
        # Amount consistency checks
        if 'total' in field_name.lower():
            subtotal = self.safe_float_convert(field_dict.get('subtotal'))
            tax_amount = self.safe_float_convert(field_dict.get('tax_amount'))
            total_amount = self.safe_float_convert(field_value)
            
            if subtotal is not None and tax_amount is not None and total_amount is not None:
                expected_total = subtotal + tax_amount
                if abs(expected_total - total_amount) < 0.01:
                    consistency_score += 0.2
                else:
                    consistency_score -= 0.1
        
        # Name consistency (similar names across fields might indicate OCR errors)
        if 'name' in field_name.lower():
            other_names = [v for k, v in field_dict.items() if 'name' in k.lower() and v]
            if other_names:
                # Check for suspiciously similar names
                for other_name in other_names:
                    if self.calculate_similarity(field_value, other_name) > 0.8 and field_value != other_name:
                        consistency_score -= 0.1
        
        return max(0.0, min(1.0, consistency_score))
    
    def calculate_overall_confidence(self, confidence_scores: List[float], fields: List[Dict], doc_type: str) -> float:
        """Calculate overall document processing confidence"""
        
        if not confidence_scores:
            return 0.0
        
        # Base calculation using weighted average
        mean_confidence = np.mean(confidence_scores)
        
        # Penalty for high variance (inconsistent confidence across fields)
        confidence_std = np.std(confidence_scores) if len(confidence_scores) > 1 else 0
        variance_penalty = min(0.2, float(confidence_std) * 0.5)
        
        # Bonus for critical fields being present and high confidence
        critical_fields = self.get_critical_fields(doc_type)
        critical_field_names = [f['name'] for f in fields]
        critical_bonus = 0
        
        for critical_field in critical_fields:
            if critical_field in critical_field_names:
                field_confidence = next((f['confidence'] for f in fields if f['name'] == critical_field), 0)
                if field_confidence > 0.8:
                    critical_bonus += 0.05
        
        # Penalty for missing critical fields
        missing_critical = len(critical_fields) - len([f for f in critical_fields if f in critical_field_names])
        missing_penalty = missing_critical * 0.1
        
        # Field count factor (more successfully extracted fields = higher confidence)
        field_count_factor = min(0.1, len(fields) * 0.01)
        
        overall_confidence = mean_confidence - variance_penalty + critical_bonus - missing_penalty + field_count_factor
        
        return float(max(0.0, min(1.0, float(overall_confidence))))
    
    def get_critical_fields(self, doc_type: str) -> List[str]:
        """Get list of critical fields for each document type"""
        critical_fields = {
            'invoice': ['invoice_number', 'total_amount', 'vendor_name', 'invoice_date'],
            'medical_bill': ['patient_name', 'provider_name', 'total_charges', 'service_date'],
            'prescription': ['patient_name', 'doctor_name', 'medications', 'prescription_date']
        }
        return critical_fields.get(doc_type, critical_fields['invoice'])
    
    def fuzzy_text_match(self, field_value: str, source_text: str, threshold: float = 0.8) -> bool:
        """Check for fuzzy match of field value in source text"""
        if not field_value or not source_text:
            return False
        
        # Simple fuzzy matching using character overlap
        field_clean = re.sub(r'[^\w]', '', field_value.lower())
        source_clean = source_text.lower()
        
        # Check if most characters of field value appear in source
        matching_chars = sum(1 for char in field_clean if char in source_clean)
        return matching_chars / len(field_clean) >= threshold if field_clean else False
    
    def find_keyword_near_value(self, keyword: str, field_value: str, source_text: str, window: int = 50) -> bool:
        """Check if keyword appears near field value in source text"""
        if not all([keyword, field_value, source_text]):
            return False
        
        source_lower = source_text.lower()
        keyword_lower = keyword.lower()
        value_lower = field_value.lower()
        
        # Find all occurrences of the field value
        value_positions = []
        start = 0
        while True:
            pos = source_lower.find(value_lower, start)
            if pos == -1:
                break
            value_positions.append(pos)
            start = pos + 1
        
        # Check if keyword appears within window of any value occurrence
        for pos in value_positions:
            window_start = max(0, pos - window)
            window_end = min(len(source_lower), pos + len(value_lower) + window)
            window_text = source_lower[window_start:window_end]
            
            if keyword_lower in window_text:
                return True
        
        return False
    
    def validate_field_context(self, field_name: str, field_value: str) -> float:
        """Validate field value makes sense for field type"""
        bonus = 0.0
        field_name_lower = field_name.lower()
        
        # Name fields should look like names
        if 'name' in field_name_lower and 'file' not in field_name_lower:
            words = field_value.split()
            if 1 <= len(words) <= 5 and all(word.isalpha() or '-' in word for word in words):
                bonus += 0.1
        
        # Amount fields should be reasonable
        elif any(term in field_name_lower for term in ['amount', 'total', 'charge', 'paid']):
            try:
                amount = float(re.sub(r'[^\d.]', '', str(field_value)))
                if 0.01 <= amount <= 1000000:  # Reasonable range
                    bonus += 0.1
            except:
                pass
        
        # ID fields should have reasonable format
        elif any(term in field_name_lower for term in ['id', 'number']):
            if 3 <= len(str(field_value)) <= 50 and any(c.isalnum() for c in str(field_value)):
                bonus += 0.1
        
        return bonus
    
    def safe_float_convert(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            # Remove common currency symbols and formatting
            clean_value = re.sub(r'[$,€£¥]', '', str(value))
            return float(clean_value)
        except (ValueError, TypeError):
            return None
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        str1_clean = str1.lower().strip()
        str2_clean = str2.lower().strip()
        
        if str1_clean == str2_clean:
            return 1.0
        
        # Simple character-based similarity
        common_chars = set(str1_clean) & set(str2_clean)
        total_chars = set(str1_clean) | set(str2_clean)
        
        return len(common_chars) / len(total_chars) if total_chars else 0.0
