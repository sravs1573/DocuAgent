import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

class DocumentValidator:
    """Validation engine for extracted document data"""
    
    def __init__(self):
        self.validation_rules = {
            'invoice': self.get_invoice_validation_rules(),
            'medical_bill': self.get_medical_bill_validation_rules(),
            'prescription': self.get_prescription_validation_rules()
        }
    
    def get_invoice_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for invoice documents"""
        return {
            'date_format': {
                'fields': ['invoice_date', 'due_date'],
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'description': 'Date must be in YYYY-MM-DD format'
            },
            'amount_format': {
                'fields': ['subtotal', 'tax_amount', 'total_amount'],
                'validation': 'numeric_positive',
                'description': 'Amounts must be positive numbers'
            },
            'totals_match': {
                'validation': 'cross_field',
                'rule': 'subtotal + tax_amount = total_amount',
                'tolerance': 0.01,
                'description': 'Subtotal plus tax should equal total amount'
            },
            'due_date_logic': {
                'validation': 'cross_field',
                'rule': 'due_date >= invoice_date',
                'description': 'Due date should be after or equal to invoice date'
            },
            'required_fields': {
                'fields': ['invoice_number', 'total_amount', 'vendor_name'],
                'validation': 'not_empty',
                'description': 'Critical fields must not be empty'
            }
        }
    
    def get_medical_bill_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for medical bill documents"""
        return {
            'date_format': {
                'fields': ['service_date', 'patient_dob'],
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'description': 'Dates must be in YYYY-MM-DD format'
            },
            'amount_format': {
                'fields': ['total_charges', 'insurance_paid', 'patient_responsibility'],
                'validation': 'numeric_positive',
                'description': 'Amounts must be positive numbers'
            },
            'charges_breakdown': {
                'validation': 'cross_field',
                'rule': 'insurance_paid + patient_responsibility = total_charges',
                'tolerance': 0.01,
                'description': 'Insurance paid plus patient responsibility should equal total charges'
            },
            'patient_age_logic': {
                'validation': 'date_logic',
                'rule': 'patient_dob should result in reasonable age (0-120 years)',
                'description': 'Patient date of birth should be realistic'
            },
            'required_fields': {
                'fields': ['patient_name', 'provider_name', 'total_charges'],
                'validation': 'not_empty',
                'description': 'Critical fields must not be empty'
            }
        }
    
    def get_prescription_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for prescription documents"""
        return {
            'date_format': {
                'fields': ['prescription_date', 'patient_dob'],
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'description': 'Dates must be in YYYY-MM-DD format'
            },
            'phone_format': {
                'fields': ['pharmacy_phone'],
                'pattern': r'^\(\d{3}\) \d{3}-\d{4}$|^\d{10}$',
                'description': 'Phone number must be in valid format'
            },
            'refills_range': {
                'fields': ['refills'],
                'validation': 'numeric_range',
                'min_value': 0,
                'max_value': 12,
                'description': 'Refills should be between 0 and 12'
            },
            'prescription_age': {
                'validation': 'date_logic',
                'rule': 'prescription_date should be within last 2 years',
                'description': 'Prescription date should be recent'
            },
            'required_fields': {
                'fields': ['patient_name', 'doctor_name', 'medications'],
                'validation': 'not_empty',
                'description': 'Critical fields must not be empty'
            }
        }
    
    def validate_extraction(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Main validation function that applies all relevant rules"""
        
        doc_type = extraction_result.get('doc_type', 'invoice')
        fields = extraction_result.get('fields', [])
        
        # Convert fields list to dict for easier validation
        field_dict = {field['name']: field['value'] for field in fields}
        
        validation_results = {
            'passed_rules': [],
            'failed_rules': [],
            'warnings': [],
            'notes': ''
        }
        
        # Get validation rules for this document type
        rules = self.validation_rules.get(doc_type, self.validation_rules['invoice'])
        
        # Apply each validation rule
        for rule_name, rule_config in rules.items():
            try:
                result = self.apply_validation_rule(rule_name, rule_config, field_dict, fields)
                if result['passed']:
                    validation_results['passed_rules'].append(rule_name)
                else:
                    validation_results['failed_rules'].append(rule_name)
                    if result.get('warning'):
                        validation_results['warnings'].append(result['message'])
            except Exception as e:
                validation_results['failed_rules'].append(f"{rule_name}_error")
                validation_results['warnings'].append(f"Validation error in {rule_name}: {str(e)}")
        
        # Generate summary notes
        validation_results['notes'] = self.generate_validation_notes(validation_results, fields)
        
        return validation_results
    
    def apply_validation_rule(
        self, 
        rule_name: str, 
        rule_config: Dict[str, Any], 
        field_dict: Dict[str, str], 
        fields: List[Dict]
    ) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        
        validation_type = rule_config.get('validation', 'pattern')
        
        if validation_type == 'pattern' or 'pattern' in rule_config:
            return self.validate_pattern(rule_config, field_dict)
        
        elif validation_type == 'numeric_positive':
            return self.validate_numeric_positive(rule_config, field_dict)
        
        elif validation_type == 'numeric_range':
            return self.validate_numeric_range(rule_config, field_dict)
        
        elif validation_type == 'cross_field':
            return self.validate_cross_field(rule_config, field_dict)
        
        elif validation_type == 'date_logic':
            return self.validate_date_logic(rule_config, field_dict)
        
        elif validation_type == 'not_empty':
            return self.validate_not_empty(rule_config, field_dict)
        
        else:
            return {'passed': False, 'message': f'Unknown validation type: {validation_type}'}
    
    def validate_pattern(self, rule_config: Dict, field_dict: Dict) -> Dict[str, Any]:
        """Validate fields against regex patterns"""
        pattern = rule_config.get('pattern', '')
        fields_to_check = rule_config.get('fields', [])
        
        for field_name in fields_to_check:
            value = field_dict.get(field_name)
            if value and not re.match(pattern, str(value)):
                return {
                    'passed': False, 
                    'message': f'{field_name}: {rule_config.get("description", "Pattern validation failed")}'
                }
        
        return {'passed': True, 'message': 'Pattern validation passed'}
    
    def validate_numeric_positive(self, rule_config: Dict, field_dict: Dict) -> Dict[str, Any]:
        """Validate that numeric fields are positive"""
        fields_to_check = rule_config.get('fields', [])
        
        for field_name in fields_to_check:
            value = field_dict.get(field_name)
            if value is not None:
                try:
                    num_value = float(value)
                    if num_value < 0:
                        return {
                            'passed': False,
                            'message': f'{field_name} must be positive, got: {value}'
                        }
                except (ValueError, TypeError):
                    return {
                        'passed': False,
                        'message': f'{field_name} is not a valid number: {value}'
                    }
        
        return {'passed': True, 'message': 'Numeric validation passed'}
    
    def validate_numeric_range(self, rule_config: Dict, field_dict: Dict) -> Dict[str, Any]:
        """Validate that numeric fields are within specified range"""
        fields_to_check = rule_config.get('fields', [])
        min_value = rule_config.get('min_value', float('-inf'))
        max_value = rule_config.get('max_value', float('inf'))
        
        for field_name in fields_to_check:
            value = field_dict.get(field_name)
            if value is not None:
                try:
                    num_value = float(value)
                    if not (min_value <= num_value <= max_value):
                        return {
                            'passed': False,
                            'message': f'{field_name} must be between {min_value} and {max_value}, got: {value}'
                        }
                except (ValueError, TypeError):
                    return {
                        'passed': False,
                        'message': f'{field_name} is not a valid number: {value}'
                    }
        
        return {'passed': True, 'message': 'Range validation passed'}
    
    def validate_cross_field(self, rule_config: Dict, field_dict: Dict) -> Dict[str, Any]:
        """Validate cross-field relationships"""
        rule = rule_config.get('rule', '')
        tolerance = rule_config.get('tolerance', 0.01)
        
        try:
            if 'subtotal + tax_amount = total_amount' in rule:
                subtotal = float(field_dict.get('subtotal', 0) or 0)
                tax_amount = float(field_dict.get('tax_amount', 0) or 0)
                total_amount = float(field_dict.get('total_amount', 0) or 0)
                
                calculated_total = subtotal + tax_amount
                if abs(calculated_total - total_amount) > tolerance:
                    return {
                        'passed': False,
                        'message': f'Total mismatch: {subtotal} + {tax_amount} = {calculated_total}, but total is {total_amount}'
                    }
            
            elif 'insurance_paid + patient_responsibility = total_charges' in rule:
                insurance_paid = float(field_dict.get('insurance_paid', 0) or 0)
                patient_responsibility = float(field_dict.get('patient_responsibility', 0) or 0)
                total_charges = float(field_dict.get('total_charges', 0) or 0)
                
                calculated_total = insurance_paid + patient_responsibility
                if abs(calculated_total - total_charges) > tolerance:
                    return {
                        'passed': False,
                        'message': f'Charges mismatch: {insurance_paid} + {patient_responsibility} = {calculated_total}, but total charges is {total_charges}'
                    }
            
            elif 'due_date >= invoice_date' in rule:
                invoice_date = field_dict.get('invoice_date')
                due_date = field_dict.get('due_date')
                
                if invoice_date and due_date:
                    if due_date < invoice_date:
                        return {
                            'passed': False,
                            'message': f'Due date {due_date} is before invoice date {invoice_date}'
                        }
        
        except (ValueError, TypeError) as e:
            return {
                'passed': False,
                'message': f'Cross-field validation error: {str(e)}'
            }
        
        return {'passed': True, 'message': 'Cross-field validation passed'}
    
    def validate_date_logic(self, rule_config: Dict, field_dict: Dict) -> Dict[str, Any]:
        """Validate date logic and reasonableness"""
        rule = rule_config.get('rule', '')
        
        try:
            if 'reasonable age' in rule:
                patient_dob = field_dict.get('patient_dob')
                if patient_dob:
                    dob_date = datetime.strptime(patient_dob, '%Y-%m-%d')
                    age = (datetime.now() - dob_date).days / 365.25
                    if not (0 <= age <= 120):
                        return {
                            'passed': False,
                            'message': f'Unrealistic patient age: {age:.1f} years'
                        }
            
            elif 'within last 2 years' in rule:
                prescription_date = field_dict.get('prescription_date')
                if prescription_date:
                    rx_date = datetime.strptime(prescription_date, '%Y-%m-%d')
                    days_old = (datetime.now() - rx_date).days
                    if days_old > 730:  # 2 years
                        return {
                            'passed': False,
                            'message': f'Prescription is {days_old} days old (over 2 years)',
                            'warning': True
                        }
        
        except (ValueError, TypeError) as e:
            return {
                'passed': False,
                'message': f'Date logic validation error: {str(e)}'
            }
        
        return {'passed': True, 'message': 'Date logic validation passed'}
    
    def validate_not_empty(self, rule_config: Dict, field_dict: Dict) -> Dict[str, Any]:
        """Validate that required fields are not empty"""
        fields_to_check = rule_config.get('fields', [])
        
        for field_name in fields_to_check:
            value = field_dict.get(field_name)
            if not value or (isinstance(value, str) and not value.strip()):
                return {
                    'passed': False,
                    'message': f'Required field {field_name} is empty or missing'
                }
        
        return {'passed': True, 'message': 'Required fields validation passed'}
    
    def generate_validation_notes(self, validation_results: Dict, fields: List[Dict]) -> str:
        """Generate human-readable validation summary"""
        passed_count = len(validation_results['passed_rules'])
        failed_count = len(validation_results['failed_rules'])
        
        # Count low confidence fields
        low_confidence_fields = [f for f in fields if f.get('confidence', 0) < 0.6]
        
        notes_parts = []
        
        if failed_count > 0:
            notes_parts.append(f"{failed_count} validation rules failed")
        
        if len(low_confidence_fields) > 0:
            notes_parts.append(f"{len(low_confidence_fields)} low-confidence fields")
        
        if len(validation_results['warnings']) > 0:
            notes_parts.append(f"{len(validation_results['warnings'])} warnings")
        
        if not notes_parts:
            return "All validations passed successfully"
        
        return "; ".join(notes_parts)
