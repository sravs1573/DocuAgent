from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class FieldExtraction(BaseModel):
    """Base model for extracted field information"""
    name: str = Field(description="Field name")
    value: Optional[str] = Field(description="Extracted value")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    source: Dict[str, Any] = Field(description="Source information including page and bbox coordinates")

class LineItem(BaseModel):
    """Model for invoice line items"""
    description: Optional[str] = Field(description="Item description")
    quantity: Optional[float] = Field(description="Item quantity")
    unit_price: Optional[float] = Field(description="Unit price")
    total: Optional[float] = Field(description="Line total")

class Medication(BaseModel):
    """Model for prescription medications"""
    name: str = Field(description="Medication name")
    dosage: Optional[str] = Field(description="Dosage information")
    frequency: Optional[str] = Field(description="Frequency of administration")
    quantity: Optional[str] = Field(description="Quantity prescribed")

class MedicalProcedure(BaseModel):
    """Model for medical procedures"""
    code: Optional[str] = Field(description="Procedure code")
    description: str = Field(description="Procedure description")
    amount: Optional[float] = Field(description="Procedure cost")

class InvoiceSchema(BaseModel):
    """Schema for invoice document extraction"""
    invoice_number: Optional[str] = Field(description="Invoice number")
    invoice_date: Optional[str] = Field(description="Invoice date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(description="Due date in YYYY-MM-DD format")
    vendor_name: Optional[str] = Field(description="Vendor/company name")
    vendor_address: Optional[str] = Field(description="Vendor address")
    customer_name: Optional[str] = Field(description="Customer name")
    customer_address: Optional[str] = Field(description="Customer address")
    subtotal: Optional[float] = Field(description="Subtotal amount")
    tax_amount: Optional[float] = Field(description="Tax amount")
    total_amount: Optional[float] = Field(description="Total amount")
    currency: Optional[str] = Field(description="Currency code")
    line_items: List[LineItem] = Field(default=[], description="List of line items")

class MedicalBillSchema(BaseModel):
    """Schema for medical bill document extraction"""
    patient_name: Optional[str] = Field(description="Patient full name")
    patient_id: Optional[str] = Field(description="Patient ID number")
    patient_dob: Optional[str] = Field(description="Patient date of birth")
    provider_name: Optional[str] = Field(description="Healthcare provider name")
    provider_address: Optional[str] = Field(description="Provider address")
    service_date: Optional[str] = Field(description="Date of service")
    diagnosis: Optional[str] = Field(description="Primary diagnosis")
    procedures: List[MedicalProcedure] = Field(default=[], description="List of procedures")
    insurance_company: Optional[str] = Field(description="Insurance company name")
    policy_number: Optional[str] = Field(description="Insurance policy number")
    total_charges: Optional[float] = Field(description="Total charges")
    insurance_paid: Optional[float] = Field(description="Amount paid by insurance")
    patient_responsibility: Optional[float] = Field(description="Patient responsibility amount")

class PrescriptionSchema(BaseModel):
    """Schema for prescription document extraction"""
    patient_name: Optional[str] = Field(description="Patient full name")
    patient_address: Optional[str] = Field(description="Patient address")
    patient_dob: Optional[str] = Field(description="Patient date of birth")
    doctor_name: Optional[str] = Field(description="Prescribing doctor name")
    doctor_license: Optional[str] = Field(description="Doctor license number")
    pharmacy_name: Optional[str] = Field(description="Pharmacy name")
    pharmacy_address: Optional[str] = Field(description="Pharmacy address")
    pharmacy_phone: Optional[str] = Field(description="Pharmacy phone number")
    prescription_date: Optional[str] = Field(description="Prescription date")
    prescription_number: Optional[str] = Field(description="Prescription number")
    medications: List[Medication] = Field(default=[], description="List of medications")
    refills: Optional[int] = Field(description="Number of refills allowed")
    doctor_signature: Optional[str] = Field(description="Doctor signature status")

class ExtractionResult(BaseModel):
    """Complete extraction result model"""
    doc_type: str = Field(description="Document type classification")
    fields: List[FieldExtraction] = Field(description="Extracted fields with confidence scores")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall extraction confidence")
    qa: Dict[str, Any] = Field(description="Quality assurance and validation results")
    processing_metadata: Dict[str, Any] = Field(description="Processing metadata and parameters")

class ValidationRule(BaseModel):
    """Model for validation rules"""
    rule_name: str = Field(description="Name of the validation rule")
    rule_type: str = Field(description="Type of validation (regex, range, cross_field, etc.)")
    parameters: Dict[str, Any] = Field(description="Rule parameters")
    error_message: str = Field(description="Error message when rule fails")

class ConfidenceMetrics(BaseModel):
    """Model for confidence calculation metrics"""
    text_clarity: float = Field(ge=0.0, le=1.0, description="Text clarity score")
    context_strength: float = Field(ge=0.0, le=1.0, description="Context strength score")
    pattern_match: float = Field(ge=0.0, le=1.0, description="Pattern matching score")
    consistency_score: float = Field(ge=0.0, le=1.0, description="Cross-field consistency score")
    final_confidence: float = Field(ge=0.0, le=1.0, description="Final calculated confidence")
