import io
import base64
from typing import Union, Dict, Any
import streamlit as st
from PIL import Image

class FileHandler:
    """Handles file upload and processing for different file types"""
    
    def __init__(self):
        self.supported_extensions = {
            'pdf': ['pdf'],
            'image': ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        }
    
    def process_uploaded_file(self, uploaded_file) -> bytes:
        """Process uploaded file and return content as bytes"""
        
        if uploaded_file is None:
            raise ValueError("No file uploaded")
        
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Validate file type
        if not self.is_supported_file_type(file_extension):
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Read file content
        file_content = uploaded_file.read()
        
        if len(file_content) == 0:
            raise ValueError("Uploaded file is empty")
        
        # Validate file size (limit to 50MB)
        if len(file_content) > 50 * 1024 * 1024:
            raise ValueError("File size exceeds 50MB limit")
        
        # Additional validation for images
        if file_extension in self.supported_extensions['image']:
            file_content = self.validate_and_process_image(file_content)
        
        return file_content
    
    def is_supported_file_type(self, extension: str) -> bool:
        """Check if file extension is supported"""
        extension = extension.lower()
        for file_type, extensions in self.supported_extensions.items():
            if extension in extensions:
                return True
        return False
    
    def get_file_type(self, filename: str) -> str:
        """Determine file type from filename"""
        extension = filename.split('.')[-1].lower()
        
        if extension in self.supported_extensions['pdf']:
            return 'pdf'
        elif extension in self.supported_extensions['image']:
            return 'image'
        else:
            return 'unknown'
    
    def validate_and_process_image(self, image_content: bytes) -> bytes:
        """Validate and process image content"""
        
        try:
            # Open image to validate it's a valid image file
            image = Image.open(io.BytesIO(image_content))
            
            # Convert to RGB if necessary (for RGBA, P mode images)
            if image.mode in ('RGBA', 'P'):
                # Create white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if image is too large (max 4000x4000 pixels)
            max_dimension = 4000
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to bytes
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=95)
            return output_buffer.getvalue()
            
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    def encode_image_to_base64(self, image_content: bytes) -> str:
        """Encode image content to base64 string"""
        return base64.b64encode(image_content).decode('utf-8')
    
    def create_download_link(self, content: str, filename: str, mime_type: str = "application/json") -> str:
        """Create a download link for content"""
        b64_content = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:{mime_type};base64,{b64_content}" download="{filename}">Download {filename}</a>'
        return href
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    def extract_file_metadata(self, uploaded_file) -> Dict[str, Any]:
        """Extract metadata from uploaded file"""
        
        if uploaded_file is None:
            return {}
        
        metadata = {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'file_extension': uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else '',
            'formatted_size': self.format_file_size(uploaded_file.size)
        }
        
        # Add file type classification
        metadata['file_type'] = self.get_file_type(uploaded_file.name)
        
        return metadata
    
    def validate_file_for_processing(self, uploaded_file) -> Dict[str, Any]:
        """Validate file and return validation result"""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            if uploaded_file is None:
                validation_result['is_valid'] = False
                validation_result['errors'].append("No file uploaded")
                return validation_result
            
            # Extract metadata
            metadata = self.extract_file_metadata(uploaded_file)
            validation_result['metadata'] = metadata
            
            # Check file extension
            if not self.is_supported_file_type(metadata['file_extension']):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Unsupported file type: {metadata['file_extension']}")
            
            # Check file size
            if metadata['size'] > 50 * 1024 * 1024:  # 50MB
                validation_result['is_valid'] = False
                validation_result['errors'].append("File size exceeds 50MB limit")
            elif metadata['size'] > 10 * 1024 * 1024:  # 10MB
                validation_result['warnings'].append("Large file size may result in slower processing")
            
            # Check if file is empty
            if metadata['size'] == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append("File appears to be empty")
            
            # Additional validation for images
            if metadata['file_type'] == 'image':
                try:
                    file_content = uploaded_file.read()
                    uploaded_file.seek(0)  # Reset file pointer
                    
                    image = Image.open(io.BytesIO(file_content))
                    
                    # Check image dimensions
                    if max(image.size) > 8000:
                        validation_result['warnings'].append("Very high resolution image may require longer processing time")
                    elif max(image.size) < 500:
                        validation_result['warnings'].append("Low resolution image may result in poor OCR accuracy")
                    
                except Exception as e:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Invalid image file: {str(e)}")
        
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"File validation error: {str(e)}")
        
        return validation_result
