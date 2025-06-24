from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, ValidationError
import re
import json
import logging

logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

class DataValidator:
    """Utility class for validating various types of data inputs"""
    
    @staticmethod
    def validate_project_description(description: str) -> ValidationResult:
        """Validate project description"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check minimum length
        if len(description.strip()) < 50:
            errors.append("Project description must be at least 50 characters long")
        
        # Check maximum length
        if len(description) > 2000:
            errors.append("Project description must be less than 2000 characters")
        
        # Check for humanitarian context indicators
        humanitarian_keywords = [
            "humanitarian", "crisis", "refugee", "disaster", "emergency",
            "aid", "relief", "assistance", "vulnerable", "community"
        ]
        
        if not any(keyword in description.lower() for keyword in humanitarian_keywords):
            warnings.append("Description should include humanitarian context indicators")
            suggestions.append("Consider adding humanitarian context (e.g., target population, crisis type)")
        
        # Check for AI/technology indicators
        tech_keywords = [
            "ai", "artificial intelligence", "machine learning", "data",
            "predict", "analyze", "automate", "algorithm", "model"
        ]
        
        if not any(keyword in description.lower() for keyword in tech_keywords):
            warnings.append("Description should mention AI/technology components")
            suggestions.append("Specify how AI/technology will be used in the project")
        
        # Check for specific problem statement
        problem_indicators = ["problem", "challenge", "issue", "need", "gap"]
        if not any(indicator in description.lower() for indicator in problem_indicators):
            suggestions.append("Include a clear problem statement")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_reflection_answers(answers: Dict[str, str]) -> ValidationResult:
        """Validate reflection phase answers"""
        errors = []
        warnings = []
        suggestions = []
        
        required_questions = [
            "problem_definition",
            "target_beneficiaries", 
            "potential_harm",
            "data_availability",
            "resource_constraints",
            "success_metrics",
            "stakeholder_involvement",
            "cultural_sensitivity"
        ]
        
        # Check all required questions are answered
        for question in required_questions:
            if question not in answers or not answers[question].strip():
                errors.append(f"Missing answer for {question.replace('_', ' ')}")
            elif len(answers[question].strip()) < 20:
                warnings.append(f"Answer for {question.replace('_', ' ')} seems too brief")
                suggestions.append(f"Provide more detail for {question.replace('_', ' ')}")
        
        # Check for thoughtful responses
        for question, answer in answers.items():
            if answer and len(answer.strip()) > 10:
                # Check for generic responses
                generic_phrases = [
                    "not sure", "don't know", "maybe", "probably", 
                    "i think", "perhaps", "unclear"
                ]
                
                if any(phrase in answer.lower() for phrase in generic_phrases):
                    warnings.append(f"Answer for {question} contains uncertainty indicators")
                    suggestions.append(f"Consider researching and providing more definitive information for {question}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_deployment_environment(env_data: Dict[str, Any]) -> ValidationResult:
        """Validate deployment environment configuration"""
        errors = []
        warnings = []
        suggestions = []
        
        required_fields = [
            "project_budget",
            "project_timeline", 
            "team_size",
            "computing_resources",
            "reliable_internet_connection",
            "local_technology_setup",
            "ai_ml_experience",
            "technical_skills",
            "learning_training_capacity",
            "stakeholder_buy_in",
            "change_management_readiness",
            "data_governance",
            "regulatory_requirements",
            "external_partnerships",
            "long_term_sustainability_plan"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in env_data:
                errors.append(f"Missing required field: {field}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_file_upload(
        filename: str,
        file_size: int,
        content_type: str,
        max_size: int = 50 * 1024 * 1024  # 50MB
    ) -> ValidationResult:
        """Validate file upload parameters"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check file size
        if file_size > max_size:
            errors.append(f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)")
        
        # Check filename
        if not filename or not filename.strip():
            errors.append("Filename cannot be empty")
        
        # Check for valid filename characters
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            errors.append("Filename contains invalid characters")
            suggestions.append("Use only letters, numbers, dots, hyphens, and underscores")
        
        # Check file extension
        allowed_extensions = ['.pdf', '.txt', '.docx', '.csv', '.xlsx']
        file_extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_extension not in allowed_extensions:
            errors.append(f"File extension {file_extension} not allowed")
            suggestions.append(f"Use one of: {', '.join(allowed_extensions)}")
        
        # Check content type
        allowed_content_types = [
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ]
        
        if content_type not in allowed_content_types:
            warnings.append(f"Unexpected content type: {content_type}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_json_structure(
        data: Union[str, Dict[str, Any]],
        required_fields: List[str],
        optional_fields: List[str] = None
    ) -> ValidationResult:
        """Validate JSON data structure"""
        errors = []
        warnings = []
        suggestions = []
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON format: {str(e)}")
                return ValidationResult(is_valid=False, errors=errors)
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
                errors.append(f"Field {field} cannot be empty")
        
        # Check for unexpected fields
        optional_fields = optional_fields or []
        all_expected_fields = set(required_fields + optional_fields)
        unexpected_fields = set(data.keys()) - all_expected_fields
        
        if unexpected_fields:
            warnings.append(f"Unexpected fields: {', '.join(unexpected_fields)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        """Validate email address format"""
        errors = []
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            errors.append("Invalid email format")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Validate URL format"""
        errors = []
        suggestions = []
        
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        if not re.match(url_pattern, url):
            errors.append("Invalid URL format")
            suggestions.append("URL should start with http:// or https://")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            suggestions=suggestions
        )