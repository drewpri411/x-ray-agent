"""
Helper Utilities for Radiology Assistant
Common utility functions used throughout the application.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def load_environment_variables(env_file: str = ".env"):
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to the .env file
    """
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logging.info(f"Loaded environment variables from {env_file}")
    else:
        logging.warning(f"Environment file {env_file} not found")

def validate_file_path(file_path: str, required_extensions: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate a file path and check file properties.
    
    Args:
        file_path: Path to the file to validate
        required_extensions: List of required file extensions
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': False,
        'errors': [],
        'warnings': [],
        'file_info': {}
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result['errors'].append(f"File not found: {file_path}")
            return validation_result
        
        # Get file info
        file_stat = os.stat(file_path)
        validation_result['file_info'] = {
            'size': file_stat.st_size,
            'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'extension': os.path.splitext(file_path)[1].lower()
        }
        
        # Check file size
        if file_stat.st_size == 0:
            validation_result['errors'].append("File is empty")
        elif file_stat.st_size < 1000:  # Less than 1KB
            validation_result['warnings'].append("File is very small")
        elif file_stat.st_size > 100 * 1024 * 1024:  # More than 100MB
            validation_result['warnings'].append("File is very large")
        
        # Check file extension
        if required_extensions:
            file_ext = validation_result['file_info']['extension']
            if file_ext not in required_extensions:
                validation_result['errors'].append(
                    f"Invalid file extension. Expected: {required_extensions}, got: {file_ext}"
                )
        
        # Mark as valid if no errors
        if not validation_result['errors']:
            validation_result['is_valid'] = True
        
        return validation_result
        
    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")
        return validation_result

def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the output file
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
        
        logging.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Data loaded from {file_path}")
        return data
        
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        return None

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp for logging and display.
    
    Args:
        timestamp: Timestamp to format (uses current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def create_output_directory(base_path: str, subdirectory: str = "") -> str:
    """
    Create an output directory with timestamp.
    
    Args:
        base_path: Base directory path
        subdirectory: Optional subdirectory name
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if subdirectory:
        output_dir = os.path.join(base_path, subdirectory, timestamp)
    else:
        output_dir = os.path.join(base_path, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    return output_dir

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe file system operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        'torch': False,
        'torchxrayvision': False,
        'langchain': False,
        'google.generativeai': False
    }
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import torchxrayvision
        dependencies['torchxrayvision'] = True
    except ImportError:
        pass
    
    try:
        import langchain
        dependencies['langchain'] = True
    except ImportError:
        pass
    
    try:
        import google.generativeai
        dependencies['google.generativeai'] = True
    except ImportError:
        pass
    

    
    return dependencies

def print_dependency_status():
    """Print the status of all dependencies."""
    dependencies = check_dependencies()
    
    print("\nDependency Status:")
    print("-" * 40)
    
    for dep, available in dependencies.items():
        status = "✅ Available" if available else "❌ Missing"
        print(f"{dep:20} {status}")
    
    missing_deps = [dep for dep, available in dependencies.items() if not available]
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies using: pip install -r requirements.txt")
    else:
        print("\nAll dependencies are available!") 