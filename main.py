"""
Main Application Entry Point for Radiology Assistant
Provides command-line interface and basic functionality.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any
from workflow import RadiologyWorkflowRunner
from utils.helpers import setup_logging, load_environment_variables
import json

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    load_environment_variables()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI Radiology Assistant - Chest X-ray Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image data/sample_xray.jpg --symptoms "Patient has cough and fever for 3 days"
  python main.py --image data/sample_xray.jpg --output results.json
  python main.py --image data/sample_xray.jpg --verbose
        """
    )
    
    parser.add_argument(
        "--image", 
        required=True,
        help="Path to the chest X-ray image file"
    )
    
    parser.add_argument(
        "--symptoms", 
        default="",
        help="Patient symptoms description"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file path for results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--model",
        default="densenet121-res224-all",
        help="TorchXRayVision model to use"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize workflow runner
        logger.info("Initializing Radiology Assistant...")
        runner = RadiologyWorkflowRunner()
        
        # Validate inputs
        logger.info("Validating inputs...")
        validation = runner.validate_inputs(args.image, args.symptoms)
        
        if not validation['is_valid']:
            logger.error("Input validation failed:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        if validation['warnings']:
            logger.warning("Input validation warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        # Run analysis
        logger.info("Starting radiology analysis...")
        results = runner.run_analysis(args.image, args.symptoms)
        
        # Display results
        display_results(results)
        
        # Save results if output file specified
        if args.output:
            save_results(results, args.output)
        
        # Exit with appropriate code
        if results['workflow_status'] == 'completed':
            logger.info("Analysis completed successfully")
            sys.exit(0)
        else:
            logger.error("Analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def display_results(results: Dict[str, Any]):
    """Display analysis results in a formatted way."""
    print("\n" + "="*80)
    print("AI RADIOLOGY ASSISTANT - ANALYSIS RESULTS")
    print("="*80)
    
    # Status
    status = results.get('workflow_status', 'unknown')
    print(f"\n📊 STATUS: {status.upper()}")
    
    if status == 'completed':
        # Triage information
        triage_result = results.get('triage_result', {})
        triage_category = triage_result.get('triage_category', 'UNKNOWN')
        urgency_score = triage_result.get('urgency_score', 0.0)
        
        print(f"\n🚨 TRIAGE ASSESSMENT:")
        print(f"   Category: {triage_category}")
        print(f"   Urgency Score: {urgency_score:.3f}")
        
        # Key findings
        image_analysis = results.get('image_analysis', {})
        key_findings = image_analysis.get('key_findings', {})
        
        if key_findings:
            print(f"\n🔍 KEY FINDINGS:")
            for pathology, score in key_findings.items():
                print(f"   • {pathology}: {score:.3f}")
        
        # Recommendations
        recommendations = triage_result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # Executive summary
        report = results.get('report', {})
        executive_summary = report.get('executive_summary', '')
        if executive_summary:
            print(f"\n📋 EXECUTIVE SUMMARY:")
            print(f"   {executive_summary}")
    
    elif status == 'error':
        # Error information
        errors = results.get('errors', [])
        print(f"\n❌ ERRORS:")
        for error in errors:
            print(f"   • {error}")
    
    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
    
    print("\n" + "="*80)
    print("IMPORTANT: This is a prototype AI system for educational/demonstration purposes only.")
    print("All findings should be reviewed by qualified healthcare professionals before clinical use.")
    print("="*80)

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")

if __name__ == "__main__":
    main() 