#!/usr/bin/env python3
"""
Test script to verify triage fix works
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_triage_fix():
    """Test that triage works properly with moderate findings."""
    print("🧪 Testing Triage Fix")
    print("=" * 30)
    
    try:
        from workflow import RadiologyWorkflowRunner
        
        # Initialize workflow runner
        runner = RadiologyWorkflowRunner()
        print("✅ Workflow runner initialized")
        
        # Test with atelectasis image (should show moderate findings)
        test_image = "data/samples/atelectasis/atelectasis_1.jpg"
        
        if os.path.exists(test_image):
            print(f"✅ Test image found: {test_image}")
            
            # Run analysis
            results = runner.run_analysis(test_image, "", generate_heatmaps=False)
            
            if results['workflow_status'] == 'completed':
                print("✅ Analysis completed successfully")
                
                # Check triage result
                triage_result = results.get('triage_result', {})
                triage_category = triage_result.get('triage_category', 'UNKNOWN')
                urgency_score = triage_result.get('urgency_score', 0.0)
                
                print(f"📊 Triage Category: {triage_category}")
                print(f"📊 Urgency Score: {urgency_score:.3f}")
                
                # Check key findings
                image_analysis = results.get('image_analysis', {})
                key_findings = image_analysis.get('key_findings', {})
                
                print(f"\n🔍 Key Findings:")
                for pathology, score in key_findings.items():
                    print(f"   {pathology}: {score:.3f}")
                
                # Check if triage is reasonable
                max_score = max(key_findings.values()) if key_findings else 0.0
                
                if max_score >= 0.6:
                    expected_category = "URGENT" if max_score >= 0.7 else "MODERATE"
                elif max_score >= 0.4:
                    expected_category = "MODERATE"
                elif max_score >= 0.2:
                    expected_category = "ROUTINE"
                else:
                    expected_category = "NORMAL"
                
                print(f"\n🎯 Expected Category: {expected_category}")
                print(f"🎯 Actual Category: {triage_category}")
                
                if triage_category != "NORMAL" or max_score < 0.5:
                    print("✅ Triage fix working - not everything is NORMAL")
                    return True
                else:
                    print("❌ Triage still too conservative")
                    return False
            else:
                print(f"❌ Analysis failed: {results.get('errors', [])}")
                return False
        else:
            print(f"❌ Test image not found: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_triage_fix()
    sys.exit(0 if success else 1) 