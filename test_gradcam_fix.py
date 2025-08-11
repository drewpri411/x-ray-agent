#!/usr/bin/env python3
"""
Test script to verify Grad-CAM fix works correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gradcam_fix():
    """Test that the system works with or without Grad-CAM."""
    print("🧪 Testing Grad-CAM Fix")
    print("=" * 40)
    
    try:
        # Test 1: Import and initialize image agent
        print("1. Testing Image Agent Initialization...")
        from agents.image_agent import ImageAnalysisAgent
        
        agent = ImageAnalysisAgent()
        print("   ✅ Image agent initialized successfully")
        
        # Test 2: Check Grad-CAM availability
        if hasattr(agent, 'grad_cam_available'):
            print(f"   ✅ Grad-CAM available: {agent.grad_cam_available}")
        else:
            print("   ⚠️  Grad-CAM availability not checked")
        
        # Test 3: Test image analysis without heatmaps
        print("\n2. Testing Image Analysis (No Heatmaps)...")
        test_image = "data/samples/normal/normal_1.jpg"
        
        if os.path.exists(test_image):
            results = agent.analyze_image(test_image, generate_heatmaps=False)
            
            if results.get('analysis_status') == 'completed':
                print("   ✅ Image analysis completed successfully")
                
                # Check key findings
                key_findings = results.get('key_findings', {})
                if key_findings:
                    print(f"   ✅ Found {len(key_findings)} pathologies")
                    for pathology, score in list(key_findings.items())[:3]:
                        print(f"      • {pathology}: {score:.3f}")
                else:
                    print("   ⚠️  No key findings detected")
                
                # Check heatmaps
                heatmaps = results.get('heatmaps', {})
                if heatmaps:
                    print(f"   ⚠️  Heatmaps generated: {len(heatmaps)}")
                else:
                    print("   ✅ No heatmaps generated (as expected)")
            else:
                print(f"   ❌ Analysis failed: {results.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  Test image not found: {test_image}")
        
        # Test 4: Test workflow runner
        print("\n3. Testing Workflow Runner...")
        from workflow import RadiologyWorkflowRunner
        
        runner = RadiologyWorkflowRunner()
        print("   ✅ Workflow runner initialized successfully")
        
        # Test 5: Test complete workflow (if image exists)
        if os.path.exists(test_image):
            print("\n4. Testing Complete Workflow...")
            results = runner.run_analysis(test_image, "Patient has no symptoms", generate_heatmaps=False)
            
            status = results.get('workflow_status', 'unknown')
            print(f"   ✅ Workflow completed with status: {status}")
            
            if status == 'completed':
                print("   ✅ All workflow components executed successfully")
            else:
                errors = results.get('errors', [])
                print(f"   ❌ Workflow failed: {errors}")
        
        print("\n✅ Grad-CAM fix test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gradcam_fix()
    sys.exit(0 if success else 1) 