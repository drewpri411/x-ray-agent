#!/usr/bin/env python3
"""
Simple test to verify triage fix works without LLM
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_triage_simple():
    """Test triage logic directly without LLM."""
    print("🧪 Testing Triage Logic (Simple)")
    print("=" * 40)
    
    try:
        from agents.image_agent import ImageAnalysisAgent
        from agents.triage_agent import TriageAgent
        
        # Initialize agents
        image_agent = ImageAnalysisAgent()
        triage_agent = TriageAgent()
        print("✅ Agents initialized")
        
        # Test with atelectasis image
        test_image = "data/samples/atelectasis/atelectasis_1.jpg"
        
        if os.path.exists(test_image):
            print(f"✅ Test image found: {test_image}")
            
            # Analyze image
            image_results = image_agent.analyze_image(test_image, generate_heatmaps=False)
            print("✅ Image analysis completed")
            
            # Get urgent findings
            urgent_findings = image_agent.get_urgent_findings(image_results)
            print("✅ Urgent findings extracted")
            
            # Check key findings
            key_findings = urgent_findings.get('key_findings', {})
            print(f"\n🔍 Key Findings:")
            for pathology, score in key_findings.items():
                print(f"   {pathology}: {score:.3f}")
            
            # Check urgent pathologies
            urgent_pathologies = urgent_findings.get('urgent_pathologies', [])
            print(f"\n🚨 Urgent Pathologies: {len(urgent_pathologies)}")
            for finding in urgent_pathologies:
                print(f"   {finding['pathology']}: {finding['score']:.3f} (threshold: {finding['threshold']:.1f})")
            
            # Check urgency score
            urgency_score = urgent_findings.get('urgency_score', 0.0)
            print(f"\n📊 Urgency Score: {urgency_score:.3f}")
            
            # Test triage assessment
            triage_result = triage_agent.assess_urgency(image_results, "")
            triage_category = triage_result.get('triage_category', 'UNKNOWN')
            final_urgency = triage_result.get('urgency_score', 0.0)
            
            print(f"\n🎯 Final Triage Category: {triage_category}")
            print(f"🎯 Final Urgency Score: {final_urgency:.3f}")
            
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
            print(f"❌ Test image not found: {test_image}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_triage_simple()
    sys.exit(0 if success else 1) 