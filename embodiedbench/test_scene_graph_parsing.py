#!/usr/bin/env python3
"""
Test script to validate scene graph parsing fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodiedbench.planner.vlm_planner_esca import VLMPlanner

def test_scene_graph_parsing():
    """Test various scene graph JSON formats"""
    
    # Create a minimal VLMPlanner instance for testing
    actions = ["find a Book", "pick up the Book", "find a SideTable", "put down the object in hand"]
    system_prompt = "Test prompt"
    examples = []
    
    planner = VLMPlanner(
        model_name="gpt-4o",
        model_type="remote", 
        actions=actions,
        system_prompt=system_prompt,
        examples=examples,
        n_shot=0
    )
    
    # Test cases with different JSON structures
    test_cases = [
        # Case 1: Complete valid JSON
        {
            "name": "Complete valid JSON",
            "json": {
                "target_state": "Book on right table",
                "current_state": "Book on bed", 
                "target_objects": ["book", "right_table"],
                "target_attributes": [],
                "target_relations": [["book", "on", "right_table"]],
                "current_objects": ["book", "bed"],
                "current_attributes": [],
                "current_relations": [["book", "on", "bed"]]
            }
        },
        
        # Case 2: Missing some keys
        {
            "name": "Missing target_objects key",
            "json": {
                "target_state": "Book on right table",
                "current_state": "Book on bed",
                "current_objects": ["book", "bed"],
                "current_relations": [["book", "on", "bed"]]
            }
        },
        
        # Case 3: Empty structure
        {
            "name": "Empty structure",
            "json": {}
        },
        
        # Case 4: None values
        {
            "name": "None values",
            "json": {
                "target_objects": None,
                "target_attributes": None,
                "target_relations": None,
                "current_objects": ["book"],
                "current_attributes": [],
                "current_relations": []
            }
        }
    ]
    
    print("Testing scene graph parsing robustness...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        try:
            result = planner.get_sg_kws(test_case['json'])
            print(f"✅ SUCCESS: Extracted {len(result[0])} target names, {len(result[1])} current names")
            print(f"   Target names: {result[0]}")
            print(f"   Current names: {result[1]}")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        print()
    
    print("All tests completed!")

if __name__ == "__main__":
    test_scene_graph_parsing()
