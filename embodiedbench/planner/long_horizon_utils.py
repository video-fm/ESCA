"""
Utilities for improving long horizon task performance
"""

def create_long_horizon_prompt_addition():
    """Additional prompt guidance for long horizon tasks"""
    return """
## **LONG HORIZON TASK GUIDELINES**

**CRITICAL RULES:**
1. **SIMPLE PLANS**: Create plans with 5-10 steps maximum, not 20+ steps
2. **VALID ACTION IDs**: Only use action IDs from 0 to the maximum available (check the action list!)
3. **STATE TRACKING**: Remember where objects are after each action
4. **SLICE REQUIREMENTS**: Objects must be on countertop/table to slice, not in fridge
5. **TOOL REQUIREMENTS**: Need a knife in hand to slice objects

**COMMON MISTAKES TO AVOID:**
- Using invalid action IDs like 230, 250+ (check action list first!)
- Trying to slice objects while they're inside containers
- Creating overly complex 20+ step plans
- Not tracking object locations between steps

**BETTER APPROACH:**
1. Break task into 2-3 major phases
2. Execute each phase completely before planning next
3. Use only valid action IDs from the provided list
"""

def simplify_long_horizon_plan(instruction):
    """Analyze instruction and provide simplified approach"""
    if "both" in instruction.lower() and "slice" in instruction.lower():
        return """
**SIMPLIFIED APPROACH FOR DUAL SLICE TASKS:**
1. **Phase 1**: Handle first object (find, slice if needed, store)
2. **Phase 2**: Handle second object (find, slice if needed, store) 
3. **Phase 3**: Final arrangement

**EXECUTION STRATEGY:**
- Execute 3-5 actions, then replan based on current state
- Don't pre-plan entire 20+ step sequence
"""
    return ""

def validate_action_ids(plan, max_action_id):
    """Validate that all action IDs in plan are valid"""
    invalid_actions = []
    for step in plan:
        if isinstance(step, dict) and 'action_id' in step:
            aid = step['action_id']
            if aid >= max_action_id or aid < 0:
                invalid_actions.append((aid, step.get('action_name', 'unknown')))
    
    return invalid_actions
