laser_sg_examples_no_revert = [
'''
Human instruction: Pick up the alarm clock and turn on the lamp.
The target state: I am holding an alarm clock and there should be a lightened lamp in my sight.
The current state: I am in a bedroom.  A small alarm clock sits on the bedside table next to a switched-off desk lamp. 
The current scene graph: 
    The objects are: [alarm_clock, bedside_table, lamp]
    The attributes are: []
    The relations are: [(alarm_clock, on, bedside_table), (alarm_clock, next to, lamp)]
The target state:
    The objects are: [alarm_clock, lamp]
    The attributes are: [(holding, alarm_clock), (lightened, lamp)]
    The relations are: []
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "I am in a bedroom. A small alarm clock sits on the bedside table next to a switched-off desk lamp.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state whether they are statisied.\\n I should hold an alarm clock. I can see an alarm clock in the scene. I am not holding an alarm clock now. Not satisfied.\\n I should lighten the lamp. I can see a lamp in the scene. It is not lightened. Not satisfied.\\n I must hold the alarm clock and then illuminate the area by turning on the desk lamp.\\n I need to check the pre- and post-conditions for the actions.\\n The \"Hold\" action requires nothing in hand. I am not holding any thing in hand, satisfied.",
    "language_plan": "First, locate the alarm clock in the room. Once found, carefully pick up the alarm clock. Next, identify the location of the desk lamp. After locating the desk lamp, proceed to turn it on.",
    "executable_plan": [
        {"action_id": 57, "action_name": "find a AlarmClock"},
        {"action_id": 98, "action_name": "pick up the AlarmClock"},
        {"action_id": 63, "action_name": "find a DeskLamp"},
        {"action_id": 151, "action_name": "turn on the DeskLamp"}
    ]
}
''',

'''
Human instruction: Examine a credit card by the light of a tall lamp.
The target state: I am holding a credit card, and a tall lamp is lightened in the scene so that I can see the card clearly.
The current state: I am in a lounge with a floor lamp in the corner and a credit card lying on a coffee table.
The current scene graph: 
    The objects are: [credit_card, coffee_table, floor_lamp]
    The attributes are: []
    The relations are: [(credit_card, on, coffee_table)]
The target state:
    The objects are: [credit_card, floor_lamp]
    The attributes are: [(holding, credit_card), (lightened, floor_lamp)]
    The relations are: []
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A lounge with a floor lamp in the corner and a credit card lying on a coffee table.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to assess whether the current scene satisfies the target state.\\n First, I should be holding a credit card.\\n Currently, the credit card is on the coffee table, not in my possession, so this condition is not satisfied.\\n Second, the floor lamp should be lightened to help me examine the credit card.\\n At present, the floor lamp is in the corner but not turned on, so this condition is also not satisfied.\\n Therefore, to achieve the target state, I must pick up the credit card from the coffee table and turn on the floor lamp to illuminate the area.\\n I also need to be aware of the pre-condition and post condition of the actions I have planed.\\n pick up requires I am holding nothing.\\n Because I am holding nothing, the precondition is satisfied.",
    "language_plan": "Start by locating the credit card in the room. Once you have found the credit card, proceed to pick it up. Then, search for the floor lamp and find its location. After finding the floor lamp, turn it on to illuminate the area so you can examine the credit card effectively.",
    "executable_plan": [
        {"action_id": 56, "action_name": "find a CreditCard"},
        {"action_id": 96, "action_name": "pick up the CreditCard"},
        {"action_id": 27, "action_name": "find a FloorLamp"},
        {"action_id": 153, "action_name": "turn on the FloorLamp"}
    ]
}
''',

'''
Human instruction: Use lamp light to look at the remote.
The target state: I am holding a remote control, and a floor lamp is turned on so I can inspect the remote clearly.
The current state: I am in a living room. A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.
The current scene graph: 
    The objects are: [remote_control, sofa, floor_lamp]
    The attributes are: []
    The relations are: [(remote_control, on, arm_of_sofa), (sofa, facing, floor_lamp)]
The target state:
    The objects are: [remote_control, floor_lamp]
    The attributes are: [(holding, remote_control), (lightened, floor_lamp)]
    The relations are: []
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to determine if the current state fulfills the goal.\\n First, I should be holding the remote control. However, it is currently resting on the arm of the sofa, so I will need to use a \"Find\" action to find the remote control.\\n Second, the floor lamp should be lightened to provide visibility.\\n At the moment, the floor lamp is off, which also does not meet the target condition.\\n Therefore, I need to pick up the remote control and turn on the floor lamp to achieve the desired state.",
    "language_plan": "First, locate the remote control in the room. Once the remote control is found, pick it up carefully. Next, search for the floor lamp and determine its location. After you have located the floor lamp, turn it on to provide sufficient light for examining the remote control.",
    "executable_plan": [
        {"action_id": 68, "action_name": "find a RemoteControl"},
        {"action_id": 116, "action_name": "pick up the RemoteControl"},
        {"action_id": 27, "action_name": "find a FloorLamp"},
        {"action_id": 153, "action_name": "turn on the FloorLamp"}
    ]
}
''',

'''
Human instruction: Set the box on the table.
The target state: The box should be placed on top of the dining table.
The current state: I am in a dining area. A cardboard box is on the floor near a dining table.
The current scene graph: 
    The objects are: [box, dining_table]
    The attributes are: []
    The relations are: [(box, near, dining_table), (box, on, floor)]
The target state:
    The objects are: [box, dining_table]
    The attributes are: []
    The relations are: [(box, on, dining_table)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
 
Output: {
    "visual_state_description": "A cardboard box is on the floor near a dining table.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check if the current configuration satisfies the target state.\\n The goal is for the box to be on the dining table. Right now, the box is on the floor near the table, which does not fulfill the required spatial relation.\\n Therefore, I need to pick up the box and place it directly on top of the dining table to achieve the target state.",
    "language_plan": "First, locate the box within the room using the \"Find\" action. After finding the box, gently pick it up. Next, search for the dining table and identify its location. Once the dining table is found, place the box on the table carefully.",
    "executable_plan": [
        {"action_id": 61, "action_name": "find a Box"},
        {"action_id": 103, "action_name": "pick up the Box"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
''',

'''
Human instruction: Place a heated slice of tomato in the kitchen sink.
The target state: A slice of tomato that has been heated is placed inside the kitchen sink.
The current state: I am in a kitchen. A whole tomato sits on the counter; a microwave and a sink are visible nearby.
The current scene graph: 
    The objects are: [tomato, counter, microwave, sink]
    The attributes are: [(whole, tomato), (cold, tomato)]
    The relations are: [(tomato, on, counter)]
The target state:
    The objects are: [tomato_slice, sink]
    The attributes are: [(sliced, tomate_slice), (heated, tomato_slice)]
    The relations are: [(tomato_slice, in, sink)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A whole tomato sits on the counter; a microwave and sink are visible nearby.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "To satisfy the target state, I need a heated slice of tomato to be placed inside the sink.\\n 
        Currently, there is only a whole tomato on the counter, and the tomato is not sliced or heated. The target state is not satisfied.\\n 
        Thus, I need to slice the tomato, heat the slice in the microwave, and place the heated slice into the sink.",
    "language_plan": "Find a butter knife and pick it up, locate a tomato and slice it. Find a sink and put down the knife. Locate a slice of tomato and pick it up, find a microwave, and open it. Place the tomato slice inside and close the microwave. Turn on the microwave to heat the tomato slice, then turn off the microwave and open it. Pick up the heated tomato slice, close the microwave, locate the kitchen sink, and place the heated tomato slice in the sink.",
    "executable_plan": [
        {"action_id": 36, "action_name": "find a ButterKnife"},
        {"action_id": 130, "action_name": "pick up the ButterKnife"},
        {"action_id": 35, "action_name": "find a Tomato"},
        {"action_id": 159, "action_name": "slice the Tomato"},
        {"action_id": 79, "action_name": "find a Sink"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 35, "action_name": "find a Tomato"},
        {"action_id": 129, "action_name": "pick up the Tomato"},
        {"action_id": 38, "action_name": "find a Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 149, "action_name": "turn on the Microwave"},
        {"action_id": 150, "action_name": "turn off the Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 35, "action_name": "find a Tomato"},
        {"action_id": 129, "action_name": "pick up the Tomato"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 79, "action_name": "find a Sink"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
''',

'''
Human instruction: Move the towel on the hanger into the cabinet.
The target state: The hand towel should be placed inside the cabinet, and the cabinet should be closed.
The current state: I am in a bathroom. A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.
The current scene graph: 
    The objects are: [hand_towel, hanger, cabinet]
    The attributes are: [(closed, cabinet)]
    The relations are: [(hand_towel, on, hanger), (cabinet, near, hanger)]
The target state:
    The objects are: [hand_towel, cabinet]
    The attributes are: [(closed, cabinet)]
    The relations are: [(hand_towel, in, cabinet)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "The hand towel is currently on the hanger and not inside the cabinet. Not satisfied.\\n The cabinet is closed, so I must open it before placing the towel inside, and close it again afterward to satisfy the target state.\\n Therefore, I need to retrieve the towel, open the cabinet, store the towel, and then close the cabinet.",
    "language_plan": "Begin by locating the hand towel on the hanger. Once you have found the hand towel, proceed to pick it up. Next, search for the cabinet and identify its location using the \"Find\" action. After finding the cabinet, open it. Place the hand towel inside the cabinet, and finally, close the cabinet door.",
    "executable_plan": [
        {"action_id": 9, "action_name": "find a HandTowel"},
        {"action_id": 89, "action_name": "pick up the HandTowel"},
        {"action_id": 48, "action_name": "find a Cabinet"},
        {"action_id": 145, "action_name": "open the Cabinet"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 146, "action_name": "close the Cabinet"}
    ]
}
''',

'''
Human instruction: Place a microwaved slice of apple on to the white table.
The target state: A microwaved apple slice is placed on the white dining table.
The current state: I am in a kitchen. A whole apple lies next to a microwave; a white dining table is visible across the kitchen.
The current scene graph: 
    The objects are: [apple, microwave, dining_table]
    The attributes are: [(white, dining_table), (closed, microwave), (whole, apple)]
    The relations are: [(apple, next to, microwave), (dining_table, near, kitchen)]
The target state:
    The objects are: [microwaved_apple_slice, dining_table]
    The attributes are: [(heated, microwaved_apple_slice), (sliced, microwaved_apple_slice), (white, dining_table)]
    The relations are: [(microwaved_apple_slice, on, dining_table)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A whole apple lies next to a microwave; a white dining table is visible across the kitchen.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a microwaved slice of apple placed on the white dining table.\\n I can see a whole apple in the scene, but it has not been sliced. Not satisfied.\\n I should heat the slice using the microwave. The microwave is visible, but no slice is currently inside or being heated. Not satisfied.\\n I should then place the heated slice on the white dining table. The table is visible, but it is currently empty. Not satisfied.\\n Therefore, I must slice the apple, microwave a slice, and place the microwaved slice on the white dining table.",
    "language_plan": "Locate an apple and pick it up, find a dining table and put the apple down, look for a knife and pick it up, find another apple and slice it, find the dining table and put the knife down, locate an apple and pick it up, search for a microwave and open it, put the apple in the microwave and close it, turn on the microwave to heat the apple slice, turn off the microwave and open it, pick up the microwaved apple slice, close the microwave, find the white table, and place the microwaved apple slice on the white table.",
    "executable_plan": [
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 107, "action_name": "pick up the Apple"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 34, "action_name": "find a Knife"},
        {"action_id": 127, "action_name": "pick up the Knife"},
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 160, "action_name": "slice the Apple"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 107, "action_name": "pick up the Apple"},
        {"action_id": 38, "action_name": "find a Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 149, "action_name": "turn on the Microwave"},
        {"action_id": 150, "action_name": "turn off the Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 107, "action_name": "pick up the Apple"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
''',

'''
Human instruction: slice a loaf of bread put a slice on the counter.
The target state: A sliced piece of bread is placed on the countertop. The knife is no longer in use.
The current state: I am in a kitchen. A loaf of bread sits beside a bread knife on the countertop.
The current scene graph: 
    The objects are: [bread_loaf, bread_knife, countertop]
    The attributes are: [(whole, bread_loaf)]
    The relations are: [(bread_loaf, beside, bread_knife), (bread_loaf, on, countertop), (bread_knife, on, countertop)]
The target state:
    The objects are: [bread_slice, countertop]
    The attributes are: [(not holding, bread_knife), (sliced, bread_slice)]
    The relations are: [(bread_slice, on, countertop)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A loaf of bread sits beside a bread knife on the countertop.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a sliced piece of bread placed on the countertop. I can see a loaf of bread and a knife in the scene, but the bread has not been sliced. Not satisfied.\\n I can also see the countertop, but there is no slice of bread on it. Not satisfied.\\n Therefore, I must use the knife to slice the bread and place one of the slices on the countertop.",
    "language_plan": "First locate a knife and pick it up, find a loaf of bread, and slice the bread. Find a cabinet, open it, and put the knife inside, then close the cabinet. Find a slice of bread, pick it up, find the countertop, and place the slice of bread on the counter.",
    "executable_plan": [
        {"action_id": 34, "action_name": "find a Knife"},
        {"action_id": 127, "action_name": "pick up the Knife"},
        {"action_id": 53, "action_name": "find a Bread"},
        {"action_id": 161, "action_name": "slice the Bread"},
        {"action_id": 48, "action_name": "find a Cabinet"},
        {"action_id": 145, "action_name": "open the Cabinet"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 146, "action_name": "close the Cabinet"},
        {"action_id": 53, "action_name": "find a Bread"},
        {"action_id": 92, "action_name": "pick up the Bread"},
        {"action_id": 78, "action_name": "find a Fridge"},
        {"action_id": 139, "action_name": "open the Fridge"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 140, "action_name": "close the Fridge"},
        {"action_id": 139, "action_name": "open the Fridge"},
        {"action_id": 53, "action_name": "find a Bread"},
        {"action_id": 92, "action_name": "pick up the Bread"},
        {"action_id": 140, "action_name": "close the Fridge"},
        {"action_id": 39, "action_name": "find a CounterTop"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
''',

'''
Human instruction: Put a clean pan in the refrigerator.
The target state: A clean pan is placed inside the refrigerator, and the fridge is closed.
The current state: I am in a kitchen. A dirty pan rests in the sink next to a faucet; the fridge is across the room.
The current scene graph: 
    The objects are: [pan, sink, faucet, fridge]
    The attributes are: [(dirty, pan), (closed, fridge)]
    The relations are: [(pan, in, sink), (sink, next to, faucet), (fridge, across_from, sink)]
The target state:
    The objects are: [pan, fridge]
    The attributes are: [(clean, pan), (closed, fridge)]
    The relations are: [(pan, in, fridge)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A dirty pan rests in the sink next to a faucet; the fridge is across the room.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a clean pan placed inside the refrigerator. I can see a pan, but it is currently dirty and sitting in the sink. Not satisfied.\\n The fridge is visible but closed, and the pan is not inside it. Not satisfied.\\n Therefore, I must wash the pan to make it clean, then place it into the refrigerator and close the door.",
    "language_plan": "Locate a pan and pick it up, find a sink, and put the pan down in the sink. Find a faucet, turn it on to clean the pan, and then turn it off. Pick up the cleaned pan, find the refrigerator, open the refrigerator, place the pan inside, and close the refrigerator.",
    "executable_plan": [
        {"action_id": 7, "action_name": "find a Pan"},
        {"action_id": 86, "action_name": "pick up the Pan"},
        {"action_id": 79, "action_name": "find a Sink"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 2, "action_name": "find a Faucet"},
        {"action_id": 155, "action_name": "turn on the Faucet"},
        {"action_id": 156, "action_name": "turn off the Faucet"},
        {"action_id": 7, "action_name": "find a Pan"},
        {"action_id": 86, "action_name": "pick up the Pan"},
        {"action_id": 78, "action_name": "find a Fridge"},
        {"action_id": 139, "action_name": "open the Fridge"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 140, "action_name": "close the Fridge"}
    ]
}
''',

'''
Human instruction: Put a box from the couch with a phone in it on the black desk.
The target state: A phone is inside the box, and the box is placed on the black desk.
The current state: I am in a living room. A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.
The current scene graph: 
    The objects are: [cellphone, box, couch, desk]
    The attributes are: [(black, desk), (open, box)]
    The relations are: [(cellphone, beside, box), (box, on, couch), (desk, opposite, couch)]
The target state:
    The objects are: [cellphone, box, desk]
    The attributes are: [(black, desk)]
    The relations are: [(cellphone, in, box), (box, on, desk)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n First, the phone should be inside the box. Currently, the phone is next to the box on the couch, not inside it. Not satisfied.\\n Second, the box should be on the black desk. Currently, the box is on the couch. Not satisfied.\\n Therefore, I must first place the phone inside the box, then move the box from the couch to the black desk.",
    "language_plan": "First, find the phone, pick it up, locate the box on the couch, place the phone inside the box, pick up the box. Then, use the \"Find\" action to find the black desk, and place the box on the desk.",
    "executable_plan": [
        {"action_id": 12, "action_name": "find a CellPhone"},
        {"action_id": 93, "action_name": "pick up the CellPhone"},
        {"action_id": 61, "action_name": "find a Box"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 103, "action_name": "pick up the Box"},
        {"action_id": 49, "action_name": "find a Desk"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
'''
]

laser_sg_examples_pre_post = [
'''
Human instruction: Pick up the alarm clock and turn on the lamp.
The target state: I am holding an alarm clock and there should be a lightened lamp in my sight.
The current state: I am in a bedroom.  A small alarm clock sits on the bedside table next to a switched-off desk lamp. 
The current scene graph: 
    The objects are: [alarm_clock, bedside_table, lamp]
    The attributes are: []
    The relations are: [(alarm_clock, on, bedside_table), (alarm_clock, next to, lamp)]
The target state:
    The objects are: [alarm_clock, lamp]
    The attributes are: [(holding, alarm_clock), (lightened, lamp)]
    The relations are: []
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "I am in a bedroom. A small alarm clock sits on the bedside table next to a switched-off desk lamp.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state whether they are statisied.\\n I should hold an alarm clock. I can see an alarm clock in the scene. I am not holding an alarm clock now. Not satisfied.\\n I should lighten the lamp. I can see a lamp in the scene. It is not lightened. Not satisfied.\\n I must hold the alarm clock and then illuminate the area by turning on the desk lamp.\\n I need to check the pre- and post-conditions for the actions.\\n The \"Hold\" action requires nothing in hand. I am not holding any thing in hand, satisfied.",
    "language_plan": "First, locate the alarm clock in the room. Once found, carefully pick up the alarm clock. Next, identify the location of the desk lamp. After locating the desk lamp, proceed to turn it on.",
    "executable_plan": [
        {"action_id": 57, "action_name": "find a AlarmClock"},
        {"action_id": 98, "action_name": "pick up the AlarmClock"},
        {"action_id": 63, "action_name": "find a DeskLamp"},
        {"action_id": 151, "action_name": "turn on the DeskLamp"}
    ]
}
''',

'''
Human instruction: Examine a credit card by the light of a tall lamp.
The target state: I am holding a credit card, and a tall lamp is lightened in the scene so that I can see the card clearly.
The current state: I am in a lounge with a floor lamp in the corner and a credit card lying on a coffee table.
The current scene graph: 
    The objects are: [credit_card, coffee_table, floor_lamp]
    The attributes are: []
    The relations are: [(credit_card, on, coffee_table)]
The target state:
    The objects are: [credit_card, floor_lamp]
    The attributes are: [(holding, credit_card), (lightened, floor_lamp)]
    The relations are: []
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A lounge with a floor lamp in the corner and a credit card lying on a coffee table.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to assess whether the current scene satisfies the target state.\\n First, I should be holding a credit card.\\n Currently, the credit card is on the coffee table, not in my possession, so this condition is not satisfied.\\n Second, the floor lamp should be lightened to help me examine the credit card.\\n At present, the floor lamp is in the corner but not turned on, so this condition is also not satisfied.\\n Therefore, to achieve the target state, I must pick up the credit card from the coffee table and turn on the floor lamp to illuminate the area.\\n I also need to be aware of the pre-condition and post condition of the actions I have planed.\\n pick up requires I am holding nothing.\\n Because I am holding nothing, the precondition is satisfied.",
    "language_plan": "Start by locating the credit card in the room. Once you have found the credit card, proceed to pick it up. Then, search for the floor lamp and find its location. After finding the floor lamp, turn it on to illuminate the area so you can examine the credit card effectively.",
    "executable_plan": [
        {"action_id": 56, "action_name": "find a CreditCard"},
        {"action_id": 96, "action_name": "pick up the CreditCard"},
        {"action_id": 27, "action_name": "find a FloorLamp"},
        {"action_id": 153, "action_name": "turn on the FloorLamp"}
    ]
}
''',

'''
Human instruction: Use lamp light to look at the remote.
The target state: I am holding a remote control, and a floor lamp is turned on so I can inspect the remote clearly.
The current state: I am in a living room. A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.
The current scene graph: 
    The objects are: [remote_control, sofa, floor_lamp]
    The attributes are: []
    The relations are: [(remote_control, on, arm_of_sofa), (sofa, facing, floor_lamp)]
The target state:
    The objects are: [remote_control, floor_lamp]
    The attributes are: [(holding, remote_control), (lightened, floor_lamp)]
    The relations are: []
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to determine if the current state fulfills the goal.\\n First, I should be holding the remote control. However, it is currently resting on the arm of the sofa, so I will need to use a \"Find\" action to find the remote control.\\n Second, the floor lamp should be lightened to provide visibility.\\n At the moment, the floor lamp is off, which also does not meet the target condition.\\n Therefore, I need to pick up the remote control and turn on the floor lamp to achieve the desired state.",
    "language_plan": "First, locate the remote control in the room. Once the remote control is found, pick it up carefully. Next, search for the floor lamp and determine its location. After you have located the floor lamp, turn it on to provide sufficient light for examining the remote control.",
    "executable_plan": [
        {"action_id": 68, "action_name": "find a RemoteControl"},
        {"action_id": 116, "action_name": "pick up the RemoteControl"},
        {"action_id": 27, "action_name": "find a FloorLamp"},
        {"action_id": 153, "action_name": "turn on the FloorLamp"}
    ]
}
''',

'''
Human instruction: Set the box on the table.
The target state: The box should be placed on top of the dining table.
The current state: I am in a dining area. A cardboard box is on the floor near a dining table.
The current scene graph: 
    The objects are: [box, dining_table]
    The attributes are: []
    The relations are: [(box, near, dining_table), (box, on, floor)]
The target state:
    The objects are: [box, dining_table]
    The attributes are: []
    The relations are: [(box, on, dining_table)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
 
Output: {
    "visual_state_description": "A cardboard box is on the floor near a dining table.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check if the current configuration satisfies the target state.\\n The goal is for the box to be on the dining table. Right now, the box is on the floor near the table, which does not fulfill the required spatial relation.\\n Therefore, I need to pick up the box and place it directly on top of the dining table to achieve the target state.",
    "language_plan": "First, locate the box within the room using the \"Find\" action. After finding the box, gently pick it up. Next, search for the dining table and identify its location. Once the dining table is found, place the box on the table carefully.",
    "executable_plan": [
        {"action_id": 61, "action_name": "find a Box"},
        {"action_id": 103, "action_name": "pick up the Box"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
''',

'''
Human instruction: Place a heated slice of tomato in the kitchen sink.
The target state: A slice of tomato that has been heated is placed inside the kitchen sink.
The current state: I am in a kitchen. A whole tomato sits on the counter; a microwave and a sink are visible nearby.
The current scene graph: 
    The objects are: [tomato, counter, microwave, sink]
    The attributes are: [(whole, tomato), (cold, tomato)]
    The relations are: [(tomato, on, counter)]
The target state:
    The objects are: [tomato_slice, sink]
    The attributes are: [(sliced, tomate_slice), (heated, tomato_slice)]
    The relations are: [(tomato_slice, in, sink)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A whole tomato sits on the counter; a microwave and sink are visible nearby.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "To satisfy the target state, I need a heated slice of tomato to be placed inside the sink.\\n 
        Currently, there is only a whole tomato on the counter, and the tomato is not sliced or heated. The target state is not satisfied.\\n 
        Thus, I need to slice the tomato, heat the slice in the microwave, and place the heated slice into the sink.",
    "language_plan": "Find a butter knife and pick it up, locate a tomato and slice it. Find a sink and put down the knife. Locate a slice of tomato and pick it up, find a microwave, and open it. Place the tomato slice inside and close the microwave. Turn on the microwave to heat the tomato slice, then turn off the microwave and open it. Pick up the heated tomato slice, close the microwave, locate the kitchen sink, and place the heated tomato slice in the sink.",
    "executable_plan": [
        {
            "action_id": 36,
            "action_name": "find a ButterKnife",
            "pre_condition": [],
            "post_condition": [["focus on", "butter_knife"]]
        },
        {
            "action_id": 130,
            "action_name": "pick up the ButterKnife",
            "pre_condition": [["focus on", "butter_knife"]],
            "post_condition": [["holding", "butter_knife"]]
        },
        {
            "action_id": 35,
            "action_name": "find a Tomato",
            "pre_condition": [],
            "post_condition": [["focus on", "tomato"]]
        },
        {
            "action_id": 159,
            "action_name": "slice the Tomato",
            "pre_condition": [["focus on", "tomato"], ["holding", "butter_knife"]],
            "post_condition": [["sliced", "tomato_slice"], ["tomato_slice", "on", "counter"]]
        },
        {
            "action_id": 48,
            "action_name": "find a Cabinet",
            "pre_condition": [],
            "post_condition": [["focus on", "cabinet"]]
        },
        {
            "action_id": 145,
            "action_name": "open the Cabinet",
            "pre_condition": [["focus on", "cabinet"]],
            "post_condition": [["open", "cabinet"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "butter_knife"], ["open", "cabinet"]],
            "post_condition": [["butter_knife", "in", "cabinet"], ["not holding", "butter_knife"]]
        },
        {
            "action_id": 146,
            "action_name": "close the Cabinet",
            "pre_condition": [["open", "cabinet"]],
            "post_condition": [["closed", "cabinet"]]
        },
        {
            "action_id": 60,
            "action_name": "find a Tomato Slice",
            "pre_condition": [],
            "post_condition": [["focus on", "tomato_slice"]]
        },
        {
            "action_id": 129,
            "action_name": "pick up the Tomato Slice",
            "pre_condition": [["focus on", "tomato_slice"]],
            "post_condition": [["holding", "tomato_slice"]]
        },
        {
            "action_id": 38,
            "action_name": "find a Microwave",
            "pre_condition": [],
            "post_condition": [["focus on", "microwave"]]
        },
        {
            "action_id": 143,
            "action_name": "open the Microwave",
            "pre_condition": [["focus on", "microwave"]],
            "post_condition": [["open", "microwave"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "tomato_slice"], ["open", "microwave"]],
            "post_condition": [["tomato_slice", "in", "microwave"], ["not holding", "tomato_slice"]]
        },
        {
            "action_id": 144,
            "action_name": "close the Microwave",
            "pre_condition": [["open", "microwave"]],
            "post_condition": [["closed", "microwave"]]
        },
        {
            "action_id": 149,
            "action_name": "turn on the Microwave",
            "pre_condition": [["closed", "microwave"], ["tomato_slice", "in", "microwave"]],
            "post_condition": [["heated", "tomato_slice"]]
        },
        {
            "action_id": 150,
            "action_name": "turn off the Microwave",
            "pre_condition": [["heated", "tomato_slice"]],
            "post_condition": [["microwave", "off"]]
        },
        {
            "action_id": 143,
            "action_name": "open the Microwave",
            "pre_condition": [["microwave", "off"]],
            "post_condition": [["open", "microwave"]]
        },
        {
            "action_id": 129,
            "action_name": "pick up the Tomato Slice",
            "pre_condition": [["tomato_slice", "in", "microwave"], ["open", "microwave"]],
            "post_condition": [["holding", "tomato_slice"]]
        },
        {
            "action_id": 144,
            "action_name": "close the Microwave",
            "pre_condition": [["open", "microwave"]],
            "post_condition": [["closed", "microwave"]]
        },
        {
            "action_id": 79,
            "action_name": "find a Sink",
            "pre_condition": [],
            "post_condition": [["focus on", "sink"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "tomato_slice"], ["focus on", "sink"]],
            "post_condition": [["tomato_slice", "in", "sink"], ["not holding", "tomato_slice"]]
        }
    ]
}
''',

'''
Human instruction: Move the towel on the hanger into the cabinet.
The target state: The hand towel should be placed inside the cabinet, and the cabinet should be closed.
The current state: I am in a bathroom. A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.
The current scene graph: 
    The objects are: [hand_towel, hanger, cabinet]
    The attributes are: [(closed, cabinet)]
    The relations are: [(hand_towel, on, hanger), (cabinet, near, hanger)]
The target state:
    The objects are: [hand_towel, cabinet]
    The attributes are: [(closed, cabinet)]
    The relations are: [(hand_towel, in, cabinet)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "The hand towel is currently on the hanger and not inside the cabinet. Not satisfied.\\n The cabinet is closed, so I must open it before placing the towel inside, and close it again afterward to satisfy the target state.\\n Therefore, I need to retrieve the towel, open the cabinet, store the towel, and then close the cabinet.",
    "language_plan": "Begin by locating the hand towel on the hanger. Once you have found the hand towel, proceed to pick it up. Next, search for the cabinet and identify its location using the \"Find\" action. After finding the cabinet, open it. Place the hand towel inside the cabinet, and finally, close the cabinet door.",
    "executable_plan": [
        {
            "action_id": 9,
            "action_name": "find a HandTowel",
            "pre_condition": [],
            "post_condition": [["focus on", "hand_towel"]]
        },
        {
            "action_id": 89,
            "action_name": "pick up the HandTowel",
            "pre_condition": [["focus on", "hand_towel"]],
            "post_condition": [["holding", "hand_towel"], ["hand_towel", "not on", "hanger"]]
        },
        {
            "action_id": 48,
            "action_name": "find a Cabinet",
            "pre_condition": [],
            "post_condition": [["focus on", "cabinet"]]
        },
        {
            "action_id": 145,
            "action_name": "open the Cabinet",
            "pre_condition": [["focus on", "cabinet"], ["closed", "cabinet"]],
            "post_condition": [["open", "cabinet"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "hand_towel"], ["open", "cabinet"]],
            "post_condition": [["hand_towel", "in", "cabinet"], ["not holding", "hand_towel"]]
        },
        {
            "action_id": 146,
            "action_name": "close the Cabinet",
            "pre_condition": [["open", "cabinet"]],
            "post_condition": [["closed", "cabinet"]]
        }
    ]
}
''',

'''
Human instruction: Place a microwaved slice of apple on to the white table.
The target state: A microwaved apple slice is placed on the white dining table.
The current state: I am in a kitchen. A whole apple lies next to a microwave; a white dining table is visible across the kitchen.
The current scene graph: 
    The objects are: [apple, microwave, dining_table]
    The attributes are: [(white, dining_table), (closed, microwave), (whole, apple)]
    The relations are: [(apple, next to, microwave), (dining_table, near, kitchen)]
The target state:
    The objects are: [microwaved_apple_slice, dining_table]
    The attributes are: [(heated, microwaved_apple_slice), (sliced, microwaved_apple_slice), (white, dining_table)]
    The relations are: [(microwaved_apple_slice, on, dining_table)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A whole apple lies next to a microwave; a white dining table is visible across the kitchen.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a microwaved slice of apple placed on the white dining table.\\n I can see a whole apple in the scene, but it has not been sliced. Not satisfied.\\n I should heat the slice using the microwave. The microwave is visible, but no slice is currently inside or being heated. Not satisfied.\\n I should then place the heated slice on the white dining table. The table is visible, but it is currently empty. Not satisfied.\\n Therefore, I must slice the apple, microwave a slice, and place the microwaved slice on the white dining table.",
    "language_plan": "Locate an apple and pick it up, find a dining table and put the apple down, look for a knife and pick it up, find another apple and slice it, find the dining table and put the knife down, locate an apple and pick it up, search for a microwave and open it, put the apple in the microwave and close it, turn on the microwave to heat the apple slice, turn off the microwave and open it, pick up the microwaved apple slice, close the microwave, find the white table, and place the microwaved apple slice on the white table.",
    "executable_plan": [
        {
            "action_id": 22,
            "action_name": "find an Apple",
            "pre_condition": [],
            "post_condition": [["focus on", "apple"]]
        },
        {
            "action_id": 107,
            "action_name": "pick up the Apple",
            "pre_condition": [["focus on", "apple"]],
            "post_condition": [["holding", "apple"]]
        },
        {
            "action_id": 18,
            "action_name": "find the DiningTable",
            "pre_condition": [],
            "post_condition": [["focus on", "dining_table"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "apple"], ["focus on", "dining_table"]],
            "post_condition": [["apple", "on", "dining_table"], ["not holding", "apple"]]
        },
        {
            "action_id": 34,
            "action_name": "find a Knife",
            "pre_condition": [],
            "post_condition": [["focus on", "knife"]]
        },
        {
            "action_id": 127,
            "action_name": "pick up the Knife",
            "pre_condition": [["focus on", "knife"]],
            "post_condition": [["holding", "knife"]]
        },
        {
            "action_id": 22,
            "action_name": "find an Apple",
            "pre_condition": [],
            "post_condition": [["focus on", "apple"]]
        },
        {
            "action_id": 160,
            "action_name": "slice the Apple",
            "pre_condition": [["focus on", "apple"], ["holding", "knife"]],
            "post_condition": [["sliced", "apple_slice"], ["apple_slice", "on", "dining_table"]]
        },
        {
            "action_id": 48,
            "action_name": "find a Cabinet",
            "pre_condition": [],
            "post_condition": [["focus on", "cabinet"]]
        },
        {
            "action_id": 145,
            "action_name": "open the Cabinet",
            "pre_condition": [["focus on", "cabinet"]],
            "post_condition": [["open", "cabinet"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "knife"], ["open", "cabinet"]],
            "post_condition": [["knife", "in", "cabinet"], ["not holding", "knife"]]
        },
        {
            "action_id": 146,
            "action_name": "close the Cabinet",
            "pre_condition": [["open", "cabinet"]],
            "post_condition": [["closed", "cabinet"]]
        },
        {
            "action_id": 61,
            "action_name": "find an Apple Slice",
            "pre_condition": [],
            "post_condition": [["focus on", "apple_slice"]]
        },
        {
            "action_id": 107,
            "action_name": "pick up the Apple Slice",
            "pre_condition": [["focus on", "apple_slice"]],
            "post_condition": [["holding", "apple_slice"]]
        },
        {
            "action_id": 38,
            "action_name": "find the Microwave",
            "pre_condition": [],
            "post_condition": [["focus on", "microwave"]]
        },
        {
            "action_id": 143,
            "action_name": "open the Microwave",
            "pre_condition": [["focus on", "microwave"]],
            "post_condition": [["open", "microwave"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "apple_slice"], ["open", "microwave"]],
            "post_condition": [["apple_slice", "in", "microwave"], ["not holding", "apple_slice"]]
        },
        {
            "action_id": 144,
            "action_name": "close the Microwave",
            "pre_condition": [["open", "microwave"]],
            "post_condition": [["closed", "microwave"]]
        },
        {
            "action_id": 149,
            "action_name": "turn on the Microwave",
            "pre_condition": [["closed", "microwave"]],
            "post_condition": [["heated", "apple_slice"]]
        },
        {
            "action_id": 150,
            "action_name": "turn off the Microwave",
            "pre_condition": [["heated", "apple_slice"]],
            "post_condition": [["microwave", "off"]]
        },
        {
            "action_id": 143,
            "action_name": "open the Microwave",
            "pre_condition": [["microwave", "off"]],
            "post_condition": [["open", "microwave"]]
        },
        {
            "action_id": 107,
            "action_name": "pick up the Apple Slice",
            "pre_condition": [["apple_slice", "in", "microwave"], ["open", "microwave"]],
            "post_condition": [["holding", "apple_slice"]]
        },
        {
            "action_id": 144,
            "action_name": "close the Microwave",
            "pre_condition": [["open", "microwave"]],
            "post_condition": [["closed", "microwave"]]
        },
        {
            "action_id": 18,
            "action_name": "find the DiningTable",
            "pre_condition": [],
            "post_condition": [["focus on", "dining_table"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "apple_slice"], ["focus on", "dining_table"]],
            "post_condition": [["apple_slice", "on", "dining_table"], ["not holding", "apple_slice"]]
        }
    ]
}
''',

'''
Human instruction: slice a loaf of bread put a slice on the counter.
The target state: A sliced piece of bread is placed on the countertop. The knife is no longer in use.
The current state: I am in a kitchen. A loaf of bread sits beside a bread knife on the countertop.
The current scene graph: 
    The objects are: [bread_loaf, bread_knife, countertop]
    The attributes are: [(whole, bread_loaf)]
    The relations are: [(bread_loaf, beside, bread_knife), (bread_loaf, on, countertop), (bread_knife, on, countertop)]
The target state:
    The objects are: [bread_slice, countertop]
    The attributes are: [(not holding, bread_knife), (sliced, bread_slice)]
    The relations are: [(bread_slice, on, countertop)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A loaf of bread sits beside a bread knife on the countertop.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a sliced piece of bread placed on the countertop. I can see a loaf of bread and a knife in the scene, but the bread has not been sliced. Not satisfied.\\n I can also see the countertop, but there is no slice of bread on it. Not satisfied.\\n Therefore, I must use the knife to slice the bread and place one of the slices on the countertop.",
    "language_plan": "First locate a knife and pick it up, find a loaf of bread, and slice the bread. Find a cabinet, open it, and put the knife inside, then close the cabinet. Find a slice of bread, pick it up, find the countertop, and place the slice of bread on the counter.",
    "executable_plan": [
        {
            "action_id": 34,
            "action_name": "find a Knife",
            "pre_condition": [],
            "post_condition": [["focus on", "bread_knife"]]
        },
        {
            "action_id": 127,
            "action_name": "pick up the Knife",
            "pre_condition": [["focus on", "bread_knife"]],
            "post_condition": [["holding", "bread_knife"]]
        },
        {
            "action_id": 53,
            "action_name": "find a Bread",
            "pre_condition": [],
            "post_condition": [["focus on", "bread_loaf"]]
        },
        {
            "action_id": 161,
            "action_name": "slice the Bread",
            "pre_condition": [["focus on", "bread_loaf"], ["holding", "bread_knife"]],
            "post_condition": [["sliced", "bread_slice"], ["on", "bread_slice", "countertop"]]
        },
        {
            "action_id": 48,
            "action_name": "find a Cabinet",
            "pre_condition": [],
            "post_condition": [["focus on", "cabinet"]]
        },
        {
            "action_id": 145,
            "action_name": "open the Cabinet",
            "pre_condition": [["focus on", "cabinet"]],
            "post_condition": [["open", "cabinet"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "bread_knife"], ["open", "cabinet"]],
            "post_condition": [["not holding", "bread_knife"], ["bread_knife", "in", "cabinet"]]
        },
        {
            "action_id": 146,
            "action_name": "close the Cabinet",
            "pre_condition": [["open", "cabinet"]],
            "post_condition": [["closed", "cabinet"]]
        },
        {
            "action_id": 60,
            "action_name": "find a Bread Slice",
            "pre_condition": [],
            "post_condition": [["focus on", "bread_slice"]]
        },
        {
            "action_id": 92,
            "action_name": "pick up the Bread Slice",
            "pre_condition": [["focus on", "bread_slice"]],
            "post_condition": [["holding", "bread_slice"]]
        },
        {
            "action_id": 39,
            "action_name": "find a CounterTop",
            "pre_condition": [],
            "post_condition": [["focus on", "countertop"]]
        },
        {
            "action_id": 133,
            "action_name": "put down the object in hand",
            "pre_condition": [["holding", "bread_slice"], ["focus on", "countertop"]],
            "post_condition": [["bread_slice", "on", "countertop"], ["not holding", "bread_slice"]]
        }
    ]
}
''',

'''
Human instruction: Put a clean pan in the refrigerator.
The target state: A clean pan is placed inside the refrigerator, and the fridge is closed.
The current state: I am in a kitchen. A dirty pan rests in the sink next to a faucet; the fridge is across the room.
The current scene graph: 
    The objects are: [pan, sink, faucet, fridge]
    The attributes are: [(dirty, pan), (closed, fridge)]
    The relations are: [(pan, in, sink), (sink, next to, faucet), (fridge, across_from, sink)]
The target state:
    The objects are: [pan, fridge]
    The attributes are: [(clean, pan), (closed, fridge)]
    The relations are: [(pan, in, fridge)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.
    
Output: {
    "visual_state_description": "A dirty pan rests in the sink next to a faucet; the fridge is across the room.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a clean pan placed inside the refrigerator. I can see a pan, but it is currently dirty and sitting in the sink. Not satisfied.\\n The fridge is visible but closed, and the pan is not inside it. Not satisfied.\\n Therefore, I must wash the pan to make it clean, then place it into the refrigerator and close the door. Notice that: A faucet is the fixture that controls the flow of water, while a sink is the basin or container that holds the water.",
    "language_plan": "Locate a pan and pick it up, find a sink, and put the pan down in the sink. Find a faucet, turn it on to clean the pan, and then turn it off. Pick up the cleaned pan, find the refrigerator, open the refrigerator, place the pan inside, and close the refrigerator.",
    "executable_plan": [
        {"action_id": 7, "action_name": "find a Pan", "pre_condition": [], "post_condition": [["focus on", "pan"]]},
        {"action_id": 86, "action_name": "pick up the Pan", "pre_condition": ["focus on", "pan"],  "post_condition": ["holding", "pan"]},
        {"action_id": 79, "action_name": "find a Sink", "pre_condition": [],  "post_condition": [["focus on", "sink"]]},
        {"action_id": 133, "action_name": "put down the object in hand", "pre_condition": [["focus on", "sink"], ["holding", "pan"]],  "post_condition": [["pan", "in", "sink"]]},
        {"action_id": 2, "action_name": "find a Faucet", "pre_condition": [["pan", "in", "sink"]],  "post_condition": [["focus on", "faucet"], ["pan", "in", "sink"]]},
        {"action_id": 155, "action_name": "turn on the Faucet", "pre_condition": [["pan", "in", "sink"], ["focus on", "faucet"]],  "post_condition": [["water", "in", "faucet"]]},
        {"action_id": 156, "action_name": "turn off the Faucet", "pre_condition": [["pan", "in", "sink"], ["focus on", "faucet"], ["water", "in", "faucet"]],  "post_condition": [["clean", "pan"], ["closed", "faucet"]]},
        {"action_id": 7, "action_name": "find a Pan", "pre_condition": [],  "post_condition": [["focus on", "pan"]]},
        {"action_id": 86, "action_name": "pick up the Pan", "pre_condition": [["focus on", "pan"]],  "post_condition": [["holding", "pan"]]},
        {"action_id": 78, "action_name": "find a Fridge", "pre_condition": [],  "post_condition": [["focus on", "fridge"]]},
        {"action_id": 139, "action_name": "open the Fridge", "pre_condition": [["focus on", "fridge"]],  "post_condition": [["open", "fridge"]]},
        {"action_id": 133, "action_name": "put down the object in hand", "pre_condition": [["holding", "pan"], ["open", "fridge"]],  "post_condition": [["pan", "in", "fridge"]]},
        {"action_id": 140, "action_name": "close the Fridge", "pre_condition": [["open", "fridge"]],  "post_condition": [["closed", "fridge"]]}
    ]
}
''',

'''
Human instruction: Put a box from the couch with a phone in it on the black desk.
The target state: A phone is inside the box, and the box is placed on the black desk.
The current state: I am in a living room. A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.
The current scene graph: 
    The objects are: [cellphone, box, couch, desk]
    The attributes are: [(black, desk), (open, box)]
    The relations are: [(cellphone, beside, box), (box, on, couch), (desk, opposite, couch)]
The target state:
    The objects are: [cellphone, box, desk]
    The attributes are: [(black, desk)]
    The relations are: [(cellphone, in, box), (box, on, desk)]
The explanation: I am on the first step of all planning procedure. There is no history for me to reflect from.

Output: {
    "visual_state_description": "A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.",
    "reflection": "I am on the first step of all planning procedure. There is no action or step order that I need to fix from the previous plan. "
    "reasoning": "I need to check each part in the target state to see whether it is satisfied.\\n First, the phone should be inside the box. Currently, the phone is next to the box on the couch, not inside it. Not satisfied.\\n Second, the box should be on the black desk. Currently, the box is on the couch. Not satisfied.\\n Therefore, I must first place the phone inside the box, then move the box from the couch to the black desk.",
    "language_plan": "First, find the phone, pick it up, locate the box on the couch, place the phone inside the box, pick up the box. Then, use the \"Find\" action to find the black desk, and place the box on the desk.",
    "executable_plan": [
        {"action_id": 12, "action_name": "find a CellPhone", "pre_condition": [],  "post_condition": [["focus on", "cellphone"]]},
        {"action_id": 93, "action_name": "pick up the CellPhone", "pre_condition": ["not holding", _], "post_condition": [["holding", "cellphone"]] },
        {"action_id": 61, "action_name": "find a Box", "pre_condition": [], "post_condition": [["focus on", "box"]] },
        {"action_id": 133, "action_name": "put down the object in hand", "pre_condition": ["holding", "cellphone"], "post_condition": [["not holding", "cellphone"], ["cellphone", "on", "box"]] },
        {"action_id": 103, "action_name": "pick up the Box", "pre_condition": ["not holding", _], "post_condition": [["holding", "box"]] },
        {"action_id": 49, "action_name": "find a Desk", "pre_condition": [], "post_condition": [["focus on", "desk"]] },
        {"action_id": 133, "action_name": "put down the object in hand", "pre_condition": ["focus on", "desk"], "post_condition": [["box", "on", "desk"]] }
    ]
}
''',

]
