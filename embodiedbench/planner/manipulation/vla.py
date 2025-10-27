import os.path as osp
import numpy as np
import cv2
import base64
import json
import ast
import random
import time
import logging
from mimetypes import guess_type
from embodiedbench.envs.eb_manipulation.eb_man_utils import ROTATION_RESOLUTION, VOXEL_SIZE
from embodiedbench.planner.vla_model import VLAModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.planner.planner_utils import local_image_to_data_url, template_manip, template_lang_manip
from embodiedbench.main import logger

VISUAL_ICL_EXAMPLES_PATH = "embodiedbench/evaluator/config/visual_icl_examples/eb_manipulation"
VISUAL_ICL_EXAMPLE_CATEGORY = {
    "pick": "pick_cube_shape",
    "place": "place_into_shape_sorter_color",
    "stack": "stack_cubes_color",
    "wipe": "wipe_table_direction"
}


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def object_to_dict(obj):
    return {
        key: getattr(obj, key)
        for key in dir(obj)
        if not key.startswith('_') and not callable(getattr(obj, key))
    }

class ManipPlannerVLA():
    def __init__(self, model_name, model_type, system_prompt, examples, n_shot=0, obs_key='front_rgb', chat_history=False, language_only=False, multiview=False, multistep=False, visual_icl=False, tp=1, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to include all the chat history for prompting
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        elif model_type == 'remote':  
            self.model = VLAModel(model_name, model_type, language_only, task_type='manip')

        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.multi_view = multiview
        self.multi_step_image = multistep
        self.visual_icl = visual_icl
        self.episode_act_feedback = []
        
        self.sleep_time_local=20
        self.sleep_time_remote=60
        self.max_retries = 5
    
    def process_prompt(self, user_instruction, avg_obj_coord, task_variation, obs, prev_act_feedback=[]):
        
        assert len(obs["joint_positions"]) == 7
        assert len([obs["gripper_open"]]) == 1
        
        return {
            "observation/exterior_image_1_left": obs["left_shoulder_rgb"].astype(np.uint8),
            "observation/exterior_image_2_left": obs["front_rgb"].astype(np.uint8),
            "observation/wrist_image_left": obs["wrist_rgb"].astype(np.uint8),
            "observation/joint_position": obs["joint_positions"].astype(np.float32),
            "observation/gripper_position": np.array([obs["gripper_open"]]),
            "prompt": user_instruction,
        }
    
    #     user_instruction = user_instruction.rstrip('.')
    #     if len(prev_act_feedback) == 0:
    #         if self.n_shot >= 1:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '\n'.join([f'Example {i}: \n{x}' for i, x in enumerate(self.examples[task_variation][:self.n_shot])])) 
    #         else:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
    #         task_prompt = f"\n## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
    #     elif self.chat_history:
    #         general_prompt = f'The human instruction is: {user_instruction}.'
    #         general_prompt += '\n\n The gripper action history:'
    #         for i, action_feedback in enumerate(prev_act_feedback):
    #             general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
    #         task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
    #     else:
    #         if self.n_shot >= 1:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '\n'.join([f'Example {i}: \n{x}' for i, x in enumerate(self.examples[task_variation][:self.n_shot])])) 
    #         else:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
    #         task_prompt = f"\n## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
    #         for i, action_feedback in enumerate(prev_act_feedback):
    #             task_prompt += f"{action_feedback}, "
    #     return general_prompt, task_prompt

    # def process_prompt_visual_icl(self, user_instruction, avg_obj_coord, prev_act_feedback=[]):
    #     user_instruction = user_instruction.rstrip('.')
    #     if len(prev_act_feedback) == 0:
    #         if self.n_shot >= 1:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '') 
    #         else:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
    #         task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
    #     elif self.chat_history:
    #         general_prompt = f'The human instruction is: {user_instruction}.'
    #         general_prompt += '\n\n The gripper action history:'
    #         for i, action_feedback in enumerate(prev_act_feedback):
    #             general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
    #         task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
    #     else:
    #         if self.n_shot >= 1:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '') 
    #         else:
    #             general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
    #         task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
    #         for i, action_feedback in enumerate(prev_act_feedback):
    #             task_prompt += f"{action_feedback}, "
    #     return general_prompt, task_prompt
    
    # def get_message(self, images, prompt, task_prompt, messages=[]):
    #     if self.language_only and not self.visual_icl:
    #         return messages + [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": prompt + task_prompt}],
    #             }
    #         ]
    #     else:
    #         if self.multi_step_image:
    #             current_message = [
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {"type": "text", "text": prompt}],
    #                 }
    #             ]

    #             # use the last three imags as multi-step images
    #             if len(images) >= 3:
    #                 multi_step_images = images[-3:-1]
    #                 current_message[0]["content"].append(  
    #                     {
    #                         "type": "text",
    #                         "text": "You are given the scene observations from the last two steps:",
    #                     }
    #                 )
    #                 for image in multi_step_images:
    #                     if type(image) == str:
    #                         image_path = image 
    #                     else:
    #                         image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
    #                         cv2.imwrite(image_path, image)
    #                     data_url = local_image_to_data_url(image_path=image_path)
    #                     current_message[0]["content"].append(
    #                         {
    #                             "type": "image_url",
    #                             "image_url": {
    #                                 "url": data_url,
    #                             }
    #                         }
    #                     )
                
    #                 # add the current task prompt and input image
    #                 current_message[0]["content"].append(
    #                     {
    #                         "type": "text",
    #                         "text": task_prompt,
    #                     }
    #                 )

    #                 # add the current step image
    #                 current_step_image = images[-1]
    #                 if type(current_step_image) == str:
    #                     image_path = current_step_image 
    #                 else:
    #                     image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
    #                     cv2.imwrite(image_path, current_step_image)
    #                 data_url = local_image_to_data_url(image_path=image_path)
    #                 current_message[0]["content"].append(
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": data_url,
    #                         }
    #                     }
    #                 )
    #             else:
    #                 full_prompt = prompt + task_prompt
    #                 current_message = [
    #                     {
    #                         "role": "user",
    #                         "content": [
    #                             {"type": "text", "text": full_prompt}],
    #                     }
    #                 ]

    #                 for image in images:
    #                     if type(image) == str:
    #                         image_path = image 
    #                     else:
    #                         image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
    #                         cv2.imwrite(image_path, image)

    #                     data_url = local_image_to_data_url(image_path=image_path)
    #                     current_message[0]["content"].append(
    #                         {
    #                             "type": "image_url",
    #                             "image_url": {
    #                                 "url": data_url,
    #                             }
    #                         }
    #                     )

    #         else:
    #             full_prompt = prompt + task_prompt
    #             current_message = [
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {"type": "text", "text": full_prompt}],
    #                 }
    #             ]

    #             for image in images:
    #                 if type(image) == str:
    #                     image_path = image 
    #                 else:
    #                     image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
    #                     cv2.imwrite(image_path, image)

    #                 data_url = local_image_to_data_url(image_path=image_path)
    #                 current_message[0]["content"].append(
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": data_url,
    #                         }
    #                     }
    #                 )
        
    #         return current_message
    
    # def get_message_visual_icl(self, images, first_prompt, task_prompt, task_variation, messages=[]):
    #     current_message = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": first_prompt}
    #             ],
    #         }
    #     ]
    #     visual_task_variation = VISUAL_ICL_EXAMPLE_CATEGORY[task_variation.split('_')[0]]
    #     task_specific_image_example_path = osp.join(VISUAL_ICL_EXAMPLES_PATH, visual_task_variation)
    #     icl_text_examples = self.examples[task_variation]
    #     stop_idx = 2
    #     for example_idx, example in enumerate(icl_text_examples):
    #         if example_idx >= stop_idx:
    #             break
    #         current_image_example_path = osp.join(task_specific_image_example_path, f"episode_{example_idx+1}_step_0_front_rgb_annotated.png")
    #         example = "Example {}:\n{}".format(example_idx+1, example)
    #         data_url = local_image_to_data_url(image_path=current_image_example_path)

    #         # Add the example image and the corresponding text to the message
    #         current_message[0]["content"].append(
    #             {
    #                 "type": "text",
    #                 "text": example,
    #             }
    #         )
    #         current_message[0]["content"].append(  
    #             {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": data_url,
    #                 }
    #             }
    #         )
    #     # add the task prompt
    #     current_message[0]["content"].append(
    #         {
    #             "type": "text",
    #             "text": task_prompt,
    #         }
    #     )

    #     for image in images:
    #         if type(image) == str:
    #             image_path = image 
    #         else:
    #             image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
    #             cv2.imwrite(image_path, image)

    #         data_url = local_image_to_data_url(image_path=image_path)
    #         current_message[0]["content"].append(
    #             {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": data_url,
    #                 }
    #             }
    #         )
    #     return current_message
    
    # def json_to_action(self, output_text):
    #     try:
    #         json_object = json.loads(output_text)
    #         action = []
    #         try:
    #             executable_plan = json_object['executable_plan'] if 'executable_plan' in json_object else json_object["properties"]["executable_plan"]
    #         except:
    #             print("Failed to get executable plan from json object")
    #             print('random action')
    #             self.output_json_error += 1
    #             arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
    #             gripper = [1.0]  # Always open
    #             action = arm + gripper
    #             return [action], None
    #         if type(executable_plan) == str:
    #             try:
    #                 executable_plan = ast.literal_eval(executable_plan)
    #             except Exception as e:
    #                 print("Failed to decode string executable plan to list executable plan:", e)
    #                 print('random action')
    #                 self.output_json_error += 1
    #                 arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
    #                 gripper = [1.0]  # Always open
    #                 action = arm + gripper
    #                 return [action], None
    #         if len(executable_plan) > 0:
    #             for x in executable_plan:
    #                 if type(x) == tuple:
    #                     x = list(x)
    #                 if 'action' in x:
    #                     list_action = x['action']
    #                 else:
    #                     if type(x) == list and type(x[0]) == int:
    #                         list_action = x
    #                     elif 'action' in x[0]:
    #                         list_action = x[0]["action"]
    #                     else:
    #                         list_action = x
    #                 if type(list_action) == str:
    #                     try:
    #                         list_action = ast.literal_eval(x['action'])
    #                     except Exception as e:
    #                         print("Failed to decode string action to list action:", e)
    #                         print('random action')
    #                         action = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)] + [1.0]
    #                         self.output_json_error += 1
    #                         return [action], None
    #                 action.append(list_action)
    #             return action, json_object
    #         else:
    #             print("Empty executable plan, quit the episode ...")
    #             self.output_json_error = -1
    #             return [], output_text
    #     except json.JSONDecodeError as e:
    #         print("Failed to decode JSON:", e)
    #         print('random action')
    #         self.output_json_error += 1
    #         arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
    #         gripper = [1.0]  # Always open
    #         action = arm + gripper
    #         return [action], None
    #     except Exception as e:
    #         print("An expected error occurred:", e)
    #         print('random action')
    #         self.output_json_error += 1
    #         arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
    #         gripper = [1.0]  # Always open
    #         action = arm + gripper
    #         return [action], None
    
    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    # def act_custom(self, prompt, obs):
    #     assert type(obs) == str # input image path
    #     out = self.model.respond(prompt, obs)
    #     out = out.replace("'",'"')
    #     out = out.replace('\"s ', "\'s ")
    #     out = out.replace('```json', '').replace('```', '')
    #     logger.debug(f"Model Output:\n{out}\n")
    #     action, _ = self.json_to_action(out)
    #     self.planner_steps += 1
    #     return action, out
    
    def act(self, observation, user_instruction, avg_obj_coord, task_variation):
        
        if type(observation) == dict:
            obs = observation
        else:
            obs = object_to_dict(observation)
            # obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, avg_obj_coord, task_variation, obs, prev_act_feedback=[])
        
        if self.model_type == 'custom':
            return self.act_custom(prompt, obs[0]) 
        else:
            attempt = 0
            while attempt < self.max_retries:
                try:
                    out = self.model.respond(prompt, obs)
                    break
                except:
                    attempt += 1
                    if self.model_type != 'local':
                        time.sleep(60)
                    else:
                        time.sleep(20)
                    out = self.model.respond(prompt, obs)
                # try:
                #     out = self.model.respond(prompt, obs)
                #     break
                # except Exception as e:
                #     attempt += 1
                #     if attempt >= self.max_retries:
                #         raise RuntimeError(f"Model failed after {self.max_retries} attempts") from e
                    
                #     wait_time = self.sleep_time_remote if self.model_type != 'local' else self.sleep_time_local
                #     print(f"[Retry {attempt}/{self.max_retries}] Error: {e}. Retrying after {wait_time}s...")
                #     time.sleep(wait_time)
                    
        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        
        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        action = out
        # action, json_output = self.json_to_action(out)
        return action, out

    def update_info(self, info):
        env_feedback = info['env_feedback']
        action = info['action']
        self.episode_act_feedback.append([action, env_feedback])

    