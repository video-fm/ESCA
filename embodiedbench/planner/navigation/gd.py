# import torch
import re
import os
import numpy as np
import cv2
import json
# import lmdeploy
# from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from openai import OpenAI
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url
# from embodiedbench.planner.eb_navigation.RemoteModel_claude import RemoteModel
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.evaluator.config.visual_icl_examples.eb_navigation.ebnav_visual_icl import create_example_json_list
from embodiedbench.planner.planner_utils import template, template_lang
from embodiedbench.main import logger
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import copy

to_tensor = transforms.ToTensor()
template = template
template_lang = template_lang

MESSAGE_WINDOW_LEN = 5

import sys
sys.path.append("/home/jianih/research/GroundingDINO")
from groundingdino.util.inference import Model as gd_Model
from groundingdino.util.inference import load_image as load_gd_image
from groundingdino.util.utils import get_phrases_from_posmap
import torch
import math
from PIL import Image

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    return result if result.endswith(".") else result + "."

def get_valid_bbox_val(v, b):
    return min(max(0, math.floor(v)), b)

def post_process_detections(det, classes):
    detections = det[0]
    class_ids = [0 if class_id is None else class_id for class_id in detections.class_id]
    det.class_names = [classes[class_id] for class_id in class_ids]
    detections.xyxy = np.concatenate((detections.xyxy, det.xyxy), axis=0)
    detections.class_names = detections.class_names + det.class_names
    return detections

def list_depth(lst):
    """Calculates the depth of a nested list."""

    if not (isinstance(lst, list) or isinstance(lst, torch.Tensor)):
        return 0
    elif (isinstance(lst, torch.Tensor) and lst.shape == torch.Size([])) or (isinstance(lst, list) and len(lst) == 0):
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)
    
def normalize_prompt(points, labels):
    if list_depth(points) == 3: 
        points = torch.stack([p.unsqueeze(0) for p in points])
        labels = torch.stack([l.unsqueeze(0) for l in labels])
    return points, labels

# modify this function to take in a bbox label and show it
def show_box(box, ax, object_id, label='hi'):
    if len(box) == 0:
        return
    
    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=0.5))
    
    if label is not None:
        # ax.text(x0, y0 - 5, label, color='white', fontsize=10, backgroundcolor=np.array(color[:3]), alpha=color[3])
        ax.text(x0, y0 - 5, label, color='white', fontsize=10, alpha=color[3])
    
def save_prompts_one_image(frame_image, boxes, points, labels, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    points, labels = normalize_prompt(points, labels)
    if type(boxes) == torch.Tensor:
        for object_id, (box, label) in enumerate(zip(boxes, labels)):
            if box is not None:
                show_box(box.cpu(), ax, object_id=object_id, label=label)
    elif type(boxes) == dict:
        for object_id, box in boxes.items():
            if box is not None:
                show_box(box.cpu(), ax, object_id=object_id)
    elif type(boxes) == list and len(boxes) == 0:
        pass
    else:
        raise Exception()
    
    for object_id, (point_ls, label_ls) in enumerate(zip(points, labels)):
        if not len(point_ls) == 0:
            show_points(point_ls.cpu(), label_ls.cpu(), ax, object_id=object_id)
        
    # Show the plot
    plt.savefig(save_path)
    plt.close()
    
    
class EBNavigationPlanner():
    def __init__(self, model_name = '', 
                 model_type = 'remote', 
                 actions = [], 
                 system_prompt = '', 
                 examples = '', 
                 scene_graph_examples = '',
                 n_shot=1, 
                 obs_key='head_rgb', 
                 chat_history=False, 
                 language_only=False, 
                 multiview = False, 
                 multistep = False, 
                 visual_icl = False, 
                 tp=1, 
                 gd_config_path='',
                 gd_checkpoint_path='',
                 kwargs={}):
        
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.set_actions(actions)
        self.planner_steps = 0
        self.output_json_error = 0

        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

        self.multiview = multiview
        self.multistep = multistep
        self.visual_icl = visual_icl

        if not self.visual_icl:
            self.examples = examples[:n_shot]
            self.scene_graph_examples = scene_graph_examples[:n_shot]
            self.language_only = language_only
        else:
            self.examples = []
            self.language_only = False
            if language_only:
                self.icl_text_only = True
            else:
                self.icl_text_only = False

        self.scene_graph_prompt = f'''## You are a robot operating in a home. You are given an image and observe the environment for performing the given task. \
            The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'}\
            You are supposed to output in JSON.'''
            
        self.first_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        self.following_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.grounding_model = gd_Model(
            model_config_path=gd_config_path, 
            model_checkpoint_path=gd_checkpoint_path
        )
    
        self.grounding_model.model = self.grounding_model.model.to("cuda:0")
        
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def process_prompt(self, user_instruction, prev_act_feedback=[]):

        user_instruction = user_instruction.rstrip('.')

        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += self.first_prompt
     
        elif self.chat_history:

            # This is to support the sliding window feature
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## The human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])

            prompt += f"\n\n{self.following_prompt}"

        else:
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
            
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += f"\n\n{self.following_prompt}"

        return prompt, scene_graph_prompt
    

    def get_message(self, image, prompt, messages=[]):

        if self.language_only:
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}],
            }
        elif self.multiview:
            data_url1 = local_image_to_data_url(image_path=image[0])
            data_url2 = local_image_to_data_url(image_path=image[1])
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url1,
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url2,
                        }
                    },
                    {"type": "text", "text": prompt}],
            }
        elif self.multistep:
            content = []
            for img_path in image:
                data_url = local_image_to_data_url(image_path=img_path)
                content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }) 
            content.append({"type": "text", "text": prompt})
            current_message = {
                "role":"user",
                "content":content
            }
        elif self.visual_icl:
            content = []
            content.append({"type": "text", "text": prompt})
            visual_example = create_example_json_list((not self.icl_text_only))
            content.extend(visual_example)
            content.append({"type": "text", "text": "Below is your current step observation, please starting planning to navigate to the target object by learning from the above-mentioned strategy and in-context learning examples. ### Output nothing else but a JSON string following the above mentioned format ###"})
            data_url = local_image_to_data_url(image_path=image)
            content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }) 
            current_message = {
                "role":"user",
                "content":content
            }
        else:
            data_url = local_image_to_data_url(image_path=image)
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }, 
                    {"type": "text", "text": prompt}],
            }

        messages = messages + [current_message]

        return messages[-MESSAGE_WINDOW_LEN:]


    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        valid = True
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print('empty plan, using random action instead')
                action = np.random.randint(len(self.actions))
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print('random action')
            self.output_json_error += 1
            action = np.random.randint(len(self.actions))
            valid = False
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            print('Using random action due to an unexpected error')
            action = np.random.randint(len(self.actions))
            valid = False
        return action, valid

        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        out = out.replace("'",'"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        logger.debug(f"Model Output:\n{out}\n")
        action, _ = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    
    def predict_gd(self, image, classes, box_threshold=0.5, text_threshold=0.4):
        caption = ". ".join(classes)
        processed_caption = preprocess_caption(caption)
        processed_image = self.grounding_model.preprocess_image(image_bgr=image).to(self.grounding_model.device)

        with torch.no_grad():
            outputs = self.grounding_model.model(processed_image.unsqueeze(0), captions=processed_caption)

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (batch_size, nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (batch_size, nq, 4)

        all_detections = []

        tokenizer = self.grounding_model.model.tokenizer

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # (n, 256)
        boxes = prediction_boxes[mask]  # (n, 4)

        tokenized = tokenizer(caption)

    
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]

        logits = logits.max(dim=1)[0]

        source_h, source_w, _ = image.shape
        detections = self.grounding_model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits
        )
        class_id = self.grounding_model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        all_detections.append(detections)

        return all_detections
    
    
    def act(self, observation, user_instruction):
        
        if type(observation) == dict:
            obs = copy.deepcopy(observation[self.obs_key])
        else:
            # assert len(observation) == 1
            obs = copy.deepcopy(observation) # input image path
        
        prompt, scene_graph_prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        
        if self.model_type == 'custom':
            return self.act_custom(prompt, obs)

        sg_message = self.get_message(obs, scene_graph_prompt)
        
        try:
            out = self.model.respond(sg_message, get_scene_graph=True)
        except Exception as e:
            print(e)
            if 'qwen' in self.model_name:
                return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
        
        sg_keywords = json.loads(out)['scene_graph_entities']
                
        prompt = prompt + f'\n\n The current image height is 600, width is 600.'

        # detections = post_process_detections(all_dets, sg_keywords)
        for ct, image_path in enumerate(observation):
            
            image_path_ls = image_path.split('/')
            image_dir, image_name = image_path_ls[:-1], image_path_ls[-1]
            image_name = image_name.split('.')[0]
            new_image_name = image_name + '_bbox.png'
            new_image_path = os.path.join(*image_dir, new_image_name)
            
            if os.path.exists(new_image_path):
                obs.append(new_image_path)
                continue
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            height, width = image.shape[0], image.shape[1]
            detection_ls = self.predict_gd(image, sg_keywords, box_threshold=0.4, text_threshold=0.4)
            assert len(detection_ls) == 1
            detections = detection_ls[0]
            
            scene_graph_text = []
            reformated = torch.tensor([(get_valid_bbox_val(x1, width), 
                        get_valid_bbox_val(y1, height),
                        get_valid_bbox_val(x2, width), 
                        get_valid_bbox_val(y2, height))
                        for x1, y1, x2, y2 in detections.xyxy])
            
            
            boxes = []
            labels = []
            
            for ct, class_id in enumerate(detections.class_id):
                if class_id is None:
                    continue
                boxes.append(reformated[ct])
                labels.append(sg_keywords[class_id])
            
            if len(boxes) == 0:
                boxes = torch.tensor([])
            else:
                boxes = torch.stack(boxes)
                
            save_prompts_one_image(frame_image=image, boxes=boxes, labels=labels, points=[], save_path=new_image_path)        
        
            if not image_path in obs:
                obs.append(image_path)
            obs.append(new_image_path) # update to bboxed image path
        
            scene_graph_text += [f"{obj_name}: {obj_box}" for (obj_name, obj_box) in zip(labels, boxes)]
            prompt += f"Its scene graph at frame {ct} is:  {scene_graph_text}."
        
        obs = sorted(obs)
        if len(self.episode_messages) == 0:
            self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        try:
            out = self.model.respond(self.episode_messages)
        except Exception as e:
            print(e)
            if 'qwen' in self.model_name:
                return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
            
        logger.debug(f"Model Output:\n{out}\n")
        action, valid = self.json_to_action(out)
        self.planner_steps += 1
        if valid:
            return action, out
        else:
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out

    def act_video(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        if self.model_type == 'custom':
            return self.act_custom(prompt, obs)

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        try:
            out = self.model.respond(self.episode_messages)
        except Exception as e:
            print(e)
            if 'qwen' in self.model_name:
                return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
            
        logger.debug(f"Model Output:\n{out}\n")
        action, valid = self.json_to_action(out)
        self.planner_steps += 1
        if valid:
            return action, out
        else:
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out


    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])