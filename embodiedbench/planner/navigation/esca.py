# import torch
import re
import os
import numpy as np
import cv2
import json
# import lmdeploy
# from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from openai import OpenAI
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_target_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url
# from embodiedbench.planner.eb_navigation.RemoteModel_claude import RemoteModel
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.evaluator.config.visual_icl_examples.eb_navigation.ebnav_visual_icl import create_example_json_list
from embodiedbench.planner.planner_utils import laser_template, laser_lang_template
from embodiedbench.main import logger
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import copy

to_tensor = transforms.ToTensor()
template = laser_template
template_lang = laser_lang_template

MESSAGE_WINDOW_LEN = 5

import sys
sys.path.append("/home/jianih/research/GroundingDINO")
from groundingdino.util.inference import Model as gd_Model
from groundingdino.util.inference import load_image as load_gd_image
from groundingdino.util.utils import get_phrases_from_posmap
import torch
import math
from PIL import Image
import numpy as np

model_lib_path = "/home/jianih/research/LASER/LASER-unified/src/models"
sys.path.append(model_lib_path)

from llava_clip_model_v3 import PredicateModel as PredicateModel_v3


def load_model(model_dir, model_name, epoch, device):
    model_name = model_name + f'.{epoch}.model'
    predicate_model = torch.load(os.path.join(model_dir, model_name), map_location=device, weights_only=False)
    return predicate_model

def bbox_to_mask(bbox, frame_height, frame_width):
    """
    Converts a bounding box into a binary mask using torch tensors.

    Parameters:
    - bbox: A list or tuple of bounding box coordinates [xmin, ymin, xmax, ymax].
    - frame_height: The height of the frame.
    - frame_width: The width of the frame.

    Returns:
    - mask: A binary mask of shape (frame_height, frame_width) with dtype torch.bool.
    """
    # Create an empty mask with the same dimensions as the frame
    mask = np.zeros((frame_height, frame_width, 1), dtype=bool)
    
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox
    
    # Ensure coordinates are integers and within frame boundaries
    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(frame_width - 1, xmax))
    ymax = int(min(frame_height - 1, ymax))
    
    # Set the pixels inside the bounding box to True
    mask[ymin:ymax+1, xmin:xmax+1, :] = True
    
    return mask

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
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
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
                 boundingbox = False, 
                 blockingobj = False,
                 scene_graph_text = False,
                 gd_only = False,
                 tp=1, 
                 top_k=1,
                 aggr_thres=0.1,
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
        self.processed_images = {}

        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

        self.multiview = multiview
        self.multistep = multistep
        self.visual_icl = visual_icl
        self.boundingbox = boundingbox
        self.blockingobj = blockingobj
        self.scene_graph_text = scene_graph_text
        self.gd_only = gd_only
        self.aggr_thres = aggr_thres
        self.k = top_k
        self.common_objects = ["wall", "door", "cabinet", "stove", "table", "shadow", "window", "mirror", "other"]

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
            Please extract the most visually identifiable features and ignore the more subtle ones. 
            Please deduce the object names with a synonym if the descriptions is verbose or functional. 
            For example, substitute "freshly baked baguette" to "loaves", "portable device to access the internet" to "laptop", "light-emiting device on my desk to provide light for my work area" to "desk lamp".
            You are supposed to output in JSON.'''
            
        self.first_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.  Note that nested quotes is not allowed. Please use single quote within double quotes properly. {template_lang if self.language_only else template}'''

        self.following_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON. Note that nested quotes is not allowed. Please use single quote within double quotes properly. {template_lang if self.language_only else template}'''

        self.scene_graph_prompt_prefix = "\n Scene Graph Information: The current image height is {w}, width is {h}. A missing scene graph means the object is not clearly identifiable from the scene. It does not necessarily means this object is not in the scene. \n"

        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.grounding_model = gd_Model(
            model_config_path=gd_config_path, 
            model_checkpoint_path=gd_checkpoint_path
        )
    
        self.grounding_model.model = self.grounding_model.model.to("cuda:0")
        
        all_model_dir = "/home/jianih/research/LASER/data/LLaVA-Video-178K-v2/models"
        model_dir = os.path.join(all_model_dir, f"ensemble-02-10")
        model_name = "ensemble-2025-02-10-14-57-22"
        epoch = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predicate_model = load_model(model_dir, model_name, epoch, device)

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
        
        failure = False
        blocking_feedback = ""
        
        if not len(prev_act_feedback) == 0:
            last_feedback = prev_act_feedback[-1]
            if last_feedback[0] != 0:
                failure = True
                blocking_feedback = last_feedback[1]
            
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
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## The human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])

            if failure:
                scene_graph_prompt += f"The last action is invalid, and the feedback is: {blocking_feedback} "
            else:
                scene_graph_prompt += f"The last action succeeds. "
                
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
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
            
            if failure:
                scene_graph_prompt += f"The last action is invalid, and the feedback is: {blocking_feedback} "
            else:
                scene_graph_prompt += f"The last action succeeds. "
                
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
        
        if len(classes) == 0:
            return [], []
        
        caption = ". ".join(classes)
        processed_caption = preprocess_caption(caption)
        processed_image = self.grounding_model.preprocess_image(image_bgr=image).to(self.grounding_model.device)

        with torch.no_grad():
            outputs = self.grounding_model.model(processed_image.unsqueeze(0), captions=processed_caption)

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (batch_size, nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (batch_size, nq, 4)

        detection_ls = []

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
        detection_ls.append(detections)

        assert len(detection_ls) == 1
        detections = detection_ls[0]
        
        reformated = torch.tensor([(get_valid_bbox_val(x1, source_w), 
                    get_valid_bbox_val(y1, source_h),
                    get_valid_bbox_val(x2, source_w), 
                    get_valid_bbox_val(y2, source_h))
                    for x1, y1, x2, y2 in detections.xyxy])
        
        boxes = []
        labels = []
        
        for ct, class_id in enumerate(detections.class_id):
            if ct == 0 and class_id is None:
                class_id = 0
                
            elif class_id is None:
                continue
            
            x1, y1, x2, y2 = reformated[ct]
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            
            boxes.append(reformated[ct])
            labels.append(classes[class_id])
            
        if len(boxes) == 0:
            boxes = torch.tensor([])
        else:
            boxes = torch.stack(boxes)
                
        return labels, boxes
    
    def predict_laser(self, 
                      image, 
                      boxes, 
                      target_names,
                      query_names,
                      query_attributes,
                      ):
        
        # Preprocess query attributes for the case where one object having multiple attribute
        query_attribute_ls = [[attr, f"not {attr}"] for attr in query_attributes]
        prob_labels, selected_object_bboxes = [], []
        
        if not len(boxes) == 0:
            height, width = image.shape[0], image.shape[1]
            batched_pred_masks = [bbox_to_mask(bbox, height, width) for bbox in boxes]
            batched_object_ids = [[0, 0, i] for i in range(len(batched_pred_masks))]

            batched_image_cate_probs, _, _, _ = \
                self.predicate_model(
                    batched_video_ids=[0],
                    batched_videos=[image],
                    batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
                    batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
                    batched_names=[query_names],  # Dataset-wise categorical labels
                    batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
                    batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
                    batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
                    batched_obj_pairs=[],  # Ground truth binary relations
                    batched_video_splits=[0],  # [number of videos]
                    batched_binary_predicates=[None],  # None indicates inference time
                )
                    
            all_unary_res = []
            for unary in query_attribute_ls:
                _, batched_image_unary_probs, _, _ = \
                    self.predicate_model(
                        batched_video_ids=[0],
                        batched_videos=[image],
                        batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
                        batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
                        batched_names=[query_names],  # Dataset-wise categorical labels
                        batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
                        batched_unary_kws=[unary],  # Dataset-wise unary predicate labels
                        batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
                        batched_obj_pairs=[],  # Ground truth binary relations
                        batched_video_splits=[0],  # [number of videos]
                        batched_binary_predicates=[None],  # None indicates inference time
                    )
                all_unary_res.append(batched_image_unary_probs)
            
            # Combine the uanry and object name information together 
            # Step 1: reformat the object name and attributes into better format
            predicted_name = {}
            for (oid, name) in batched_image_cate_probs[0]:
                prob = batched_image_cate_probs[0][(oid, name)].item()
                if not oid in predicted_name:
                    predicted_name[oid] = []
                predicted_name[oid].append((prob, name))
            
            for oid in predicted_name:
                predicted_name[oid] = sorted(predicted_name[oid], reverse=True)   
                
            reformatted_unary_res = {}
            for unary_res in all_unary_res:
                for  (fid, oid, kw) in unary_res[0].keys():
                    if not oid in reformatted_unary_res:
                        reformatted_unary_res[oid] = {}
                    if kw in query_attributes:
                        reformatted_unary_res[oid][kw] = unary_res[0][(fid, oid, kw)].item()
            
            # Step 2: Aggregate the object name and attribute predictions together
            aggr_res = {}            
            
            for oid in predicted_name:
                top_prob, name = predicted_name[oid][0]
                if not oid in aggr_res:
                    aggr_res[oid] = []
                
                if name in target_names:
                    aggr_res[oid].append(top_prob)
                else:
                    # the top predicted name is not the target name
                    aggr_res[oid].append(0)
                    
            for oid in reformatted_unary_res: 
                
                if not oid in aggr_res:
                    aggr_res[oid] = []
                    
                for attr in reformatted_unary_res[oid]:
                    aggr_res[oid].append(reformatted_unary_res[oid][attr])
            
            # Thresholding
            avg_aggr_res = {}
            for oid in aggr_res:
                probs = aggr_res[oid]
                avg_val = sum(probs) / len(probs)
                if avg_val > self.aggr_thres:
                    avg_aggr_res[oid] = avg_val
            
            # Only pick the top-k object according to the setip
            top_objects = sorted([(v, k) for k, v in avg_aggr_res.items()], reverse=True)[:self.k]
            top_objects_probs = {oid: prob for prob, oid in top_objects}
            
            selected_objects = list(top_objects_probs.keys())
            selected_object_bboxes = boxes[selected_objects]
                                
            prob_labels = []
            for oid in top_objects_probs:
                prob_labels.append(predicted_name[oid][0])

        return prob_labels, selected_object_bboxes
    
    def get_sg_kws(self, sg_info):
        
        target_names = sg_info['target_name']
        target_attributes = sg_info['target_attribute']   
        related_objects = set(sg_info['related_objects'])
        
        target_relations = sg_info['target_relation']
        relations = set()

        for rel in target_relations:
            if not len(rel) == 3:
                continue
            (s, r, o) = rel
            related_objects.add(s)
            related_objects.add(o)
            relations.add(r)
            
        related_objects = list(related_objects)
        relations = list(relations)
        
        if self.blockingobj:
            blocking_objects = sg_info['blocking_name']
            query_objects = blocking_objects + related_objects + target_names + self.common_objects 
            gd_query_objects = blocking_objects + target_names
        else:
            query_objects = related_objects + target_names + self.common_objects 
            gd_query_objects = target_names
            
        return target_names, target_attributes, query_objects, gd_query_objects
    
    def act(self, observation, user_instruction):
        
        if type(observation) == dict:
            obs = copy.deepcopy(observation[self.obs_key])
        else:
            obs = copy.deepcopy(observation) # input image path
        
        # Query the VLM for information for generating scene graph.
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
        
        sg_info = json.loads(out)
        target_names, target_attributes, query_objects, gd_query_objects = self.get_sg_kws(sg_info)

        # Prep for scene graph infused prompt
        scene_graph_prompt = []

        if isinstance(obs, str):
            obs = [obs]
        
        if isinstance(observation, str):
            observation = [observation]
        
        # For each queried image, perform scene graph inference
        for image_id, image_path in enumerate(observation):
            
            image_path_ls = image_path.split('/')
            image_dir, image_name = image_path_ls[:-1], image_path_ls[-1]
            image_name = image_name.split('.')[0]
            bbox_file_name = image_name + '_bbox.png'
            bbox_path = os.path.join(*image_dir, bbox_file_name)
            prompt_image_path = image_path
            
            if image_path in self.processed_images:
                processed_image_path = self.processed_images[image_path]
                if not processed_image_path in obs:
                    obs.append(processed_image_path)
                    continue
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[0], image.shape[1]
            

            if self.gd_only:
                
                # Predict bounding box with grounding dino
                target = ' '.join(target_attributes) + ' ' + ' '.join(target_names)
                labels, boxes = self.predict_gd(image, target, box_threshold=0.2, text_threshold=0.1)
            
                target_labels, target_bboxes = [], []
                for label, box in zip(labels, boxes):
                    if not label is None:
                        target_labels.append(label)
                        target_bboxes.append(box)
                        
                if len(target_bboxes) > 0:
                    target_bboxes = torch.stack(target_bboxes)
            
            else:
                labels, boxes = self.predict_gd(image, gd_query_objects, box_threshold=0.1, text_threshold=0.1)

                target_labels, target_bboxes = self.predict_laser(
                      image=image, 
                      boxes=boxes, 
                      target_names=target_names,
                      query_names=query_objects,
                      query_attributes=target_attributes,
                      )
                
            if self.boundingbox:
                if not len(target_bboxes) == 0:
                    save_prompts_one_image(frame_image=image, 
                                            boxes=target_bboxes, 
                                            labels=["" for _ in target_bboxes], 
                                            points=[], 
                                            save_path=bbox_path)    
                    prompt_image_path = bbox_path  
                
            else:
                target_bboxes = []
                target_labels = []
            
            self.processed_images[image_path] = prompt_image_path
            
            if type(target_bboxes) == torch.Tensor:
                bbox_list = target_bboxes.tolist()
            else:
                bbox_list = target_bboxes
            
            # Arrange the prompt into vlm
            if self.scene_graph_text and len(target_labels) > 0:
                current_scene_graph_prompt = ",".join([f"{obj_name}: {obj_box}" for (obj_name, obj_box) in zip(target_labels, bbox_list)])
                scene_graph_prompt.append(f"Its scene graph at frame {image_id} is: {current_scene_graph_prompt}. ")
        
            if self.multiview or self.multistep:
                if not image_path == prompt_image_path:
                    obs.append(prompt_image_path) # update to bboxed image path
            else:
                obs = [prompt_image_path]
                
        # Wrap the scene graph prompt with prefix and add it to the prompt
        if self.scene_graph_text and len(scene_graph_prompt) > 0:
            prompt += self.scene_graph_prompt_prefix.format(w=width, h=height) + "\n".join(scene_graph_prompt)
            
        obs = sorted(list(set(obs)))
        
        if not self.multiview and not self.multistep:
            assert len(obs) == 1
            obs = obs[0]
                
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
        
        if valid:
            plan = json.loads(out)
            plan['scene_graph'] = sg_info
            out_str = json.dumps(plan)
        else:
            out_str = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
                   
        self.planner_steps += 1
        return action, out_str

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