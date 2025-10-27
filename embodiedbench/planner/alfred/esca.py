import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import (
    llm_generation_guide,
    vlm_target_generation_guide,
)
from embodiedbench.planner.planner_utils import (
    local_image_to_data_url,
    laser_template,
    laser_lang_template,
    fix_json,
)
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger
from PIL import Image

import copy
import matplotlib.pyplot as plt

# SAM2 imports
import sys
sam2_path = "/home/asethi04/sam2"
if sam2_path not in sys.path:
    sys.path.append(sam2_path)
    
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError as e:
    print(f"SAM2 not available: {e}")
    SAM2_AVAILABLE = False


template = laser_template

template_lang = laser_lang_template

MESSAGE_WINDOW_LEN = 5

import sys

# Use the pip-installed groundingdino-py package
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDINGDINO_AVAILABLE = True
    print("Using pip-installed groundingdino-py")
except ImportError as e:
    print(f"groundingdino-py not available: {e}")
    GROUNDINGDINO_AVAILABLE = False

import torch
import math

import numpy as np


model_lib_path = "/home/asethi04/LASER/LASER-unified/src/models"
sys.path.append(model_lib_path)

from llava_clip_model_v3 import PredicateModel as PredicateModel_v3

from collections import defaultdict

import json, textwrap, re, yaml          # pip install pyyaml
def robust_load_json_full(raw: str):
    """
    Heuristically clean up a model reply and return a Python object.
    Works in this order:
      1. direct ``json.loads``            (fast-path)
      2. json-after-fixups               (escape badly placed quotes)
      3. yaml.safe_load (very forgiving)
    Raises the last JSON error if every attempt fails.
    """

    # --- 0) quick path -------------------------------------------------
    try:
        return json.loads(raw)
    except Exception as strict_err:
        pass                      # fallthrough to the repair pipeline

    txt = raw.translate(str.maketrans({
        "“": '"', "”": '"',       # fancy → ASCII quotes
        "‘": "'", "’": "'"
    }))

    # -- 1) stray quotes OUTSIDE strings → replace by apostrophe --------
    txt = re.sub(r'(?<![\\"])("(?![\s]*[,:}\]]))', "'", txt)

    # -- 2) ':'  'value'   →   ":" "value"
    txt = re.sub(r'(":[ \t]*)\'', r'\1"', txt)

    # -- 3) escape single quotes inside strings ------------------------
    def _esc_single(m):
        return re.sub(r"(?<!\\)'", r"\\'", m.group(0))
    txt = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _esc_single, txt)

    # -- 4) **escape double quotes that sit INSIDE a double-quoted value**
    def _esc_double(m):
        inner = m.group(1)
        inner_fixed = re.sub(r'(?<!\\)"', r'\\"', inner)
        return f'"{inner_fixed}"'
    txt = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _esc_double, txt)

    # -- 5) remove control characters ----------------------------------
    txt = re.sub(r"[\x00-\x1F]", "", txt)

    # ---------- try JSON again ----------------------------------------
    try:
        return json.loads(txt)
    except Exception as json_err:
        last_err = json_err

    # ---------- final fallback: YAML ----------------------------------
    try:
        return yaml.safe_load(txt)
    except Exception:
        # give the *JSON* error – easier to understand for callers
        raise last_err

def robust_load_json(raw: str):
    """
    Best-effort loader for the sometimes messy JSON coming from the VLM.

    Strategy
    --------
    1.  plain ``json.loads``                                     (fast path)
    2.  sanitise obvious issues (un-escaped quotes, raw \n \r \t)
    3.  strip ``` fences   ➜  sanitise again
    4.  final fall-back:   *extract* the plan list with RegEx and
        build a minimal JSON around it so that downstream code can continue
        even if the rest of the payload is hopelessly malformed.

    Will raise the last JSONDecodeError only if **all** four passes fail.
    """
    # ---- 1) fast path --------------------------------------------------------
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e1:
        pass

    # ---- 2) sanitise & retry -------------------------------------------------
    try:
        return json.loads(_sanitize(raw))
    except json.JSONDecodeError as e2:
        pass

    # ---- 3) remove markdown fences and retry --------------------------------
    try:
        stripped = re.sub(r"```(?:json)?|```", "", raw, flags=re.I).strip()
        return json.loads(_sanitize(stripped))
    except json.JSONDecodeError as e3:
        pass

    # ---- 4) last-ditch: regex-pull a plan list ------------------------------
    plan_pat = re.compile(
        r'"(?:executable_plan|plan)"\s*:\s*(\[[^\]]*\])', re.S | re.I
    )
    m = plan_pat.search(raw)
    if not m:
        # propagate the *original* error – easiest for debugging
        raise e1

    plan_txt = m.group(1)
    # minimal valid JSON object
    return json.loads(f'{{"plan": {plan_txt}}}')


# ──────────────────────────────────────────────────────────────────────────────
#  _sanitize  – escape the two big culprits:    
#     • stray " inside strings      →   \"
#     • bare control chars          →   \n / \r / \t
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize(text: str) -> str:
    buf, in_str, i, L = [], False, 0, len(text)
    while i < L:
        ch = text[i]

        # toggle / escape stray quotes ----------------------------------------
        if ch == '"':
            esc = i and text[i - 1] == '\\'
            if not esc:
                in_str = not in_str
            else:
                buf.append(ch)
                i += 1
                continue

        # if we *are* inside a JSON string …
        if in_str:
            # 1) un-escaped quotes that appear *within* the string -------------
            if ch == '"' and (i + 1 < L and text[i + 1] not in ',:}]\r\n\t '):
                buf.append('\\')

            # 2) raw control characters  --------------------------------------
            if ch in '\n\r\t':
                buf.append('\\' + {'\n': 'n', '\r': 'r', '\t': 't'}[ch])
                i += 1
                continue

        buf.append(ch)
        i += 1

    return ''.join(buf)
    
def clean_name(input_name):
    if '_' in input_name:
        input_name_ls = input_name.split('_')
        if input_name_ls[-1].isdigit():
            new_name = ' '.join(input_name_ls[:-1])
        else:
            new_name = ' '.join(input_name_ls)
    else:
        new_name = input_name
    return new_name

def format_scene_graph(target_objects, target_attributes, target_relations):
     # Format structured target state
    if len(target_objects) > 0:
        clean_obj = [target_obj for target_obj in target_objects ]
        object_str = f"The objects are: {clean_obj}"
    else:
        object_str = ""
    
    if len(target_attributes) > 0:
        attr_str_ls = []
        for attr_info in target_attributes:
            if len(attr_info) != 2:
                continue
            attr, obj = attr_info
            attr_str_ls.append(f"({attr}, {obj})")
            
        attr_str = (
            "The attributes are: [" +
            ", ".join(attr_str_ls)
            + "]" if target_attributes else "The attributes are: []"
        )
    else:
        attr_str = ""
        
    rel_str_ls = []
    rel_str = ""
    if len(target_relations) > 0:
        
        for target_info in target_relations:
            if len(target_info) != 3: 
                continue
        
            subj, pred, obj = target_info
            rel_str_ls.append(f"({subj}, {pred}, {obj})")
            
        rel_str = (
            "The relations are: [" +
            ", ".join(rel_str_ls)
            + "]" if target_relations else "The relations are: []"
        )
    else:
        rel_str = ""
    
    output_str_ls = [object_str, attr_str,  rel_str]
    output_str_ls = [i for i in output_str_ls if not i == ""]
    output_str = " ".join(output_str_ls)
    return output_str

def format_state_descriptions(sg_info, current_obj, current_attr, current_rel):
    """Simplified scene graph formatting for GPT-4o"""
    # Extract basic elements with fallback key names
    target_desc = None
    current_desc = None
    
    # Try multiple possible key names for target state
    for key in ["target_state", "target_description", "goal_state", "desired_state"]:
        if key in sg_info and sg_info[key]:
            target_desc = sg_info[key]
            break
    
    # Try multiple possible key names for current state  
    for key in ["current_state", "current_description", "scene_description", "observation"]:
        if key in sg_info and sg_info[key]:
            current_desc = sg_info[key]
            break
    
    # Fallback descriptions if not found
    if not target_desc:
        target_desc = "Task goal not clearly specified"
    if not current_desc:
        current_desc = "Current scene state not described"
    
    # Create a concise, focused description
    sg_str = f"\n\n## Scene Context:\n"
    sg_str += f"**Target:** {target_desc}\n"
    sg_str += f"**Current:** {current_desc}\n"
    
    # Only add detected objects if they're relevant and few in number
    if current_obj and len(current_obj) <= 3:
        obj_str = ", ".join(current_obj)
        sg_str += f"**Detected Objects:** {obj_str}\n"
    
    return sg_str
    
def get_topk_per_object_name(aggr_res, avg_aggr_res, k):
    name_to_ids = defaultdict(list)

    # Group object ids by object name
    for obj_id, content in aggr_res.items():
        if "name" not in content:
            continue
        prob, name = content["name"]
        if name == "other":
            continue
        name_to_ids[name].append(obj_id)

    # Collect top-k per name
    topk_oid_to_score = {}
    for name, obj_ids in name_to_ids.items():
        valid_ids = [(obj_id, avg_aggr_res[obj_id]) for obj_id in obj_ids if obj_id in avg_aggr_res]
        sorted_ids = sorted(valid_ids, key=lambda x: x[1], reverse=True)[:k]
        for obj_id, score in sorted_ids:
            topk_oid_to_score[obj_id] = score

    return topk_oid_to_score

import heapq
from itertools import combinations

def get_topk_per_relation(predicted_name, binary_predicates, top_k):
    # Map from relation -> list of (score, (id1, relation, id2), (name1, name2))
    relation_to_scored_pairs = defaultdict(list)

    for (relation, obj1_label, obj2_label) in binary_predicates:
        for id1, id2 in combinations(predicted_name.keys(), 2):
            if id1 == id2:
                continue
            
            preds1 = predicted_name.get(id1, [])
            preds2 = predicted_name.get(id2, [])

            score1 = next((p for p, label in preds1 if label == obj1_label), 0.0)
            score2 = next((p for p, label in preds2 if label == obj2_label), 0.0)

            combined_score = score1 * score2
            if combined_score > 0:
                relation_to_scored_pairs[relation].append((combined_score, (id1, relation, id2), (obj1_label, obj2_label)))

    # For each relation, keep top-k entries
    topk_per_relation = {}
    for rel, scored_list in relation_to_scored_pairs.items():
        topk = heapq.nlargest(top_k, scored_list, key=lambda x: x[0])
        topk_per_relation[rel] = [
            {"pair": pair, "score": score, "names": (subj, obj)}
            for score, pair, (subj, obj) in topk
        ]

    return topk_per_relation

def merge_object_pair_ids(topk_per_relation): 
    all_object_pairs = set()
    
    for rel, topk_res_ls in topk_per_relation.items():
        for topk_res in topk_res_ls:
            subj, _, obj = topk_res['pair']
            all_object_pairs.add((0, 0, (subj, obj)))

    return sorted(list(all_object_pairs))

def load_model(model_dir, model_name, epoch, device):
    model_name = model_name + f".{epoch}.model"
    predicate_model = torch.load(
        os.path.join(model_dir, model_name), map_location=device, weights_only=False
    )
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
    mask[ymin : ymax + 1, xmin : xmax + 1, :] = True

    return mask

def convert_gd_boxes_to_pixel(boxes, img_width, img_height):
    """
    Convert GroundingDINO boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
    """
    if boxes is None or len(boxes) == 0:
        return []
    
    converted = []
    for box in boxes:
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        cx, cy, w, h = box
        # Convert from center format to corner format and denormalize
        x1 = int((cx - w/2) * img_width)
        y1 = int((cy - h/2) * img_height)
        x2 = int((cx + w/2) * img_width)
        y2 = int((cy + h/2) * img_height)
        converted.append([x1, y1, x2, y2])
    
    return converted

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    return result if result.endswith(".") else result + "."


def get_valid_bbox_val(v, b):
    return min(max(0, math.floor(v)), b)


def post_process_detections(det, classes):
    detections = det[0]
    class_ids = [
        0 if class_id is None else class_id for class_id in detections.class_id
    ]
    det.class_names = [classes[class_id] for class_id in class_ids]
    detections.xyxy = np.concatenate((detections.xyxy, det.xyxy), axis=0)
    detections.class_names = detections.class_names + det.class_names
    return detections


def list_depth(lst):
    """Calculates the depth of a nested list."""
    if not (isinstance(lst, list) or isinstance(lst, torch.Tensor)):
        return 0
    elif (isinstance(lst, torch.Tensor) and lst.shape == torch.Size([])) or (
        isinstance(lst, list) and len(lst) == 0
    ):
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)


def normalize_prompt(points, labels):

    if list_depth(points) == 3:
        points = torch.stack([p.unsqueeze(0) for p in points])
        labels = torch.stack([l.unsqueeze(0) for l in labels])
    return points, labels


# modify this function to take in a bbox label and show it


def show_box(box, ax, object_id, label="hi"):
    if len(box) == 0:
        return

    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=0.5)
    )

    if label is not None:
        ax.text(x0, y0 - 5, label, color="black", fontsize=10, alpha=color[3])


def save_prompts_one_image(frame_image, boxes, points, labels, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis("off")
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

    # for object_id, (point_ls, label_ls) in enumerate(zip(points, labels)):
    #     if not len(point_ls) == 0:
    #         show_points(point_ls.cpu(), label_ls.cpu(), ax, object_id=object_id)
    # Show the plot
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def check_device_status(self):
    """Check and print device status for all models"""
    print("\n" + "="*50)
    print("DEVICE STATUS CHECK")
    print("="*50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("-" * 30)
    
    # Check GroundingDINO model device
    if hasattr(self, 'grounding_model') and self.grounding_model is not None:
        try:
            # Get the device of the first parameter
            device = next(self.grounding_model.parameters()).device
            print(f"GroundingDINO model device: {device}")
        except Exception as e:
            print(f"GroundingDINO model device: Unable to determine ({e})")
    else:
        print("GroundingDINO model: Not available")
    
    # Check SAM2 model device
    if hasattr(self, 'sam2_predictor') and self.sam2_predictor is not None:
        try:
            device = next(self.sam2_predictor.model.parameters()).device
            print(f"SAM2 model device: {device}")
        except Exception as e:
            print(f"SAM2 model device: Unable to determine ({e})")
    else:
        print("SAM2 model: Not available")
    
    # Check LASER predicate model device
    if hasattr(self, 'predicate_model') and self.predicate_model is not None:
        try:
            device = next(self.predicate_model.parameters()).device
            print(f"LASER predicate model device: {device}")
        except Exception as e:
            print(f"LASER predicate model device: Unable to determine ({e})")
    else:
        print("LASER predicate model: Not available")
    
    # Check main VLM model (if it has device info)
    if hasattr(self, 'model') and self.model is not None:
        if hasattr(self.model, 'device'):
            print(f"Main VLM model device: {self.model.device}")
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
            print(f"Main VLM model device: {self.model.model.device}")
        else:
            print("Main VLM model device: Unable to determine (likely remote/API)")
    else:
        print("Main VLM model: Not available")
    
    print("="*50)

class VLMPlanner:
    def __init__(
        self,
        model_name,
        model_type,
        actions,
        system_prompt,
        examples,
        n_shot=0,
        obs_key="head_rgb",
        chat_history=False,
        language_only=False,
        use_feedback=True,
        multistep=0,
        tp=1,
        kwargs={},
        scene_graph_examples="",
        gd_config_path="",
        gd_checkpoint_path="",
        boundingbox = False, 
        blockingobj = False,
        scene_graph_text = False,
        gd_only = False,
        top_k=1,
        aggr_thres=0.3,
        gd_box_threshold = 0.1,
        gd_text_threshold = 0.1,
        rel_thres=0.3,
        # Add SAM2 parameters
        use_sam2=False,
        sam2_config_path="",
        sam2_checkpoint_path="",
    ):
        
        self.last_visual_description = ""
        self.last_reasoning = ""
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.scene_graph_examples = scene_graph_examples[:n_shot]
        self.n_shot = n_shot
        self.chat_history = (
            chat_history  # whether to includ all the chat history for prompting
        )
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == "custom":
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp, task_type="alfred")

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.processed_images = {}
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop("action_key", "action_id")
        self.gd_box_threshold = gd_box_threshold
        self.gd_text_threshold = gd_text_threshold
        self.rel_thres = rel_thres
        
        # Optimize prompt for GPT-4o - SIMPLIFIED VERSION
        if "gpt-4o" in model_name.lower():
            self.scene_graph_prompt = f'''## Simple Scene Analysis
Look at the image and describe what you see.

**Output Format (JSON):**
{{
  "target_state": "what should be achieved",
  "current_state": "what you currently see",
  "target_objects": ["objects", "needed"],
  "current_objects": ["objects", "you", "see"],
  "target_attributes": [],
  "current_attributes": [],
  "target_relations": [],
  "current_relations": [],
  "explanation": "Brief explanation of what needs to be done"
}}'''
        else:
            self.scene_graph_prompt = f'''## You are a robot operating in a home. You are given an image and observe the environment for performing the given task. \
                The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'}\
                The input text given to you contains the previous action history, and please deduce the current state from both visual image and previous action histories. \
                Please deduce the object names with a synonym if the descriptions is verbose or functional, or you cannot find it in the scene. 
                For example, substitute "freshly baked baguette" to "loaves", "portable device to access the internet" to "laptop", "light-emiting device on my desk to provide light for my work area" to "desk lamp".
                You are supposed to output in JSON.'''
            
        self.scene_graph_prompt_prefix = "\n A missing scene graph means the object is not clearly identifiable from the scene. It does not necessarily means this object is not in the scene. \n"
            

        # Initialize GroundingDINO with groundingdino-py package
        if GROUNDINGDINO_AVAILABLE:
            try:
                # For groundingdino-py, we need to download and load the model
                # The model will be downloaded automatically if not present
                from groundingdino.util.inference import load_model as load_grounding_model
                import groundingdino
                import os
                
                # Use the correct config path from the installed package
                package_dir = os.path.dirname(groundingdino.__file__)
                model_config_path = os.path.join(package_dir, "config", "GroundingDINO_SwinT_OGC.py")
                
                # Check if weights file exists, if not download it
                model_checkpoint_path = "/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth"
                if not os.path.exists(model_checkpoint_path):
                    print("Model weights not found. Please download them using:")
                    print("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                    self.grounding_model = None
                    self.gd_available = False
                else:
                    # Try to load with groundingdino-py's load_model function
                    self.grounding_model = load_grounding_model(model_config_path, model_checkpoint_path)
                    print("GroundingDINO initialized successfully with groundingdino-py package")
                    self.gd_available = True
            except Exception as e:
                print(f"Failed to initialize GroundingDINO: {e}")
                print("This is expected if model weights are not downloaded. GroundingDINO will be disabled.")
                print("To download weights, run: wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                self.grounding_model = None
                self.gd_available = False
        else:
            self.grounding_model = None
            self.gd_available = False
            print("GroundingDINO not available - some functionality will be limited")
        
        # Add SAM2 initialization
        self.use_sam2 = use_sam2 and SAM2_AVAILABLE
        if self.use_sam2:
            try:
                # Don't mess with Hydra initialization - SAM2 already handles this in its __init__.py
                # Just build the SAM2 model directly using the build_sam2 function
                self.sam2_predictor = SAM2ImagePredictor(
                    build_sam2(sam2_config_path, sam2_checkpoint_path, device="cuda:0", apply_postprocessing=False)
                )
                print("SAM2 initialized successfully")
            except Exception as e:
                print(f"SAM2 initialization failed: {e}")
                self.use_sam2 = False
                self.sam2_predictor = None
        else:
            self.sam2_predictor = None
            if use_sam2 and not SAM2_AVAILABLE:
                print("SAM2 requested but not available")
        
        self.boundingbox = boundingbox
        self.blockingobj = blockingobj
        self.scene_graph_text = scene_graph_text
        self.gd_only = gd_only
        self.aggr_thres = aggr_thres
        self.k = top_k
        
        all_model_dir = "/home/asethi04/common-data/jiani_common/llava_models"
        model_dir = os.path.join(all_model_dir, f"ensemble-02-10")
        model_name = "ensemble-2025-02-10-14-57-22"
        epoch = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # try:
        #     self.predicate_model = load_model(model_dir, model_name, epoch, device)
        # except Exception as e:
        #     print(f"Warning: Failed to load predicate model: {e}")
        #     print("Continuing without predicate model - some functionality may be limited")
        #     self.predicate_model = None
        clip_model_name = "openai/clip-vit-base-patch16"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predicate_model = PredicateModel_v3(hidden_dim = 128, num_top_pairs=1, device=device, model_name=clip_model_name).to(device)
        
        self.common_objects = ["wall", "door", "cabinet", "stove", "counter", "window", "other"]

    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ""
        for i in range(len(available_actions)):
            available_action_str += (
                "\naction id " + str(i) + ": " + str(available_actions[i])
            )
            if i < len(available_actions) - 1:
                available_action_str += ", "
        return available_action_str

    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        last_env_feedback = (
            prev_act_feedback[-1][1] if len(prev_act_feedback) else "N/A"
        )              
        user_instruction = user_instruction.rstrip(".")
        failure = False
        blocking_feedback = ""
        
        # Add long horizon task guidance and multi-object task handling
        is_long_horizon = any(keyword in user_instruction.lower() 
                             for keyword in ["both", "slice", "put", "move", "place"])
        is_multi_object = any(keyword in user_instruction.lower() 
                             for keyword in ["two", "both", "three", "multiple", "several"])
        is_from_to_task = " from " in user_instruction.lower() and " to " in user_instruction.lower()
        is_cleaning_task = any(keyword in user_instruction.lower() 
                             for keyword in ["rinsed", "washed", "cleaned", "clean", "wash", "rinse", "scrub", "scrubbed", "wet"])
        is_wetting_task = any(keyword in user_instruction.lower() 
                             for keyword in ["wet", "wash", "rinse", "clean"])
        is_slicing_task = any(keyword in user_instruction.lower() 
                             for keyword in ["slice", "piece", "cut", "chop"])
        is_heating_task = any(keyword in user_instruction.lower() 
                             for keyword in ["heat", "heated", "warm", "microwaved", "cooked", "cooking", "hot"])
        is_cooling_task = any(keyword in user_instruction.lower() 
                             for keyword in ["cool", "cold", "chill", "refrigerate", "chilled", "cooled"])
        
        # Add spatial reasoning guidance for spatial tasks
        is_spatial_task = any(spatial_word in user_instruction.lower() for spatial_word in 
                             ['left', 'right', 'next to', 'beside', 'near', 'on top', 'under', 'between', 'two', 'both', 'from'])
        
        # Add washing/wetting guidance
        washing_guidance = ""
        if is_wetting_task:
            washing_guidance = f"""
## **CRITICAL WASHING/WETTING SEQUENCE**

**URGENT - "WET" OBJECTS**: If task mentions "wet [object]", you MUST wash it first!

**CRITICAL ERRORS IN YOUR EXECUTION**: 
1. You turned faucet on/off but NEVER put the soap in the sink basin! (Soap stayed DRY!)
2. When you did "put down object", you put it on FLOOR, not IN sink basin!
3. You tried to "pick up the SoapBar" while already holding it! (Invalid action!)
4. You tried "find a Cabinet" but it doesn't exist - should try "find a Drawer"!

**MANDATORY WASHING/CLEANING SEQUENCE** (for ALL rinsing/washing tasks):
1. pick up the [Object] (get the object first)
2. find a Sink (establish spatial context)
3. **CRITICAL**: put down the object in hand (place object IN the sink basin FIRST!)
4. find a Faucet (find the faucet for cleaning)
5. turn on the Faucet (start water flow - object is already in sink)
6. turn off the Faucet (stop water flow - object is now cleaned)
7. find a [Object] (find the object again in the sink basin!)
8. pick up the [Object] (get the now wet/cleaned object from sink basin)
9. THEN proceed with placing the wet/cleaned object

**CRITICAL FOR ALL WASHING/CLEANING TASKS**: This sequence is MANDATORY for any task involving:
- **Washing words**: "cleaned", "wet", "rinsed", "washed", "clean", "rinse", "wash"
- **Cleaning context**: "wash the [object]", "clean the [object]", "rinse off the [object]"
- **State descriptions**: "wet [object]", "clean [object]", "rinsed [object]"

The object MUST be placed in the sink basin under running water to become wet/cleaned.

**WRONG**: Just "turn on Faucet" → "turn off Faucet" (object stays dry!)
**ALSO WRONG**: "put down object" without "find a Sink" first (object goes on floor, not in basin!)
**RIGHT**: "find a Sink" → "put down object in sink basin" → "find a Faucet" → "turn on Faucet" → "turn off Faucet" (object gets wet!)

**EXAMPLE - "Put wet soap in cabinet"**:
CORRECT: [24, 113, 79, 2, 155, 79, 133, 156, 24, 113, 51, 147, 133, 148]
1. find a SoapBar
2. pick up the SoapBar
3. find a Sink  
4. find a Faucet
5. turn on the Faucet (water starts flowing)
6. **CRITICAL**: find a Sink (go to sink basin location)
7. **CRITICAL**: put down the object in hand (soap goes IN the sink basin under running water!)
8. turn off the Faucet (stop water)
9. **CRITICAL**: find a SoapBar (find the soap again in the sink basin!)
10. pick up the SoapBar (get the now wet soap from sink basin)
11. find a Drawer (or Cabinet)
12. open the Drawer
13. put down the object in hand (wet soap goes in drawer)
14. close the Drawer

**PROBLEM IN YOUR LOG**: Robot skipped steps 6-8 - never put soap IN sink AND never found it again!

**ADDITIONAL CRITICAL ERRORS FROM YOUR LOG**:
1. **"Robot is currently holding SoapBar"** → You tried to pick up soap you already had!
2. **"Cannot find Cabinet"** → You should try "find a Drawer" or "find a CounterTop" instead!

**KEY**: The object must interact with running water to become "wet"!
"""

        spatial_guidance = ""
        if is_spatial_task:
            spatial_guidance = f"""
## **CRITICAL SPATIAL REASONING RULES**

**URGENT - MULTI-OBJECT HANDLING**: If task mentions "two X", "both X", or multiple objects:

**CORRECT SEQUENCE FOR MULTI-OBJECTS**:
1. find a [Destination] → open the [Destination] (if needed)
2. find a [Object] → pick up the [Object] (first instance)
3. find a [Destination] → put down the object in hand (go back to destination!)
4. find a [Object_2] → pick up the [Object] (second instance - use specific instance!)
5. find a [Destination] → put down the object in hand (go back to destination!)
6. close the [Destination]

**CRITICAL - USE SPECIFIC OBJECT INSTANCES**: 
- For "two knives" → `find a Knife` + `pick up the Knife`, then `find a Knife_2` + `pick up the Knife`
- For "both lettuce" → `find a Lettuce` + `pick up the Lettuce`, then `find a Lettuce_2` + `pick up the Lettuce`  
- For "three apples" → `find a Apple` + `pick up the Apple`, then `find a Apple_2` + `pick up the Apple`, then `find a Apple_3` + `pick up the Apple`
- **For "two bottles"** → `find a SoapBottle` + `pick up the SoapBottle`, then `find a SoapBottle_2` + `pick up the SoapBottle`

**CRITICAL OBJECT NAMING**:
- **"bottles"** = use `find a SoapBottle` (NOT `find a Glassbottle`)
- **"glasses"** = use `find a WineGlass` (NOT `find a Glass`)
- **"plates"** = use `find a Plate` (NOT `find a DishPlate`)
- Check available actions - use EXACT names from action list!

**OBJECT INSTANCE PATTERN**:
1. First object: `find a [Object]` (finds first instance)
2. Subsequent objects: `find a [Object_2]`, `find a [Object_3]`, etc. (finds specific instances)
3. Always use `pick up the [Object]` (generic pickup action works from specific location)

**NEVER DO**: `pick up the Knife` while getting "Robot is currently holding Knife"!

**SPATIAL DIRECTIONS - BE SPECIFIC**:
- "drawer to the LEFT of stove" → find a Drawer (NOT find a Cabinet!)
- "right side of table" → find specific location, not generic table
- "from dining table" → go to dining table first, then find objects there

**RECEPTACLE IDENTIFICATION**:
- **Drawer** = slide-out compartment → find a Drawer → open the Drawer
- **Cabinet** = door-front storage → find a Cabinet → open the Cabinet  
- **CounterTop** = flat surface → find a CounterTop (no opening)
**CRITICAL**: "drawer to left of stove" means DRAWER, not Cabinet!

**RECEPTACLE ALTERNATIVES** (when primary fails):
- If "Cannot find Cabinet" → try: find a Drawer, find a CounterTop, find a Shelf
- If "Cannot find Drawer" → try: find a Cabinet, find a CounterTop
- If "Cannot find CounterTop" → try: find a Cabinet, find a Drawer

**ENVIRONMENTAL FEEDBACK HANDLING**:
- **"Robot is currently holding [Object]"** → DON'T try to pick up again! You already have it!
- **"Cannot find Cabinet"** → Try related objects: find a Drawer, find a Dresser, find a CounterTop, find a Shelf
- **"Cannot find Drawer"** → Try related objects: find a Cabinet, find a Dresser, find a CounterTop
- **"Cannot find Dresser"** → Try related objects: find a Cabinet, find a Drawer, find a SideTable
- **"Cabinet is not visible"** → You may have found the wrong cabinet! Try: find a Sink → find a Cabinet (re-establish spatial context)
- **"Putting object on [Receptacle] failed"** → You may be at wrong receptacle! Try: find a Sink → find a Cabinet (re-establish spatial context)
- "Open action failed. The [Receptacle] is already open" → skip opening, go straight to putting down
- "Close action failed" → drawer might not be ready to close, ensure all objects are properly placed inside

**CRITICAL - RECEPTACLE SUBSTITUTION RULES**:
- **Cabinet fails** → Try: Drawer, Dresser, Shelf, CounterTop
- **Drawer fails** → Try: Cabinet, Dresser, CounterTop  
- **Dresser fails** → Try: Cabinet, Drawer, SideTable
- **Table fails** → Try: CounterTop, SideTable, Desk
- **Shelf fails** → Try: Cabinet, CounterTop, SideTable
- **Always try related storage/surface objects when primary receptacle fails!**

**CRITICAL ERRORS TO AVOID**:
1. **Double Pickup Error**: If holding SoapBar, DON'T do "pick up the SoapBar" again!
2. **Missing Receptacle Error**: If "Cannot find Cabinet", try "find a Drawer" or "find a CounterTop"

**EXAMPLE - "Put two knives from dining table in drawer to left of stove"**:
CORRECT: [51, 147, 34, 127, 51, 133, 240, 127, 51, 133, 148]
1. find a Drawer (drawer to left of stove - NOT Cabinet!)
2. open the Drawer
3. find a Knife (first knife - generic finds first instance)
4. pick up the Knife (picks up first knife)
5. find a Drawer (go back to drawer!)
6. put down the object in hand (first knife in drawer)
7. find a Knife_2 (second knife - SPECIFIC instance!)
8. pick up the Knife (picks up second knife from Knife_2 location)
9. find a Drawer (go back to drawer again!)
10. put down the object in hand (second knife in drawer)
11. close the Drawer

**KEY**: Always `find a Drawer` before each `put down` to ensure objects go INTO the drawer, not at the object's location!

**CRITICAL - SINK-CABINET SPATIAL RELATIONSHIPS**:
When task mentions "cabinet under the sink" or "cabinet below the sink":

**CORRECT SEQUENCE FOR SINK-CABINET TASKS**:
1. find a Sink (first locate the sink to establish spatial context)
2. find a Cabinet (find the cabinet that is spatially related to the sink)
3. open the Cabinet (open the specific cabinet under the sink)
4. [Object manipulation sequence]
5. close the Cabinet

**SPATIAL CONTEXT MAINTENANCE**:
- **CRITICAL**: Always `find a Sink` BEFORE `find a Cabinet` to establish spatial relationship
- **CRITICAL**: After `find a Sink`, the next `find a Cabinet` will target the cabinet under that sink
- **CRITICAL**: To return to the SAME cabinet under the sink, use: `find a Sink` → `find a Cabinet` (re-establish spatial context)
- **CRITICAL**: Maintain spatial context by referencing the sink-cabinet relationship

**CRITICAL - SPATIAL POSITIONING TASKS**:
When task mentions "right of", "left of", "next to", or other spatial relationships:

**CORRECT SEQUENCE FOR SPATIAL POSITIONING**:
1. find a [ReferenceObject] (e.g., find a Sink - establish spatial reference)
2. find a [TargetSurface] (e.g., find a CounterTop - find the surface for placement)
3. put down the object in hand (place object on the surface - the spatial relationship is maintained by the reference)

**SPATIAL POSITIONING RULES**:
- **CRITICAL**: Always `find a [ReferenceObject]` BEFORE `find a [TargetSurface]` to establish spatial context
- **CRITICAL**: The spatial relationship is maintained by the reference object, not by the placement action
- **CRITICAL**: After establishing spatial context, use generic `put down` - the environment will respect the spatial relationship
- **CRITICAL**: If spatial positioning fails, re-establish context: `find a [ReferenceObject]` → `find a [TargetSurface]` → `put down`

**QUICK SPATIAL POSITIONING GUIDE**:
- **"left of [object]"** → find a [Object] → find a [TargetSurface] → put down
- **"right of [object]"** → find a [Object] → find a [TargetSurface] → put down  
- **"top of [object]"** → find a [Object] → find a [TargetSurface] → put down
- **"above [object]"** → find a [Object] → find a [TargetSurface] → put down
- **"below [object]"** → find a [Object] → find a [TargetSurface] → put down
- **"next to [object]"** → find a [Object] → find a [TargetSurface] → put down
- **"in front of [object]"** → find a [Object] → find a [TargetSurface] → put down
- **"behind [object]"** → find a [Object] → find a [TargetSurface] → put down

**KEY RULE**: First find the reference object, then find the target surface - this establishes the spatial relationship!

**EXAMPLE - "Put two bottles in a cabinet under the sink"**:
CORRECT: [79, 48, 145, 76, 126, 79, 48, 133, 177, 126, 79, 48, 133, 146]
1. find a Sink (establish spatial context - this is THE sink)
2. find a Cabinet (find the cabinet under the sink we just located)
3. open the Cabinet (open the specific cabinet under the sink)
4. find a SoapBottle_1 (first bottle - specific instance)
5. pick up the SoapBottle (pick up first bottle)
6. find a Sink (re-establish spatial context to ensure we go to the RIGHT cabinet!)
7. find a Cabinet (find the SAME cabinet under the sink we established earlier)
8. put down the object in hand (first bottle in cabinet under sink)
9. find a SoapBottle_2 (second bottle - specific instance)
10. pick up the SoapBottle (pick up second bottle)
11. find a Sink (re-establish spatial context again to ensure we go to the RIGHT cabinet!)
12. find a Cabinet (find the SAME cabinet under the sink we established earlier)
13. put down the object in hand (second bottle in cabinet under sink)
14. close the Cabinet (close the cabinet under the sink)

**EXAMPLE - "Put cleaned lettuce on the counter, right of the sink"**:
CORRECT: [55, 95, 79, 133, 2, 155, 156, 55, 95, 79, 39, 133]
1. find a Lettuce (locate the lettuce)
2. pick up the Lettuce (pick up the lettuce)
3. find a Sink (establish spatial reference - this is THE sink)
4. put down the object in hand (place lettuce in sink basin FIRST)
5. find a Faucet (find the faucet for cleaning)
6. turn on the Faucet (start water flow - lettuce is already in sink)
7. turn off the Faucet (stop water flow - lettuce is now cleaned)
8. find a Lettuce (find the cleaned lettuce in sink)
9. pick up the Lettuce (pick up the cleaned lettuce)
10. find a Sink (re-establish spatial reference for positioning)
11. find a CounterTop (find the counter surface)
12. put down the object in hand (place lettuce on counter - spatial relationship maintained by sink reference)

**TASK-SPECIFIC PATTERNS**:

**"Pick Two Objects" Tasks** (e.g., "Put two bottles in cabinet"):
1. find a [Object] (first object)
2. pick up the [Object] (pick up first object)
3. find a [Receptacle] (find target location)
4. put down the object in hand (place first object)
5. find a [Object] (second object - may need to specify instance)
6. pick up the [Object] (pick up second object)
7. find a [Receptacle] (return to target location)
8. put down the object in hand (place second object)

**"Movable Receptacle" Tasks** (e.g., "Put mug with pen in it on table"):
1. find a [Container] (e.g., find a Mug)
2. pick up the [Container] (pick up container)
3. find a [Object] (e.g., find a Pen)
4. pick up the [Object] (pick up object to put in container)
5. find a [Container] (return to container)
6. put down the object in hand (place object IN container)
7. find a [TargetSurface] (e.g., find a DiningTable)
8. put down the object in hand (place container with object on target surface)

**"Clean Then Place" Tasks** (e.g., "Wash ladle and put it back"):
1. find a [Object] (locate object to clean)
2. pick up the [Object] (pick up object)
3. find a Sink (establish spatial context)
4. put down the object in hand (place object in sink basin)
5. find a Faucet (find faucet for cleaning)
6. turn on the Faucet (start water flow)
7. turn off the Faucet (stop water flow - object is now cleaned)
8. find a [Object] (find cleaned object in sink)
9. pick up the [Object] (pick up cleaned object)
10. find a [TargetSurface] (find where to place cleaned object)
11. put down the object in hand (place cleaned object)

**KEY INSIGHT**: The spatial relationship is established by finding the reference object FIRST, then finding the target object. This ensures the robot targets the correct spatially-related object.
"""

        long_horizon_guidance = ""
        if is_long_horizon:
            long_horizon_guidance = f"""
## **CRITICAL ACTION SEQUENCE RULES**

**URGENT - ENVIRONMENTAL FEEDBACK**: If you get "Knife is not visible because it is in [Location]":
→ **CounterTop**: You MUST go to the EXACT CounterTop where the knife is! Try: find a Knife → pick up the Knife (go directly to knife location)
→ **Drawer_X**: find a Drawer_X → open the Drawer_X →  pick up the Knife  
→ **Cabinet_X**: find a Cabinet_X → open the Cabinet_X →  pick up the Knife

**CRITICAL CounterTop Issue**: If "find a CounterTop" then "find a Knife" works but "pick up the Knife" fails with "Knife is not visible because it is in CounterTop", the knife is at a DIFFERENT CounterTop location! Solution: find a Knife → pick up the Knife (go directly to knife!)

**RULE 1 - SLICING COMES FIRST**: If task mentions "slice", "piece", "cut", "chop" - SLICE FIRST!

**MANDATORY SLICING SEQUENCE**:
1. **IF KNIFE ACCESS FAILS**: If you get "Knife is not visible because it is in [Location]":
   → **OPENABLE**: find a [Location] → open the [Location] → pick up the Knife
   → **NON-OPENABLE**: find a [Location] → pick up the Knife
   **EXAMPLES**: 
   - "Drawer_3" → find a Drawer_3 → open the Drawer_3 → pick up the Knife
   - "CounterTop" → find a CounterTop → pick up the Knife (no opening needed)
2. find a Knife → pick up the Knife  
3. find a [Object] → slice the [Object]
4. put down the knife (or store it)
5. pick up the [Object] (sliced piece - you're already at this location!)
6. THEN proceed with moving sliced pieces

**RULE 2 - LOCATION AWARENESS**: You CANNOT pick up an object unless you are AT its location first!

**WRONG**: find Apple → pick up Apple (fails if you navigate away from apple)
**CORRECT**: find Apple → find GarbageCan → pick up Apple (go to garbage can, then pick up apple that's there)

**COMPLETE EXAMPLE: "Put a microwaved slice of bread in the fridge"**

**SITUATION**: You try "find a Knife" → "pick up the Knife" but get error: 
"Knife is not visible because it is in CounterTop. Go there to pick the object instead."

**CRITICAL**: CounterTop cannot be opened! Just go there and find the knife on the surface!

**CORRECT ACTION SEQUENCE WITH KNIFE ACCESS**:
[34, 127, 53, 161, 133, 92, 38, 143, 133, 144, 149, 150, 143, 92, 144, 78, 139, 133, 140]

**Step-by-step Translation:**
1. find a Knife (go directly to knife location - skip generic CounterTop!)
2. pick up the Knife (get slicing tool from its specific location)
3. find a Bread (locate object to slice)
4. slice the Bread (slice while holding knife - bread stays in place)
5. put down the object in hand (put down knife)
6. pick up the Bread (pick up sliced piece - you're already at this location!)
7. find a Microwave (for heating)
8. open the Microwave
9. put down the object in hand (place slice in microwave)
10. close the Microwave
11. turn on the Microwave
12. turn off the Microwave
13. open the Microwave
14. pick up the Bread (get heated slice)
15. close the Microwave
16. find a Fridge (destination)
17. open the Fridge
18. put down the object in hand (place slice in fridge)
19. close the Fridge

**CRITICAL**: NEVER try "find a Bread" → "pick up the Bread" → "slice the Bread" - this ALWAYS fails!
**ALWAYS**: Get knife first, then go to object and slice it while holding knife!

**KEY EFFICIENCY RULE**: After slicing, you're already at the location - directly pick up the sliced piece!

**KEY INSIGHT**: Apple is already IN the garbage can, so go there first to pick it up!

**KEY RULE**: Use "warmed" = microwave sequence. Use "put in garbage can" = find garbage can then put down.

## **ENVIRONMENTAL FEEDBACK HANDLING**

**CRITICAL**: When you get feedback like "Apple is not visible because it is in GarbageCan. Go there to pick the object instead."

**IMMEDIATE RESPONSE**:
1. find a GarbageCan (go to the garbage can)
2. pick up the Apple (directly pick up the apple that's there)

**NO OTHER ACTIONS NEEDED** - just go to the location mentioned in the feedback and pick up the object directly!

## **SLICING TASK FAILURES**

**ERROR 1**: "Knife is not visible because it is in [Location]"
**SOLUTION**: Go to that EXACT location first! Don't use generic "Cabinet" or "Drawer"!
→ If feedback says "Drawer_3" → find a Drawer_3 → open the Drawer_3 → pick up the Knife
→ If feedback says "Cabinet_2" → find a Cabinet_2 → open the Cabinet_2 → pick up the Knife
→ **SPECIAL CASE - CounterTop**: If feedback says "CounterTop" → find a CounterTop → find a Knife → pick up the Knife
  (CounterTop cannot be opened - just go there and find the knife on the surface)

**ERROR 2**: "Slice action failed" or "Robot is currently holding [Object]"  
**CAUSE**: Trying to slice while holding the object
**SOLUTION**: NEVER pick up object before slicing! Get knife first!

**ERROR 3**: "Robot is not holding any object" when trying to pick up knife
**CAUSE**: Trying to pick up knife from wrong location
**SOLUTION**: Follow environmental feedback to find correct storage location

**CORRECT SLICING FLOW**:
1. Put down any object you're holding first
2. Handle knife access (open drawer/cabinet if needed)
3. find a Knife → pick up the Knife
4. find a [Object] → slice the [Object] (while holding knife)
5. Repeat slicing for all objects needed  
6. Put down knife, THEN pick up sliced pieces
"""
        
        
        if not len(prev_act_feedback) == 0:
            last_feedback = prev_act_feedback[-1]
            if last_feedback[0] != 133:
                failure = True
                blocking_feedback = last_feedback[1]
                
            # Handle specific "object is in location" feedback
            if "is not visible because it is in" in last_feedback[1]:
                long_horizon_guidance += f"""
## **URGENT: ENVIRONMENTAL FEEDBACK RECEIVED**

**Feedback**: "{last_feedback[1]}"

**IMMEDIATE ACTION REQUIRED**: The object you're trying to pick up is at a different location!
**SOLUTION**: 
1. Go to the location mentioned in the feedback
2. Pick up the object directly from that location

**Example**: If feedback says "Apple is not visible because it is in GarbageCan"
**DO THIS**: [find a GarbageCan, pick up the Apple]
"""
                
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                
                prompt = self.system_prompt.format(
                    len(self.actions) - 1,
                    self.available_action_str,
                    "\n\n".join(
                        [
                            f"## Task Execution Example {i}: \n {x}"
                            for i, x in enumerate(self.examples[: self.n_shot])
                        ]
                    ),
                )
                scene_graph_prompt = self.scene_graph_prompt + "\n\n".join(
                    [
                        f"## Vision Recognition Example {i}: \n {x}"
                        for i, x in enumerate(self.scene_graph_examples)
                    ]
                )
            else:

                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(
                    len(self.actions) - 1, self.available_action_str, ""
                )

            prompt += f"\n\n## Now the human instruction is: {user_instruction}."

            scene_graph_prompt += (
                f"\n\n## Now the human instruction is: {user_instruction}."
            )

            if self.language_only:
                prompt += f" You are supposed to output in json. Note that nested quotes is not allowed. Please use single quote within double quotes properly. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. Note that nested quotes is not allowed. Please use single quote within double quotes properly. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            
            # Add long horizon guidance to initial prompt
            if long_horizon_guidance:
                prompt += f"\n\n{long_horizon_guidance}"
            
            # Add spatial guidance to initial prompt
            if spatial_guidance:
                prompt += f"\n\n{spatial_guidance}"
            
            # Add washing guidance to initial prompt
            if washing_guidance:
                prompt += f"\n\n{washing_guidance}"

        elif self.chat_history:
            
            # This is to support the sliding window feature
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f"The human instruction is: {user_instruction}."
            prompt += "\n\n The action history:"
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += "\nStep {}, action id {}, {}, env feedback: {}".format(
                        i,
                        action_feedback[0],
                        self.actions[action_feedback[0]],
                        action_feedback[1],
                    )
                else:
                    prompt += "\nStep {}, action id {}, {}".format(
                        i, action_feedback[0], self.actions[action_feedback[0]]
                    )
            if failure:
                scene_graph_prompt += f"The last action is invalid, and the feedback is: {blocking_feedback} "
            else:
                scene_graph_prompt += f"The last action succeeds. "
                
            scene_graph_prompt += f'In the explanation field, please include what are the previous actions that leads you to this description. Also provide candidate related objects if one thing is not found. \n\n## Now the human instruction is: {user_instruction}.'

            if self.language_only:
                prompt += f"""\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
            else:
                prompt += f"""\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
        else:
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + "\n\n".join(
                    [
                        f"## Vision Recognition Example {i}: \n {x}"
                        for i, x in enumerate(self.scene_graph_examples)
                    ]
                )
                prompt = self.system_prompt.format(
                    len(self.actions) - 1,
                    self.available_action_str,
                    "\n\n".join(
                        [
                            f"## Task Execution Example  {i}: \n {x}"
                            for i, x in enumerate(self.examples[: self.n_shot])
                        ]
                    ),
                )
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(
                    len(self.actions) - 1, self.available_action_str, ""
                )
            prompt += f"\n\n## Now the human instruction is: {user_instruction}."
            
            action_history_prompt = "\n\n The action history:"
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    action_history_prompt += "\nStep {}, action id {}, {}, env feedback: {}".format(
                        i,
                        action_feedback[0],
                        self.actions[action_feedback[0]],
                        action_feedback[1],
                    )
                else:
                    action_history_prompt += "\nStep {}, action id {}, {}".format(
                        i, action_feedback[0], self.actions[action_feedback[0]]
                    )
            
            prompt += action_history_prompt
            scene_graph_prompt += action_history_prompt
                    
            if failure:
                scene_graph_prompt += f"The last action is invalid, and the feedback is: {blocking_feedback} "
            else:
                scene_graph_prompt += f"The last action succeeds. "
                
            # scene_graph_prompt += (
            #     # f"\n\nEnvironmental Feedback: {last_env_feedback}"
            #     f"\n\nLast Visual Description: {self.last_visual_description}\n"
            # )
            
            # scene_graph_prompt += (
            #     # f"\n\nEnvironmental Feedback: {last_env_feedback}"
            #     f"\n\nReasoning and Reflection: {self.last_reasoning}\n"
            # )
            
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            if self.language_only:
                prompt += f"""\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
            else:
                prompt += f"""\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
            
            # Add spatial guidance to continuation prompts
            if spatial_guidance:
                prompt += f"\n\n{spatial_guidance}"
            
            # Add washing guidance to continuation prompts
            if washing_guidance:
                prompt += f"\n\n{washing_guidance}"
        
        return prompt, scene_graph_prompt

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image
            elif type(image) == list:
                image_path = image[-1]
            else:
                image_path = "./evaluation/tmp_{}.png".format(len(messages) // 2)
                np_image = image

                if isinstance(np_image, torch.Tensor):
                    # Move to CPU (if on GPU), detach from graph, and convert to numpy
                    # Also check shape, e.g. if [C,H,W], permute to [H,W,C]
                    if np_image.ndim == 3 and np_image.shape[0] <= 4:  # typical [C,H,W]
                        np_image = np_image.permute(1, 2, 0)
                    np_image = np_image.detach().cpu().numpy()

                    # Make sure it's a copy and thus writable
                    np_image = np_image.copy()

                if isinstance(np_image, Image.Image):
                    # Convert from PIL to numpy array
                    np_image = np.array(image)

                    # Optional: If `image` is in RGB and you want OpenCV's default BGR:
                    # np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(image_path, np_image)

            if self.multistep:  # handle multiple images
                ind = int(image_path.split("step_")[-1].strip(".png"))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind + 1):
                    temp_path = (
                        "".join(image_path.split("step_")[:-1]) + f"step_{str(i)}.png"
                    )
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            },
                        }
                    )
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                    {"type": "text", "text": prompt},
                ]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.last_visual_description = ""
        self.last_reasoning = ""
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r"\*\*\d+\*\*"
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip("*"))
        else:
            print("random action")
            action = np.random.randint(len(self.actions))
        return action

    def json_to_action(self, output_text, json_key="executable_plan"):
        valid = True
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print("empty plan, stop here")
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if act >= len(self.actions) or act < 0:
                        print("found invlid action")
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
            valid = False
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
            valid = False
        return action, valid
    
    def json_to_action_alt(
            self,
            output_text: str,
            json_keys: tuple = ("executable_plan", "plan"),
        ):
        """
        Parse the model’s JSON and convert it to a list of action-ids.

        • Accept either ``"executable_plan"`` or ``"plan"``.  
        • Trust an ``action_id`` only if it is in range *and* its name matches
        the supplied ``action_name`` after normalisation.  
        • Otherwise attempt to resolve the step purely by its ``action_name``;
        **the name must exactly (case-insensitive) match one entry in
        ``self.actions``.**  
        • Returns  (action_list | int_flag ,  valid_bool)
            -2  … empty plan / stop  
            -1  … could not resolve a step           

        Normalisation = lower-case & collapse multiple spaces.
        """
        def norm(txt: str) -> str:                # helper
            return " ".join(txt.lower().split())

        # build fast lookup: normalised name  ->  id
        name2id = {norm(a): i for i, a in enumerate(self.actions)}

        try:
            # bad = output_text
            # # quick sanitiser – remove any quote directly before a t in isn"t ➜ isn't
            # bad = re.sub(r'isn"t', "isn't", bad)

            # obj = json.loads(bad) 
            def parse_json_with_fixes(json_str):
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try fixing common escape issues
                    fixed_str = json_str.replace("\\'", "'")
                    return json.loads(fixed_str)
  
            obj = parse_json_with_fixes(output_text)

            # locate a list plan (first key that exists & is a list)
            plan = next((obj[k] for k in json_keys if isinstance(obj.get(k), list)), None)
            if plan is None:
                raise ValueError("no plan key found")

            action_ids = []
            valid = True
            for step in plan:
                if not isinstance(step, dict):
                    raise ValueError("plan step not dict")

                aid = step.get(self.action_key)          # e.g. "action_id"
                anm = step.get("action_name")
                nm  = norm(anm) if isinstance(anm, str) else None

                resolved = None

                # ── (1)  validate a provided id ─────────────────────────────────
                if (isinstance(aid, int)
                        and 0 <= aid < len(self.actions)
                        and nm == norm(self.actions[aid])):
                    resolved = aid

                # ── (2)  otherwise resolve by exact name match ──────────────────
                if resolved is None and nm in name2id:
                    resolved = name2id[nm]

                # ── (3) Enhanced validation for long horizon tasks ──────────────
                if resolved is None and isinstance(aid, int):
                    # Check if action ID is just out of bounds but name is valid
                    if aid >= len(self.actions) and nm in name2id:
                        resolved = name2id[nm]
                        print(f"Fixed invalid action ID {aid} -> {resolved} for '{anm}'")
                    
                if resolved is None:      # could not resolve → invalid plan
                    print(f"Could not resolve action: id={aid}, name='{anm}', normalized='{nm}'")
                    print(f"Available actions: {list(name2id.keys())[:10]}...")  # Show first 10
                    valid = False
                    break

                action_ids.append(resolved)

            if not action_ids:            # empty => stop signal
                action = -2
                valid  = True
            elif valid is False:          # at least one step unresolved
                action = -1
            else:
                action = action_ids
                valid  = True

        except Exception as e:
            print("Could not parse model JSON:", e)
            self.output_json_error += 1
            action = -1
            valid  = False

        return action, valid

    def predict_gd(self, image, classes, box_threshold=0.5, text_threshold=0.4):
        if not self.gd_available or self.grounding_model is None:
            print("GroundingDINO not available, returning empty results")
            return [], []
            
        # Convert classes list to caption
        caption = ". ".join(classes)
        processed_caption = preprocess_caption(caption)
        
        try:
            # Use groundingdino-py API
            # Convert image to the format expected by groundingdino-py
            from PIL import Image as PILImage
            import torchvision.transforms as T
            
           # After your RGB conversion
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Convert to PIL Image if it's a numpy array
            if isinstance(rgb_image, np.ndarray):
                pil_image = PILImage.fromarray(rgb_image)
            else:
                pil_image = rgb_image  # Already a PIL Image

            # Create the transform
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization

            ])

            # Convert to tensor
            image_tensor = transform(pil_image)
                
           
            
            # Use groundingdino-py's predict function
            boxes, logits, phrases = predict(
                model=self.grounding_model,
                image=image_tensor,
                caption=processed_caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Convert results to the expected format
            labels = phrases if phrases is not None else []
            
            # Convert boxes to tensor if they aren't already
            if boxes is not None and len(boxes) > 0:
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes)
            else:
                boxes = torch.tensor([])        
        except Exception as e:
            print(f"Error in GroundingDINO prediction: {e}")
            return [], []
                
        return labels, boxes
    
    
    
    # def predict_laser(self, 
    #               image, 
    #               boxes,  # Now expects pixel coordinates as list of [x1, y1, x2, y2]
    #               masks=None,  # Optional masks parameter
    #               target_names=None,
    #               target_relation_kws=None,
    #               target_relation_preds=None,
    #               query_names=None,
    #               query_relations=None,
    #               query_relation_kws=None,
    #               query_attributes=None,
    #               ):
        
    #     # Preprocess query attributes for the case where one object having multiple attribute
    #     query_attribute_ls = [[attr, f"not {attr}"] for attr in query_attributes]
    #     prob_labels, selected_object_bboxes, identified_rels = [], [], []
        
    #     if not len(boxes) == 0 and self.predicate_model is not None:
    #         height, width = image.shape[0], image.shape[1]
            
    #         # Use provided masks or generate from bboxes
    #         if masks is not None and len(masks) > 0:
    #             # Ensure masks have the right shape
    #             batched_pred_masks = []
    #             for mask in masks:
    #                 if isinstance(mask, torch.Tensor):
    #                     mask = mask.cpu().numpy()
    #                 # Add channel dimension if needed
    #                 if mask.ndim == 2:
    #                     mask = mask[:, :, np.newaxis]
    #                 batched_pred_masks.append(mask)
    #         else:
    #             # Fall back to generating masks from bboxes
    #             batched_pred_masks = [bbox_to_mask(bbox, height, width) for bbox in boxes]
            
    #         batched_object_ids = [[0, 0, i] for i in range(len(batched_pred_masks))]
            
    #         batched_image_cate_probs, _, _, _ = \
    #             self.predicate_model(
    #                 batched_video_ids=[0],
    #                 batched_videos=[image],
    #                 batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
    #                 batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
    #                 batched_names=[query_names],  # Dataset-wise categorical labels
    #                 batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
    #                 batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
    #                 batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
    #                 batched_obj_pairs=[],  # Ground truth binary relations
    #                 batched_video_splits=[0],  # [number of videos]
    #                 batched_binary_predicates=[[]],  # None indicates inference time
    #             )
                    
    #         all_unary_res = []
    #         for unary in query_attribute_ls:
    #             _, batched_image_unary_probs, _, _ = \
    #                 self.predicate_model(
    #                     batched_video_ids=[0],
    #                     batched_videos=[image],
    #                     batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
    #                     batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
    #                     batched_names=[query_names],  # Dataset-wise categorical labels
    #                     batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
    #                     batched_unary_kws=[unary],  # Dataset-wise unary predicate labels
    #                     batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
    #                     batched_obj_pairs=[],  # Ground truth binary relations
    #                     batched_video_splits=[0],  # [number of videos]
    #                     batched_binary_predicates=[None],  # None indicates inference time
    #                 )
    #             all_unary_res.append(batched_image_unary_probs)
            
    #         # Combine the unary and object name information together 
    #         # Step 1: reformat the object name, attributes, and relations into better format
    #         predicted_name = {}
    #         for (oid, name) in batched_image_cate_probs[0]:
    #             prob = batched_image_cate_probs[0][(oid, name)].item()
    #             if not oid in predicted_name:
    #                 predicted_name[oid] = []
    #             predicted_name[oid].append((prob, name))
            
    #         for oid in predicted_name:
    #             predicted_name[oid] = sorted(predicted_name[oid], reverse=True)   
                    
    #         reformatted_unary_res = {}
    #         for unary_res in all_unary_res:
    #             for  (fid, oid, kw) in unary_res[0].keys():
    #                 if not oid in reformatted_unary_res:
    #                     reformatted_unary_res[oid] = {}
    #                 if kw in query_attributes:
    #                     reformatted_unary_res[oid][kw] = unary_res[0][(fid, oid, kw)].item()
            
    #         topk_obj_pair_info = get_topk_per_relation(predicted_name, binary_predicates=query_relations, top_k=30)
    #         top_obj_pairs = merge_object_pair_ids(topk_obj_pair_info)
            
    #         batched_image_rel_probs = {}
    #         if len(top_obj_pairs) > 0:
    #             _, _, batched_image_rel_probs, _ = \
    #                 self.predicate_model(
    #                     batched_video_ids=[0],
    #                     batched_videos=[image],
    #                     batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
    #                     batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
    #                     batched_names=[query_names],  # Dataset-wise categorical labels
    #                     batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
    #                     batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
    #                     batched_binary_kws=[query_relation_kws],  # Dataset-wise binary predicate labels
    #                     batched_obj_pairs=top_obj_pairs,  # Ground truth binary relations
    #                     batched_video_splits=[0],  # [number of videos]
    #                     batched_binary_predicates=[query_relations],  # None indicates inference time
    #                 )
                
    #         # Step 3: Aggregate the object name and attribute predictions together
    #         aggr_res = {}            
            
    #         for oid in predicted_name:
    #             top_prob, name = predicted_name[oid][0]
    #             if not oid in aggr_res:
    #                 aggr_res[oid] = {}
                
    #             if name in target_names:
    #                 aggr_res[oid]['name'] = top_prob, name
    #             else:
    #                 # the top predicted name is not the target name
    #                 aggr_res[oid]['name'] = 0, "other"
                    
    #         for oid in reformatted_unary_res: 
                
    #             if not oid in aggr_res:
    #                 continue
                
    #             if not 'attr' in aggr_res[oid]:
    #                 aggr_res[oid]['attr'] = []
    #             for attr in reformatted_unary_res[oid]:
    #                 aggr_res[oid]['attr'].append((reformatted_unary_res[oid][attr], attr))
            
    #         if len(batched_image_rel_probs) > 0:
    #             rel_preds = batched_image_rel_probs[0]
    #         else:
    #             rel_preds = {}

    #         identified_rels = []
    #         for (_, (subj, obj), rel), p in rel_preds.items():
    #             subj_prob, pred_subj_name = predicted_name[subj][0]
    #             obj_prob, pred_obj_name = predicted_name[obj][0]
    #             rel_prob = p
    #             if (rel, pred_subj_name, pred_obj_name) in target_relation_preds:
    #                 prob = subj_prob * obj_prob * rel_prob
    #                 if prob > self.rel_thres:
    #                     identified_rels.append((prob, (rel, subj, obj), (rel, pred_subj_name, pred_obj_name)))
                
    #         # Thresholding
    #         avg_aggr_res = {}
    #         for oid in aggr_res:
    #             name_prob, name = aggr_res[oid]['name']
                
    #             # Target Name must rank at top 1
    #             if name_prob == 0:
    #                 continue
                
    #             # Attribute aggregation
    #             probs = [name_prob]

    #             if 'attr' in aggr_res[oid]:
    #                 probs += [p for p, _ in aggr_res[oid]['attr']]
                
    #             avg_val = sum(probs) / len(probs)
                    
    #             if avg_val > self.aggr_thres:
    #                 avg_aggr_res[oid] = avg_val
            
    #         # Relation aggregation
                
    #         # Only pick the top-k object according to the setup
    #         top_objects_probs = get_topk_per_object_name(aggr_res, avg_aggr_res, k=self.k)
            
    #         selected_objects = list(top_objects_probs.keys())
    #         # Handle pixel coordinate boxes (list of lists)
    #         selected_object_bboxes = [boxes[idx] for idx in selected_objects]
                                
    #         prob_labels = []
    #         for oid in top_objects_probs:
    #             prob, label = predicted_name[oid][0]
    #             prob_labels.append(f"{prob:.2f}::{label}")

    #     return prob_labels, selected_object_bboxes, identified_rels
    def predict_laser(self, 
                  image, 
                  boxes,  # Expects pixel coordinates as list of [x1, y1, x2, y2]
                  masks=None,  # Optional masks parameter
                  target_names=None,
                  query_names=None,
                  threshold=0.3,  # Probability threshold
                  ):
        prob_labels, selected_object_bboxes, selected_masks = [], [], []
        
        if not len(boxes) == 0 and self.predicate_model is not None:
            height, width = image.shape[0], image.shape[1]
            
            # Use provided masks or generate from bboxes
            if masks is not None and len(masks) > 0:
                batched_pred_masks = []
                for mask in masks:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 2:
                        mask = mask[:, :, np.newaxis]
                    batched_pred_masks.append(mask)
            else:
                batched_pred_masks = [bbox_to_mask(bbox, height, width) for bbox in boxes]
            
            batched_object_ids = [[0, 0, i] for i in range(len(batched_pred_masks))]
            
            # Get categorical predictions only
            batched_image_cate_probs, _, _, _ = \
                self.predicate_model(
                    batched_video_ids=[0],
                    batched_videos=[image],
                    batched_masks=batched_pred_masks,
                    batched_bboxes=boxes,
                    batched_names=[query_names],
                    batched_object_ids=batched_object_ids,
                    batched_unary_kws=[[]],
                    batched_binary_kws=[[]],
                    batched_obj_pairs=[],
                    batched_video_splits=[0],
                    batched_binary_predicates=[[]],
                )
            
            # Process each object
            for oid in range(len(boxes)):
                # Find the best matching category for this object
                best_prob = 0
                best_name = None
                
                for (obj_id, name) in batched_image_cate_probs[0]:
                    if obj_id == oid:
                        prob = batched_image_cate_probs[0][(obj_id, name)].item()
                        if prob > best_prob:
                            best_prob = prob
                            best_name = name
                
                # Check if this object matches our criteria with stricter filtering
                if best_name in target_names and best_prob > threshold:
                    # Additional filtering for GPT-4o: only keep high-confidence predictions
                    if best_prob > 0.7 or len(prob_labels) < 3:  # Keep high confidence or limit total objects
                        prob_labels.append(f"{best_prob:.2f}::{best_name}")
                        selected_object_bboxes.append(boxes[oid])
                        # Also append the corresponding mask
                        selected_masks.append(batched_pred_masks[oid])
        
        return prob_labels, selected_object_bboxes, selected_masks
    
    def predict_sam2_masks(self, image, boxes):
        """
        Generate SAM2 masks from bounding boxes
        Args:
            image: RGB image array
            boxes: tensor of bounding boxes [N, 4] in format [x1, y1, x2, y2]
        Returns:
            masks: list of binary masks
        """
        if not self.use_sam2 or self.sam2_predictor is None:
            return []
            
        if len(boxes) == 0:
            return []
            
        try:
            # Set image for SAM2 predictor
            self.sam2_predictor.set_image(np.array(image))
            
            # Convert boxes to numpy if needed
            if isinstance(boxes, torch.Tensor):
                boxes_np = boxes.cpu().numpy()
            else:
                boxes_np = np.array(boxes)
                
            # Generate masks for all boxes
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
            )
            
            # Squeeze masks if needed
            masks = [masks[i].squeeze(0) if masks[i].shape[0] == 1 else masks[i] for i in range(len(masks))]
            
            return masks
            
        except Exception as e:
            print(f"SAM2 prediction failed: {e}")
            return []
    
    def save_mask_contours_and_boxes(self, frame_image, masks, boxes, labels, save_path):
        """
        Save image with mask contours and bounding boxes
        """
        if len(masks) == 0 or len(boxes) == 0:
            return
            
        contour_img = frame_image.copy()
        
        # Draw mask contours
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Squeeze mask if needed
            if mask.ndim == 3:
                mask = mask.squeeze()
                
            segmentation = mask.astype(np.uint8)
            contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Use thin red line for better visibility
            cv2.drawContours(contour_img, contours, -1, color=(255, 0, 0), thickness=1)
        
        # Draw labels
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            x0, y0, x1, y1 = map(int, box)
            
            # Draw label with subtle shadow for visibility
            # Shadow
            cv2.putText(contour_img, label, (x0+1, y0-4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            # Main text
            cv2.putText(contour_img, label, (x0, y0-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save result
        cv2.imwrite(save_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))
    
    def get_sg_kws(self, sg_info):
        """Extract scene graph keywords with robust error handling"""
        
        # Provide default empty lists for missing keys
        target_objects = sg_info.get('target_objects', [])
        target_attributes = sg_info.get('target_attributes', [])   
        target_relations = sg_info.get('target_relations', [])
        current_objects = sg_info.get('current_objects', [])
        current_attributes = sg_info.get('current_attributes', [])
        current_relations = sg_info.get('current_relations', [])
        
        # Handle case where values might be None
        target_objects = target_objects if target_objects is not None else []
        target_attributes = target_attributes if target_attributes is not None else []
        target_relations = target_relations if target_relations is not None else []
        current_objects = current_objects if current_objects is not None else []
        current_attributes = current_attributes if current_attributes is not None else []
        current_relations = current_relations if current_relations is not None else []
        
        target_names = list(set([clean_name(object) for object in target_objects if object]))
        current_names = list(set([clean_name(object) for object in current_objects if object]))
        target_attribute_kws = list(set([ta[1] for ta in target_attributes if len(ta) == 2 and ta[1]]))
        current_attribute_kws = list(set([ca[1] for ca in current_attributes if len(ca) == 2 and ca[1]]))

        target_relation_preds = set()
        current_relation_preds = set()
        target_relation_kws = set()
        
        # Process target relations with error handling
        for rel in target_relations:
            try:
                if not isinstance(rel, (list, tuple)) or len(rel) != 3:
                    continue
                (s, r, o) = rel
                if s and r and o:  # Ensure none are None or empty
                    s = clean_name(str(s))
                    o = clean_name(str(o))
                    target_relation_preds.add((str(r), s, o))
                    target_relation_kws.add(str(r))
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Error processing target relation {rel}: {e}")
                continue
            
        # Process current relations with error handling    
        for rel in current_relations:
            try:
                if not isinstance(rel, (list, tuple)) or len(rel) != 3:
                    continue
                (s, r, o) = rel
                if s and r and o:  # Ensure none are None or empty
                    s = clean_name(str(s))
                    o = clean_name(str(o))
                    current_relation_preds.add((str(r), s, o))
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Error processing current relation {rel}: {e}")
                continue

        query_objects = list(set(target_names + current_names + self.common_objects)) 
        query_relation_preds = list(target_relation_preds.union(current_relation_preds))
        query_other_preds = list(set([("other", sub, obj) for (_, sub, obj) in query_relation_preds]))
        query_relation_preds = query_relation_preds + query_other_preds
        query_relation_kws = list(set([r for (r, _, _) in query_relation_preds]))
        
        return target_names, current_names, query_objects, \
               target_attribute_kws, current_attribute_kws, \
               target_relation_preds, current_relation_preds, \
               target_relation_kws, query_relation_kws, query_relation_preds

    def act_custom(self, prompt, obs):
        assert type(obs) == str  # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
        out = out.replace("'", '"')
        out = out.replace('"s ', "'s ")
        out = out.replace('"re ', "'re ")
        out = out.replace('"ll ', "'ll ")
        out = out.replace('"t ', "'t ")
        out = out.replace('"d ', "'d ")
        out = out.replace('"m ', "'m ")
        out = out.replace('"ve ', "'ve ")
        out = out.replace("```json", "").replace("```", "")
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action_alt(out)
        self.planner_steps += 1
        return action, out

    def act(self, observation, user_instruction):

        if type(observation) == dict:
            obs = copy.deepcopy(observation[self.obs_key])
        else:
            # assert len(observation) == 1
            obs = copy.deepcopy(observation)  # input image path

        prompt, scene_graph_prompt = self.process_prompt(
            user_instruction, prev_act_feedback=self.episode_act_feedback
        )
        # some models do not support json scheme, add style into prompt
        if (
            "claude" in self.model_name
            or "InternVL" in self.model_name
            or "Qwen" in self.model_name
            or self.model_type == "custom"
        ):
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == "custom":
            return self.act_custom(prompt, obs)

        sg_message = self.get_message(obs, scene_graph_prompt)

        try:
            out = self.model.respond(sg_message, get_scene_graph=True)
        except Exception as e:
            print(f"Scene graph generation failed: {e}")
            if "qwen" in self.model_name:
                return (
                    -2,
                    """{
                    "visual_state_description": "qwen model generate empty action due to inappropriate content check",
                    "reasoning_and_reflection": "invalid json, random action",
                    "language_plan": "invalid json, random action"
                }""",
                )
            else:
                # For other models, create a fallback response
                out = """{
                    "target_state": "Unable to generate scene graph due to API error",
                    "current_state": "Scene graph generation failed",
                    "target_objects": [],
                    "current_objects": [],
                    "target_attributes": [],
                    "current_attributes": [],
                    "target_relations": [],
                    "current_relations": [],
                    "explanation": "API error occurred during scene graph generation"
                }"""

        # Parse scene graph JSON with error handling
        try:
            sg_info = robust_load_json_full(out)
            # Validate that sg_info is a dictionary
            if not isinstance(sg_info, dict):
                raise ValueError(f"Scene graph output is not a dictionary: {type(sg_info)}")
            
            # Print parsed scene graph for debugging
            print(f"Parsed scene graph keys: {list(sg_info.keys())}")
            print(f"Full scene graph content: {sg_info}")  # Show complete content for debugging
            
            # Handle different key formats from VLM and ensure all required keys exist
            key_mappings = {
                'target_objects': ['target_objects', 'target_name', 'target_object'],
                'target_attributes': ['target_attributes', 'target_attribute'],
                'target_relations': ['target_relations', 'target_relation'],
                'current_objects': ['current_objects', 'related_objects', 'current_object'],
                'current_attributes': ['current_attributes', 'current_attribute'],
                'current_relations': ['current_relations', 'current_relation']
            }
            
            for expected_key, possible_keys in key_mappings.items():
                if expected_key not in sg_info:
                    # Try to find the value under alternative key names
                    found_value = None
                    for alt_key in possible_keys:
                        if alt_key in sg_info:
                            found_value = sg_info[alt_key]
                            print(f"Found '{alt_key}' instead of '{expected_key}', mapping it")
                            break
                    
                    if found_value is not None:
                        # Ensure the value is a list (VLM might return single strings)
                        if isinstance(found_value, str):
                            sg_info[expected_key] = [found_value]
                        elif isinstance(found_value, list):
                            sg_info[expected_key] = found_value
                        else:
                            print(f"Unexpected type for '{expected_key}': {type(found_value)}, converting to list")
                            sg_info[expected_key] = [str(found_value)]
                    else:
                        print(f"Missing key '{expected_key}' in scene graph, using empty list")
                        sg_info[expected_key] = []
            
        except Exception as e:
            print(f"Scene graph JSON parsing failed: {e}")
            if 'out' in locals():
                print(f"Raw VLM output: {out[:500]}...")  # Show first 500 chars for debugging
            else:
                print("No VLM output available for debugging")
            # Use default scene graph structure if parsing fails
            sg_info = {
                "target_state": "Unable to parse scene graph",
                "current_state": "Unable to parse scene graph", 
                "target_objects": [],
                "target_attributes": [],
                "target_relations": [],
                "current_objects": [],
                "current_attributes": [],
                "current_relations": [],
                "explanation": "JSON parsing failed, using default structure"
            }

        target_state = sg_info.get("target_state", "No target state available")
        current_state = sg_info.get("current_state", "No current state available")
        try:
            target_names, current_names, query_objects, \
            target_attribute_kws, current_attribute_kws, \
            target_relation_preds, current_relation_preds, \
                target_relation_kws, query_relation_kws, query_relation_preds = self.get_sg_kws(sg_info)
            print(f"Successfully extracted scene graph keywords: {len(target_names)} target objects, {len(current_names)} current objects")
        except Exception as e:
            print(f"Error processing scene graph keywords: {e}")
            print(f"Scene graph info structure: {sg_info}")
            # Use empty defaults if scene graph processing fails
            target_names = []
            current_names = []
            query_objects = self.common_objects.copy()
            target_attribute_kws = []
            current_attribute_kws = []
            target_relation_preds = set()
            current_relation_preds = set()
            target_relation_kws = set()
            query_relation_kws = []
            query_relation_preds = []
        scene_graph_prompt = []
        sg_text = ""
         
        if type(obs) == str:
            obs = [obs]
        if type(observation) == str:
            observation = [observation]

        for image_id, image_path in enumerate(observation):
            image_path_ls = image_path.split("/")
            image_dir, image_name = image_path_ls[:-1], image_path_ls[-1]
            image_name = image_name.split(".")[0]
            bbox_file_name = image_name + '_bbox.png'

            bbox_path = os.path.join(*image_dir, bbox_file_name)

            prompt_image_path = image_path

            if image_path in self.processed_images:
                processed_image_path = self.processed_images[image_path]
                if not processed_image_path in obs:
                    obs = [processed_image_path]
                    continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[0], image.shape[1]
            
            if self.gd_only:
                
                # Predict bounding box with grounding dino
                labels, boxes = self.predict_gd(image, target_names, box_threshold=self.gd_box_threshold, text_threshold=self.gd_text_threshold)
            
                target_labels, target_bboxes = [], []
                for label, box in zip(labels, boxes):
                    if label in target_names:
                        target_labels.append(label)
                        target_bboxes.append(box)
                        
                if len(target_bboxes) > 0:
                    target_bboxes = torch.stack(target_bboxes)
            
            else:
                # og_laels, og_boxes = self.predict_gd(image, gd_query_objects, box_threshold=0.4, text_threshold=0.4)
                labels, boxes = self.predict_gd(image, query_objects,
                                box_threshold=self.gd_box_threshold, 
                                text_threshold=self.gd_text_threshold)

                if len(boxes) != 0:
                    # Get correct image dimensions
                    img_height, img_width = image.shape[0], image.shape[1]
                    
                    # Convert boxes to pixel coordinates for LASER
                    boxes_pixel = convert_gd_boxes_to_pixel(boxes, img_width, img_height)
                    
                    # Generate SAM2 masks if enabled (SAM2 expects pixel coordinates)
                    sam2_masks = []
                    if self.use_sam2 and len(boxes_pixel) > 0:
                        # Convert to tensor for SAM2
                        boxes_pixel_tensor = torch.tensor(boxes_pixel, dtype=torch.float32)
                        sam2_masks = self.predict_sam2_masks(image, boxes_pixel_tensor)
                    
                    # # Call predict_laser with pixel coordinates and masks
                    # target_labels, target_bboxes, identified_rels = self.predict_laser(
                    #     image=image, 
                    #     boxes=boxes_pixel,  # Pixel coordinates as list
                    #     masks=sam2_masks if sam2_masks else None,  # Pass masks if available
                    #     target_names=target_names,
                    #     target_relation_kws=target_relation_kws,
                    #     target_relation_preds=target_relation_preds,
                    #     query_names=list(set(query_objects)),
                    #     query_attributes=target_attribute_kws,
                    #     query_relation_kws=query_relation_kws,
                    #     query_relations=query_relation_preds,
                    # )
                    # Call predict_laser with pixel coordinates and masks
                    target_labels, target_bboxes, identified_masks = self.predict_laser(
                        image=image, 
                        boxes=boxes_pixel,  # Pixel coordinates as list
                        masks=sam2_masks if sam2_masks else None,  # Pass masks if available
                        target_names=target_names,
                        query_names=list(set(query_objects)),  # All possible object names to check
                        threshold=self.aggr_thres  # Use the existing threshold from self
                    )
                else:
                    target_labels = []
                    target_bboxes = []
                    identified_rels = []

                # 4. Update visualization part:
                # Save visualization with boxes and optionally masks
                if not len(target_bboxes) == 0:
                    # Convert list of lists to tensor for visualization functions
                    target_bboxes_tensor = torch.tensor(target_bboxes, dtype=torch.float32)
                    
                    # Use improved visualization for GPT-4o
                    from embodiedbench.planner.improved_bbox_viz import save_improved_bbox_image, extract_confidence_from_labels
                    
                    if self.use_sam2 and len(sam2_masks) > 0:
                        # Save with both masks and boxes
                        mask_bbox_path = bbox_path.replace('_bbox.png', '_mask_bbox.png')
                        self.save_mask_contours_and_boxes(
                            frame_image=image, 
                            masks=identified_masks,
                            boxes=target_bboxes_tensor, 
                            labels=target_labels,
                            save_path=mask_bbox_path
                        )
                        prompt_image_path = mask_bbox_path
                    else:
                        # Use improved visualization with confidence scores
                        clean_labels, confidences = extract_confidence_from_labels(target_labels)
                        save_improved_bbox_image(
                            frame_image=image,
                            boxes=target_bboxes,
                            labels=clean_labels,
                            confidence_scores=confidences,
                            save_path=bbox_path
                        )
                        prompt_image_path = bbox_path
                
                else:
                    target_bboxes = []
                    target_labels = []
            
            self.processed_images[image_path] = prompt_image_path

            # Arrange the prompt into vlm
            if self.scene_graph_text:
                try:
                    sg_text = format_state_descriptions(sg_info, target_labels, [], [])
                    print(f"Generated scene graph text: {sg_text[:200]}...")
                except Exception as e:
                    print(f"Error formatting scene graph text: {e}")
                    # Fallback to simple state description
                    sg_text = f"\n\n## Scene Context:\n**Target:** {target_state}\n**Current:** {current_state}\n"
                    if target_labels and len(target_labels) <= 3:
                        clean_labels = [label.split("::")[-1] if "::" in label else label for label in target_labels]
                        sg_text += f"**Detected Objects:** {', '.join(clean_labels)}\n"
        
        if self.multistep:
            if not image_path == prompt_image_path:
                obs.append(prompt_image_path) # update to bboxed image path
        
        # Add fallback logic for poor detection quality
        use_scene_graph = self.scene_graph_text and sg_text is not None
        if hasattr(self, 'config') and self.config.get('fallback_to_original', False):
            # Check detection quality - if poor, skip scene graph enhancement
            if len(target_labels) == 0 or (len(target_labels) == 1 and any(conf < 0.5 for conf in [0.5])):  # Placeholder confidence check
                use_scene_graph = False
                print("Poor detection quality - falling back to original VLM mode")
        
        if use_scene_graph:
            marker = "Now the human instruction is"
            try:
                insert_idx = prompt.index(marker)      # first char of the marker
                prompt = prompt[:insert_idx] + sg_text + prompt[insert_idx:]
            except ValueError:
                # marker not found – fallback: append at the end
                prompt += sg_text
            
        
        else:
            obs = [prompt_image_path]
            
        obs = sorted(list(set(obs)))
        
        if not self.multistep:
            assert len(obs) == 1
            obs = obs[0]

        if len(self.episode_messages) == 0:
            self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(
                    obs, prompt, self.episode_messages
                )
            else:
                self.episode_messages = self.get_message(obs, prompt)

        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if "gemini-1.5-pro" in self.model_name or "gemini-2.0-flash" in self.model_name:
            try:
                out = self.model.respond(self.episode_messages)
                time.sleep(10)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(20)
                out = self.model.respond(self.episode_messages)
        else:
            try:
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != "local":
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        logger.debug(f"Model Output:\n{out}\n")

        if self.chat_history:
            self.episode_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": out}],
                }
            )
        
        # Clear processed images cache to force re-analysis after each step
        # This ensures the robot gets fresh visual input and doesn't reuse stale scene descriptions
        self.processed_images.clear()
        # out = out.replace("'", '"')
        out = out.replace('"s ', "'s ")
        out = out.replace('"re ', "'re ")
        out = out.replace('"ll ', "'ll ")
        out = out.replace('"t ', "'t ")
        out = out.replace('"d ', "'d ")
        out = out.replace('"m ', "'m ")
        out = out.replace('"ve ', "'ve ")
        out = out.replace("```json", "").replace("```", "")
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action, valid = self.json_to_action_alt(out)
        
        if valid:
            try:
                def parse_json_with_fixes(json_str):
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try fixing common escape issues
                        fixed_str = json_str.replace("\\'", "'")
                        return json.loads(fixed_str) 
                plan = parse_json_with_fixes(out)
                self.last_reasoning = plan.get(
                        "reasoning_and_reflection", ""
                )
                self.last_visual_description = plan.get(
                        "visual_state_description", ""
                )
                
                plan['scene_graph'] = sg_info
                out_str = json.dumps(plan)
            except Exception as e:
                print(f"Error parsing final JSON output: {e}")
                self.last_visual_description = ""
                self.last_reasoning = ""
                out_str = '''{"visual_state_description":"json parsing failed", "reasoning_and_reflection":"json parsing failed",
                       "language_plan":"json parsing failed"}'''
        else:
            self.last_visual_description = ""
            self.last_reasoning = ""
            out_str = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
                   
        self.planner_steps += 1
        return action, out_str

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([info["action_id"], info["env_feedback"]])