import os
import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, ValidEvalSets
from embodiedbench.planner.vlm_planner_esca import VLMPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from embodiedbench.evaluator.config.system_prompts import alfred_laser_system_prompt
from embodiedbench.main import logger
from alfred_examples import laser_sg_examples_pre_post, laser_sg_examples_no_revert

import os

example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_examples.json')

#common sense, base, and ...
target_examples = [
'''
   The action history:
   Step 0, action id 10, find a spray bottle, env feedback: Last action executed successfully.
   Step 1, action id 90, pick up the spray bottle, env feedback: Last action executed successfully.
   Step 2, action id 17, find a Toilet, env feedback: Last action executed successfully.
   Step 3, action id 133, put down the object in hand,, env feedback: Last action executed successfully.
   Step 4, action id 10, find a spray bottle, env feedback: Last action executed successfully.
   Step 5, action id 90, pick up the spray bottle, env feedback: Last action executed successfully.
   The last action succeeds.
   
   Environmental Feedback of last action: The last action succeed. 
         
   Human Instruction: Place two yellow bottles on top of a toilet.
      
   Target state and current state in JSON: {
        "target_state": "There are two yellow bottles on the top of a toilet."
        "target_objects": ["bottle_1", "bottle_2", "toilet_1"],
        "target_attributes": [["yellow", "bottle_1"], ["yellow", "bottle_2"]]
        "target_relations": [["bottle_1", "on", "toilet_1"], ["bottle_2", "on", "toilet_1"]], 
        "current_state": "There are two pink bottles and a toilet in the scene. One pink bottle is on the top of the toilet. I am holding another pink bottle.",
        "explanation": "I have pick up the spray bottle in step 1, this is the pink bottle on the toilet. I have pick up the spray bottle in step 5, this is the pink bottle I am holding right now. From the possible actions, I may try the soap bottle, wine bottle and glass bottle instead. ",
        "current_objects": ["bottle_1", "bottle_2" "toilet_1"],
        "current_attributes": [["pink", "bottle_1"], ["pink", "bottle_2"]]
        "current_relations": [["bottle_1", "on", "toilet_1"], ["bottle_2", "in front of", "toilet_1"]], 
        
        }
''',

''' 
    The action history:
    Step 1, action id 109, pick up the Ladle, env feedback: Last action executed successfully.
    Step 2, action id 2, find a Faucet, env feedback: Last action executed successfully.
    Step 3, action id 155, turn on the Faucet, env feedback: Last action executed successfully.
    Step 4, action id 156, turn off the Faucet, env feedback: Last action executed successfully.
    Step 5, action id 39, find a DiningTable, env feedback: Last action executed successfully.
    Step 6, action id 133, put down the object in hand, env feedback: Last action executed successfully.
    The last action succeeds. 

    Environmental Feedback: Last action executed successfully.
    
    Human Instruction: Rinse off something for serving soup and move it to the table.
        
    Target state and current state in JSON: {
        "target_state": "There is a rinsed ladle on a table."
        "target_objects": ["ladle_1", "table_1"],
        "target_attributes": [["rinsed", "ladle_1"]]
        "target_relations": [["ladle_1", "on", "table_1"]], 
        "current_state": "There is a bread,  an egg, a ladle not rinsed and a table. "
        "explanation": "Although I can see a ladle on the table, the task has not terminated. This indicates some criteria in the target state is not met. I need to check the steps in detail to verify. In Step 2,3,4, although I have opened and closed the Faucet, I have not put the ladle in water. It means the Ladle is not rinsed. ",
        "current_objects": ["ladle_1", "table_1", "egg_1", "bread_1"],
        "current_attributes": [["not rinsed", "ladle_1"]]
        "current_relations": [["ladle_1", "on", "table_1"], ["egg_1", "on", "table_1"], ["bread_1", "on", "table_1"]], 
        }
''', 

'''
   The action history:
    Step 1, action id 19, find a Spoon, env feedback: Last action executed successfully.
    Step 2, action id 105, pick up the Spoon, env feedback: Last action executed successfully.
    Step 3, action id 31, find a Plate, env feedback: Last action executed successfully.
    Step 4, action id 122, pick up the Plate, env feedback: Last action is invalid. Robot is currently holding Spoon.
    
   Human Instruction: Set plate with a spoon in it on the kitchen table.
      
   Target state and current state in JSON: {
        "target_state": "There is a spoon on a plate. The plate is on a kitchen table."
        "target_objects": ["spoon_1", "plate_1", "kitchen_table_1"],
        "target_attributes": []
        "target_relations": [["spoon_1", "in", "plate_1"], ["plate_1", "on", "kitchen_table_1"]], 
        "current_state": "I am currently holding a plate. There is a spoon and an apple on the counter. There is a pan on the sink. "
        "explanation": "My action to pick up the plate is failed. I need to have a detailed look into the action history and preconditions of each action. In step 2 and step 4, I tried to pick up two things, spoon and the plate in a row. We know that the action, pickup, requires holding nothing at hand, but I am holding a plate currently. While in the target state, this plate should locate at the kitchen table at the end. ",
        "current_objects": ["plate_1", "spoon_1", "apple_1", "counter_1", "sink_1", "pan_1"],
        "current_attributes": [["holding", "plate_1"]]
        "current_relations": [["spoon_1", "on", "counter_1"], ["apple_1", "on", "counter_1"], ["pan_1", "in", "sink_1"]], 
        }
'''
]

    
exploration_example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_long_horizon_examples.json')

system_prompt = alfred_laser_system_prompt

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
                
class EB_AlfredEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None
        # self.gd_checkpoint_path = config['gd_checkpoint_path']

        # self.gd_config_path = config['gd_config_path']
        
        self.gd_checkpoint_path= "/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth"
        self.gd_config_path = "/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        
        # Add SAM2 configuration
        self.sam2_config_path = config.get('sam2_config_path', '')
        self.sam2_checkpoint_path = config.get('sam2_checkpoint_path', '')
        self.use_sam2 = config.get('use_sam2', False)

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets

        print(valid_eval_sets)
        
        
        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
                        
            self.env = EBAlfEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                          exp_name=exp_name, selected_indexes=self.config.get('selected_indexes', []), 
                                          detection_box=False,
                                          resolution=self.config.get('resolution', 500), 
                                          )
            examples = json.load(open(example_path, 'r+')) if self.eval_set != 'long_horizon' else json.load(open(exploration_example_path, 'r+'))
            
            if self.config['sg_text']:
                few_shot_examples = laser_sg_examples_no_revert
            else:
                few_shot_examples = examples
            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, system_prompt, examples=few_shot_examples, scene_graph_examples=target_examples, n_shot=self.config['n_shots'], 
                                            obs_key='head_rgb', chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                                            use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), 
                                            boundingbox=self.config['detection_box'], scene_graph_text=self.config['sg_text'],
                                            gd_only=self.config['gd_only'], tp=self.config['tp'],
                                            blockingobj=self.config['block_objects'],
                                            gd_checkpoint_path=self.gd_checkpoint_path, gd_config_path=self.gd_config_path,
                                            top_k=self.config['top_k'], aggr_thres=self.config['aggr_thres'], 
                                            gd_box_threshold=self.config['gd_box_threshold'], gd_text_threshold=self.config['gd_text_threshold'],
                                            # Add SAM2 parameters
                                            use_sam2=self.use_sam2,
                                            sam2_config_path=self.sam2_config_path,
                                            sam2_checkpoint_path=self.sam2_checkpoint_path)

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            current_frames = [obs['head_rgb']]
            test_ct = 0
            
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            # update the action space for alfred due to dynamic objects
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            while not done:
                # try: 
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")
                    if action == -2: # empty plan stop here
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        break 
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    
                    # mutiple actions
                    if type(action) == list:
                        current_frames = []
                        # Limit plan length for long horizon tasks
                        max_plan_length = self.config.get('max_plan_length', 10)
                        if len(action) > max_plan_length:
                            print(f"Plan too long ({len(action)} steps), truncating to {max_plan_length}")
                            action = action[:max_plan_length]
                            
                        for action_single in action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]:
                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            current_frames.append(obs['head_rgb'])
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            print(f"Last action success: {info['last_action_success']}")
                            print(f"Environmental feedback: {info['env_feedback']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            if done or not info['last_action_success']:
                                # stop or replanning
                                print("Invalid action or task complete. If invalid then Replanning.")
                                break
                    else: # single action
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        current_frames = [obs['head_rgb']]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        print(f"Last action success: {info['last_action_success']}")
                        print(f"Environmental feedback: {info['env_feedback']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                # except Exception as e: 
                #     print(e)
                #     test_ct += 1
                #     if test_ct > self.config['max_test_ct']:
                #         if not 'info' in locals():
                #             info = {}
                #             info['task_success'] = False
                #             info['task_progress'] = 0
                #             info["env_step"] = 30
                #             info["episode_elapsed_seconds"] = 1000
                #         break
                #     time.sleep(10)

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)

            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            progress_bar.update()


if __name__ == '__main__':
    import argparse
    # def parse_arguments():
    #     parser = argparse.ArgumentParser(description='Change configuration parameters.')
    #     parser.add_argument('--model_name', default='Qwen/Qwen2.5-VL-72B-Instruct', type=str, help='Name of the model.')
    #     parser.add_argument('--n_shots', type=int, help='Number of examples')
    #     parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
    #     parser.add_argument('--model_type', default='remote', type=str, help='Type of the model.')
    #     parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
    #     parser.add_argument('--exp_name', default='gd_laser', type=str, help='Name of the experiment.')
    #     parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
    #     parser.add_argument('--detection_box', type=int, help='Set to True to enable detection.')
    #     parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
    #     parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
    #     parser.add_argument('--resolution', default=500, type=int, help='Resolution for processing.')
    #     parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
    #     parser.add_argument('--tp', default=4, type=int, help='number of tensor parallel splits of the model parameters')
    #     return parser.parse_args()


    config = {
        'model_name': 'gpt-4o',  # 'Qwen/Qwen2-VL-7B-Instruct',
        'n_shots': 10,
        'down_sample_ratio': 1.0,
        'model_type': 'remote', # 'local', 
        'language_only': 0,
        'exp_name': 'gpt_4o_esca_integration_final_no_sam_NEW_FINAL_spatial',
        'chat_history': 0, 
        'detection_box': True,
        'eval_sets': ['spatial'],
        # 'eval_sets': ['visual_appearance'],
        # 'eval_sets': ['long_horizon', 'common_sense', 'visual_appearance'],
        'selected_indexes': [], 
        # 'eval_sets': ['base'],
        # 'selected_indexes': list(range(5, 10)) + list(range(40,50)),
        'multistep':0, 
        'block_objects': False,
        'sg_text': True,
        'resolution': 500, 
        'env_feedback': 1,
        'tp': 4,
        'action_num_per_plan': 5,
        'fov': 100,
        'sleep_time': 0,
        'purpose': "retest",
        'icl_abl':0, 
        'visual': 0,
        'gd_only': False,
        'top_k': 1, 
        'aggr_thres': 0.75,
        'fallback_to_original': True,  # Fallback to original VLM if detection quality is poor
        'max_plan_length': 20,  # Limit plan length for long horizon tasks
        # 'gd_config_path': '/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        # 'gd_checkpoint_path': '/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth',
        'gd_box_threshold': 0.6,  # Increase to reduce false positives
        'gd_text_threshold': 0.5,  # Increase for more confident detections
        'max_test_ct': 5,
        # SAM2 configuration
        'use_sam2': False,  # Set to True to enable SAM2
        'sam2_config_path': "configs/sam2.1/sam2.1_hiera_b+.yaml",
        'sam2_checkpoint_path': "/home/asethi04/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    }

    # args = parse_arguments()
    # update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()




