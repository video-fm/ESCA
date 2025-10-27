import re
import os
import numpy as np
from tqdm import tqdm
import json
from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv, ValidEvalSets
from embodiedbench.planner.nav_planner_gd import EBNavigationPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
import sys
import warnings
import cv2

from time import sleep

from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt
from embodiedbench.evaluator.config.eb_navigation_example import examples, scene_graph_examples
from embodiedbench.main import logger

system_prompt = eb_navigation_system_prompt
examples = examples

class EB_NavigationEvaluator():
    def __init__(self, config):

        self.model_name = config['model_name']
        self.eval_sets = config["eval_sets"]
        self.gd_checkpoint_path = config['gd_checkpoint_path']
        self.gd_config_path = config['gd_config_path']
        self.eval_set = None
        self.config = config

        # print("Config: ")
        # print(self.config)
        # print("end config")
        
        
        self.env = None
        self.planner = None

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
        self.eval_sets = list(valid_eval_sets)
        if type(self.eval_sets) == list and len(self.eval_sets) == 0:
            self.eval_sets = ValidEvalSets
            
        for eval_set in self.eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"

            # if eval_set == "visual_appearance":
            #     selected_indexes = list(range(22, 60))
            # else: 
            #     selected_indexes = None
                
            self.env = EBNavigationEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                   exp_name=exp_name, multiview=self.config['multiview'], boundingbox=self.config['detection_box'], 
                                   multistep = self.config['multistep'], resolution = self.config['resolution'],
                                   )

            self.planner = EBNavigationPlanner(model_name=self.model_name, model_type = self.config['model_type'], 
                                           actions = self.env.language_skill_set, system_prompt = system_prompt, 
                                           examples = examples, scene_graph_examples=scene_graph_examples,
                                           n_shot=self.config['n_shots'], obs_key='head_rgb', 
                                           chat_history=self.config['chat_history'], language_only=self.config['language_only'], 
                                           multiview=self.config['multiview'], multistep = self.config['multistep'], visual_icl = self.config['visual_icl'],
                                           gd_checkpoint_path=self.gd_checkpoint_path, gd_config_path=self.gd_config_path)
            
            self.evaluate()
            
            average_json_values(os.path.join(self.env.log_path, 'results'), selected_key = None)
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': []}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")
            self.planner.reset()
            done = False
            while not done:
                try:
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")
                    reasoning = json.loads(reasoning)
                    if type(action) == list:
                        
                        for i, action_single in enumerate( action[:min(self.env._max_episode_steps - self.env._current_step + 1, len(action))] ):
                            if i==0:
                                obs, reward, done, info = self.env.step(action_single,reasoning,1)
                            else:
                                obs, reward, done, info = self.env.step(action_single,reasoning,0)
                            
                            print(f"Executed action: {action_single}, Task success: {info['task_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                                                        
                            episode_info['reward'].append(reward)

                            if done==True:
                                break

                            if info['last_action_success'] == 0:
                                # stop for replanning
                                print('invalid action, start replanning')
                                break
                    else:
                        obs, reward, done, info = self.env.step(action, reasoning, 1)
                        
                        print(f"Executed action: {action}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)

                except Exception as e:
                    sleep(1)
                    print(e)
                    print("retrying...")

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            # episode_info["task_progress"] = info['task_progress']
            # episode_info['subgoal_reward'] = info['subgoal_reward']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            # episode_info["num_invalid_actions"] = info["num_invalid_actions"]
            # episode_info["num_invalid_action_ratio"] = info["num_invalid_actions"] / info["env_step"]
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]
            self.save_episode_metric(episode_info)
            progress_bar.update()

    def check_config_valid(self):
        if self.config['multiview'] + self.config['multistep'] + self.config['visual_icl'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multiview, multistep, visual_icl, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multiview'] or self.config['multistep']:
                logger.warning("Language only mode should not have multiview or multistep enabled. Setting these arguments to False ...")
                self.config['multiview'] = 0
                self.config['multistep'] = 0

def save_video_from_rgb_images(images, output_path, fps=30):
    """
    Save a list of RGB images to an MP4 video.

    Args:
        images (List[np.ndarray]): List of images in RGB format (H x W x 3).
        output_path (str): Path to save the output MP4 video.
        fps (int): Frames per second.
    """
    if not images:
        raise ValueError("Image list is empty")

    # Get height and width from first image
    height, width, _ = images[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' or 'H264'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        # Convert RGB (default from many libraries) to BGR (OpenCV default)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(bgr_img)

    out.release()
    print(f"Video saved to {output_path}")
    
if __name__ == '__main__':
    
    # config = {
    #     'model_name': sys.argv[2],  #'gpt-4o-mini', claude-3-5-sonnet-20241022, sys.argv[2]
    #     # 'model_name': "OpenGVLab/InternVL2_5-38B",
    #     'down_sample_ratio': 1, # 0.000666,
    #     'model_type': 'remote',
    #     'language_only': False,
    #     'dataset': sys.argv[1],
    #     'chat_history': True, 
    #     'action_num_per_plan': 5,
    #     'fov': 100,
    #     'n_shots' : int(sys.argv[4]),   # int(sys.argv[3])
    #     'sleep_time':  0, #int(sys.argv[3]),
    #     'multiview': 0, #sys.argv[3]=='1',
    #     'boundingbox': 0, #sys.argv[4]=='1',
    #     'target_only': 0, #sys.argv[5]=='1',
    #     'multistep':0, #sys.argv[6]=='1',
    #     'resolution': 500, #int(sys.argv[7]),
    #     'purpose': "retest", #sys.argv[8],
    #     'exp_name': sys.argv[3],
    #     'icl_abl':0, #sys.argv[10]=='1',
    #     'visual':0 #sys.argv[11]=='1',
    # }
    
    config = {'model_name': 'OpenGVLab/InternVL2_5-38B', 
              'down_sample_ratio': 1, 
              'model_type': 'remote', 
              'language_only': False, 
              'eval_sets': ['long_horizon'], 
              'chat_history': False, 
              'n_shots': 3, 
              'multiview': False, 
              'detection_box': False, 
              'multistep': True, 
              'resolution': 500, 
              'exp_name': 'multistep_baseline_gd_two_imgs', 
              'visual_icl': False, 
              'tp': 4, 
              'gd_config_path': '/home/jianih/research/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
              'gd_checkpoint_path': '/home/jianih/research/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth',}
    
    evaluator = EB_NavigationEvaluator(config)
    evaluator.evaluate_main()