import os
import re
import cv2
from collections import defaultdict
from glob import glob

def generate_videos_from_pngs(input_dir, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)
    
    pattern = re.compile(r"episode_(\d+)_step_(\d+)_([^_]+)_rgb\.png")
    frames_dict = defaultdict(list)

    for file_path in glob(os.path.join(input_dir, "*.png")):
        file_name = os.path.basename(file_path)
        match = pattern.match(file_name)
        if match:
            episode = int(match.group(1))
            step = int(match.group(2))
            config = match.group(3)
            frames_dict[(episode, config)].append((step, file_path))

    for (episode, config), step_files in frames_dict.items():
        step_files.sort()
        if not step_files:
            continue

        first_img = cv2.imread(step_files[0][1])
        height, width, _ = first_img.shape

        output_path = os.path.join(output_dir, f"episode_{episode}_{config}_rgb.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for _, img_path in step_files:
            img = cv2.imread(img_path)
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))  # safeguard
            writer.write(img)
        writer.release()
    print(f"Saved videos to: {output_dir}")


def generate_all_episode_videos(data_dir, fps=10):
    images_root = os.path.join(data_dir, "images")
    videos_root = os.path.join(data_dir, "videos")
    os.makedirs(videos_root, exist_ok=True)

    for episode_folder in sorted(os.listdir(images_root)):
        episode_path = os.path.join(images_root, episode_folder)
        if os.path.isdir(episode_path) and episode_folder.startswith("episode_"):
            output_path = os.path.join(videos_root, episode_folder)
            print(f"Processing {episode_path} → {output_path}")
            generate_videos_from_pngs(episode_path, output_path, fps=fps)
    print("All episodes processed.")


if __name__ == "__main__":
    data_dir = "/home/jianih/research/EmbodiedBench/running/eb_manipulation/pi0_droid/n_shot=10_resolution=500_detection_box=0_multiview=1_multistep=0_visual_icl=0/base"
    generate_all_episode_videos(data_dir, fps=10)
    print("Done.")
