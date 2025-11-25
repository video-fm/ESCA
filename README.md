<div align="center">
  
  
  <h1><img src="docs/images/esca_logo.png" alt="ESCA Logo" width="45"/> ESCA: Contextualizing Embodied Agents via Scene-Graph Generation</h1>

  [![Paper](https://img.shields.io/badge/arXiv-ESCA_paper-b31b1b.svg)](https://arxiv.org/abs/2304.07647)
  [![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-ESCA--video--87K-yellow)](https://huggingface.co/datasets/video-fm/ESCA-video-87K)
  [![Model](https://img.shields.io/badge/🤗%20HuggingFace-SGCLIP--model-green)](https://huggingface.co/video-fm/vine_v0)
  [![Code](https://img.shields.io/badge/GitHub-LASER_code-blue?logo=github)](https://github.com/video-fm/ESCA)


[Jiani Huang](https://www.cis.upenn.edu/~jianih/) • [Amish Sethi](https://amishsethi.github.io/personal-website/) • [Matthew Kuo](https://www.linkedin.com/in/kuomat/) • [Mayank Keoliya](https://www.seas.upenn.edu/~mkeoliya/) • [Neelay Velingker](https://nvelingker.github.io/) • [JungHo Jung](https://www.linkedin.com/in/diffunity/) • [Ser-Nam Lim](https://sites.google.com/site/sernam) • [Ziyang Li](https://liby99.github.io/) • [Mayur Naik](https://www.cis.upenn.edu/~mhnaik/)

**University of Pennsylvania** · **University of Central Florida**
</div>

---


# Overview
We introduce **ESCA** (Embodied and Scene-Graph Contextualized Agent), a framework designed to contextualize Multi-modal Large Language Models (MLLMs) through open-domain scene graph generation. ESCA provides structured visual grounding, helping MLLMs make sense of complex and ambiguous sensory environments. At its core is SGClip, a CLIP-based model that captures semantic visual features, including entity classes, physical attributes, actions, interactions, and inter-object relations.

ESCA operates as a multi-stage pipeline, generating task-aware scene graphs that reflect both visual content and human instructions. Through experiments on two challenging embodied environments, we demonstrate that ESCA consistently improves the performance of all evaluated MLLMs, including both open-source and proprietary models.


## 🚀 **Key Features**

- 🛠️ **Structured Scene Understanding:**
  ESCA decomposes visual understanding into four modular stages: concept extraction, object identification, scene graph prediction, and visual summarization.

- 🎯 **SGClip Model:**
  A CLIP-based foundation model for structured scene understanding that supports open-domain concept coverage and probabilistic predictions.

- ⚡ **Transfer Protocol:**
  A general transfer protocol based on customizable prompt templates that enables ESCA to generalize across different downstream tasks.

- 🏹 **ESCA-Video-87K Dataset:**
  A large-scale dataset with 87K video clips, paired with natural language captions, object traces, and spatial-temporal programmatic specifications.

- 🔧 **Neurosymbolic Learning:**
  A model-driven, self-supervised learning pipeline that eliminates the need for manual scene graph annotations.

# 🖥️ Installation
**Note: we need to install three conda environments, one for EB-ALFRED and EB-Habitat, one for EB-Navigation, and one for EB-Manipulation. Please use ssh download instead of HTTP download to avoid error during git lfs pull.**

Download repo
```bash
git clone git@github.com:EmbodiedBench/EmbodiedBench.git
cd EmbodiedBench
```

**You have two options for installation: you can either use
```bash install.sh``` or manually run the provided commands. After completing the installation with `bash install.sh`, you will need to start the headless server and verify that each environment is properly set up.**

1️⃣ Environment for ```Habitat and Alfred```
```bash
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e .
```
2️⃣ Environment for ```EB-Navigation```
```bash
conda env create -f conda_envs/environment_eb-nav.yaml
conda activate embench_nav
pip install -e .
```
3️⃣ Environment for ```EB-Manipulation```
```bash
conda env create -f conda_envs/environment_eb-man.yaml
conda activate embench_man
pip install -e .
```

* Install Coppelia Simulator

CoppeliaSim V4.1.0 required for Ubuntu 20.04; you can find other versions here (https://www.coppeliarobotics.com/previousVersions#)

```bash
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ /PATH/YOU/WANT/TO/PLACE/COPPELIASIM
```

* Add the following to your *~/.bashrc* file:

```bash
export COPPELIASIM_ROOT=/PATH/YOU/WANT/TO/PLACE/COPPELIASIM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

> Remember to source your bashrc (`source ~/.bashrc`) or
zshrc (`source ~/.zshrc`) after this.

* Install the PyRep, EB-Manipulation package and dataset:
```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
pip install -r requirements.txt
pip install -e .
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation
mv EB-Manipulation/data/ ./
rm -rf EB-Manipulation/
cd ../../..
```

> Remember that whenever you re-install the PyRep, simAddOnScript_PyRep.lua will be overwritten. Then, you should copy this again.

* Run the following code to ensure the EB-Manipulation is working correctly (start headless server if you have not):
```bash
conda activate embench_man
export DISPLAY=:1
python -m embodiedbench.envs.eb_manipulation.EBManEnv
```

**Note: EB-Alfred, EB-Habitat and EB-Manipulation require downloading large datasets from Hugging Face or GitHub repositories. Ensure Git LFS is properly initialized by running the following commands:**
```bash
git lfs install
git lfs pull
```

## Start Headless Server
Please run startx.py script before running experiment on headless servers. The server should be started in another tmux window. We use X_DISPLAY id=1 by default.
```bash
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

## EB-Alfred
Download dataset from huggingface.
```bash
# Create checkpoints directory
mkdir -p GroundingDINO/checkpoints

# Download the model checkpoint
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth -P GroundingDINO/checkpoints/
```

3. Set the environment variable for GroundingDINO path:
```bash
export GROUNDING_DINO_PATH=/path/to/your/GroundingDINO
```

### SAM 2.1 Setup
1. Clone the SAM2 repository inside the EmbodiedBench folder:
```bash
git clone https://github.com/facebookresearch/sam2.git
```
2. Follow the instructions in SAM2's README to finish the setup
3. Set the environment variable for the SAM2 repo path
```bash
export SAM_REPO_PATH=/path/to/your/SAM2
```

## EB-Navigation

Run the following code to ensure the EB-Navigation environment is working correctly.
```bash
conda activate embench_nav
python -m embodiedbench.envs.eb_navigation.EBNavEnv
```

## EB-Manipulation
* Install Coppelia Simulator

CoppeliaSim V4.1.0 required for Ubuntu 20.04; you can find other versions here (https://www.coppeliarobotics.com/previousVersions#)

```bash
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ /PATH/YOU/WANT/TO/PLACE/COPPELIASIM
```

* Add the following to your *~/.bashrc* file:

```bash
mv /path/to/downloaded/model_file.model /path/to/EmbodiedBench/models/
```

## Start Headless Server
Please run startx.py script before running experiment on headless servers. The server should be started in another tmux window. We use X_DISPLAY id=1 by default.
```bash
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

## Running Evaluators

EmbodiedBench now uses an organized task-based structure. Evaluators are organized by task (alfred, navigation, habitat, manipulation) with three variants:
- **base**: Baseline VLM implementation
- **gd**: Grounding DINO integration  
- **esca**: Enhanced Scene Context Awareness (LASER + Grounding DINO)

### EB-ALFRED Evaluators

Run the baseline evaluator:
```bash
conda activate embench
python -m embodiedbench.evaluator.alfred.base
```

Run with Grounding DINO:
```bash
conda activate embench
python -m embodiedbench.evaluator.alfred.gd
```

Run with ESCA (recommended):
```bash
conda activate embench
python -m embodiedbench.evaluator.alfred.esca
```

### EB-Navigation Evaluators

Run the baseline evaluator:
```bash
conda activate embench_nav
python -m embodiedbench.evaluator.navigation.base
```

Run with Grounding DINO:
```bash
conda activate embench_nav
python -m embodiedbench.evaluator.navigation.gd
```

Run with ESCA (recommended):
```bash
conda activate embench_nav
python -m embodiedbench.evaluator.navigation.esca
```

### EB-Habitat Evaluator

```bash
conda activate embench
python -m embodiedbench.evaluator.habitat.base
```

### EB-Manipulation Evaluators

Run the baseline evaluator:
```bash
conda activate embench_man
python -m embodiedbench.evaluator.manipulation.base
```

Run with VLA:
```bash
conda activate embench_man
python -m embodiedbench.evaluator.manipulation.vla
```

### Alternative: Using main.py interface

You can also use the unified main.py interface:
```bash
conda activate embench
python -m embodiedbench.main env=eb-alf model_name=Qwen/Qwen2-VL-7B-Instruct model_type=local exp_name='baseline' tp=1
python -m embodiedbench.main env=eb-hab model_name=OpenGVLab/InternVL2_5-8B model_type=local exp_name='baseline' tp=1
```

All evaluators support various configuration options through command-line arguments or config files. Key parameters include:
- `model_name`: The MLLM to use (e.g., 'gpt-4o', 'gemini-2.0-flash')
- `n_shots`: Number of examples for few-shot learning
- `detection_box`: Enable/disable detection box visualization
- `sg_text`: Enable/disable scene graph text output
- `gd_only`: Use only Grounding DINO for object detection without the scene graph generation of ESCA
- `top_k`: Number of top predictions to consider
- `aggr_thres`: Aggregation threshold for predictions

# Citation
```
@inproceedings{huang2025esca,
      title={ESCA: Contextualizing Embodied Agents via Scene-Graph Generation}, 
      author={Jiani Huang, Amish Sethi, Matthew Kuo, Mayank Keoliya, Neelay Velingker, JungHo Jung, Ser-Nam Lim, Ziyang Li, Mayur Naik},
      year={2025},
      booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS)},
}
```
