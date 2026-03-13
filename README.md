<div align="center">
<p align="center">
  <h2>Resonate: Reinforcing Text-to-Audio Generation with Online Feedbacks from Large Audio Language Models</h2>
  <!-- <a href=>Paper</a> | <a href="https://meanaudio.github.io/">Webpage</a>  -->

  [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.11661)
  [![Hugging Face Model](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/AndreasXi/Resonate)
  [![Hugging Face Space](https://img.shields.io/badge/Space-HuggingFace-blueviolet?logo=huggingface)](https://huggingface.co/spaces/chenxie95/Resonate)
  [![Webpage](https://img.shields.io/badge/Website-Visit-orange?logo=googlechrome&logoColor=white)](https://resonatedemo.github.io/)

</p>
</div>


## Overview 
Reosnate is a SOTA text-to-audio generator reinforced with online GRPO algorithm. 
It leverages the strong reasoning capabilities of modern Large Audio Language Models as reward models. 
This repo provides a comprehensive pipeline for audio generation, covering Pre-training, SFT, DPO, and GRPO. 

<div align="center">
  <img src="sets/Resonate_model.png" alt="" width="800">
</div>


## Environmental Setup

1. Create a new conda environment:

```bash
conda create -n resonate python=3.11 -y
conda activate resonate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

2. Install with pip:

```bash
git clone https://github.com/xiquan-li/Resonate.git

cd Resonate
pip install -e .
```

<!-- (If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip) -->


## Quick Start

<!-- **1. Download pre-trained models:** -->
To generate audio with our pre-trained model, simply run: 
```bash 
python demo.py --prompt 'your prompt'
```
This will automatically download the pre-trained checkpoints from huggingface, and generate audio according to your prompt. 
By default, this will use [Resonate-GRPO](https://huggingface.co/AndreasXi/Resonate/blob/main/Resonate_GRPO.pth). 
The output audio will be at `Resonate/output/`, and the checkpoints will be at `Resonate/weights/`. 


## Training
Before training, make sure that all files from [here](https://huggingface.co/AndreasXi/Resonate/blob/main/Resonate_GRPO.pth) are placed in `Resonate/weights`. 

### GRPO Training
To launch GRPO training, run:

```bash
bash scripts/train_grpo.sh
```

This script uses the configuration file `Resonate/config/GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic.yaml`. By default, the training employs Qwen2.5-Omni-7B as the reward model and uses `Resonate/data/AudioCaps/GRPO_Meta/train_metadata.jsonl` as the dataset (which corresponds to the AudioCaps training set).
The training outputs will be saved to `exps/TTA/{exp_id}`. 

**Note:** The default configuration requires approximately 90GB of GPU memory. To reduce memory usage, you can decrease the values of `sample.num_audio_per_prompt` and `sample.train_batch_size`.

We also provide another config that leverages the CLAP model for GRPO training, which is at `Resonate/config/GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_clapreward.yaml`.


### DPO Training

To launch DPO training, run:
```bash
bash scripts/train_dpo.sh
```
This script uses the configuration file `Resonate/config/DPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic_offline.yaml`. 


### Pre-training with Flow Matching

Before starting flow-matching training, you need to prepare JSONL metadata file for your dataset, which should have the same format as `Resonate/data/AudioCaps/train_audiocaps_wduration.jsonl`. The metadata file must include the following keys: `audio_id`, `audio_path`, and `caption`. 

After preparing your metadata file, update your data configuration file accordingly (see `Resonate/config/data/audiocaps.yaml` for reference). Also, ensure that the training config `Resonate/config/T2A_pretrain_10s_fixedbsz_fluxaudio_flant5_44kMMVAE.yaml` points to your data config under the `defaults/override data` section. 

Finally, to launch Flow Matching-based pre-training, run:
```bash
bash scripts/train_fm.sh 
```

The output will also be saved to `exps/TTA/{exp_id}`.  


## Inference

Once the training is complete, you can run inference using your trained checkpoint with the following command:
```bash 
bash Resonate/scripts/infer.sh
```
Make sure to modify `ckpt_path` in the script to point to the specific checkpoint you obtained from your training run.

## Evaluation
We provide a simple script to evaluate your model's performance on [TTA-Bench](https://arxiv.org/abs/2509.02398). 
First, install the required packages:
```
pip install msclap audiobox_aesthetics torchcodec
```

Then run:
```
scripts/eval.sh
```
Before running the script, update `ckpt_path` to point to your trained model checkpoint, and set `output_path` to the directory where you want to save the results.

The script will automatically perform batch inference on the accuracy subset of TTA-Bench and compute AudioBox-AES, CLAPScore, and AQAScore (based on Qwen3-Omni-Instruct).

## Citation
TODO

## Acknowledgement

This codebase is built upon [MeanAudio](https://github.com/xiquan-li/MeanAudio) and [FlowGRPO](https://github.com/yifan123/flow_grpo). We sincerely thank the authors and contributors of these projects for open-sourcing their excellent work.