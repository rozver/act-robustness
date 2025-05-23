# ACT: Action Chunking with Transformers

### *New*: [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.

#### Project Website: https://tonyzhaozh.github.io/aloha/

This repo contains the implementation of ACT, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate ACT in sim or real.
For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the episode after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


### Evaluation

To evaluate a trained policy, use the following commands with the `--eval` flag. This will load the best validation checkpoint and run 50 evaluation episodes.

#### Transfer Cube (Scripted)
```bash
python imitate_episodes.py \
  --eval \
  --policy_class ACT \
  --task_name sim_transfer_cube_scripted \
  --batch_size 16 \
  --seed 1 \
  --num_epochs 500 \
  --lr 1e-4 \
  --ckpt_dir /Users/viviencheng/Desktop/weights/sim_transfer_cube_scripted \
  --kl_weight 1 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200
```

#### Insertion (Scripted)
```bash
python imitate_episodes.py \
  --eval \
  --policy_class ACT \
  --task_name sim_insertion_scripted \
  --batch_size 16 \
  --seed 1 \
  --num_epochs 500 \
  --lr 1e-4 \
  --ckpt_dir /Users/viviencheng/Desktop/weights/sim_insertion_scripted \
  --kl_weight 1 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200
```

#### Transfer Cube (Human)
```bash
python imitate_episodes.py \
  --eval \
  --policy_class ACT \
  --task_name sim_transfer_cube_human \
  --batch_size 16 \
  --seed 1 \
  --num_epochs 500 \
  --lr 1e-4 \
  --ckpt_dir /Users/viviencheng/Desktop/weights/sim_transfer_cube_human \
  --kl_weight 1 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --dim_feedforward 3200
```

Additional evaluation options:
- Add `--temporal_agg` to enable temporal ensembling
- Add `--onscreen_render` to see real-time rendering during evaluation
- Add `--rm_outliers` to remove outlier actions during evaluation
