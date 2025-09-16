# Neural Robot Dynamics (NeRD)

This repository contains the implementation for the paper [Neural Robot Dynamics](https://neural-robot-dynamics.github.io/) (CoRL 2025).

[***Neural Robot Dynamics***](https://neural-robot-dynamics.github.io/) <br/>
[Jie Xu](https://people.csail.mit.edu/jiex), [Eric Heiden](https://eric-heiden.com/), [Iretiayo Akinola](https://research.nvidia.com/person/iretiayo-akinola), [Dieter Fox](https://homes.cs.washington.edu/~fox/), [Miles Macklin](https://blog.mmacklin.com/about/), [Yashraj Narang](https://research.nvidia.com/person/yashraj-narang) <br/>
***CoRL 2025***

In this paper, we propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. In this repository, we demonstrate how to integrate the learned NeRD models as an interchangeable backend solver within [NVIDIA Warp](https://developer.nvidia.com/warp-python). 

<p align="center">
    <img src="figures/overview.png" alt="overview" width="800" />
</p>


## Installation
- The code has been tested on Ubuntu 20.04 with Python 3.8 and CUDA 12.9.
- `git clone xxx`
- Create an Anaconda virtual environment (recommended)
  ```
  conda env create -n nerd python=3.8
  conda activate nerd
  ```
- Install [PyTorch 2.2.2](https://pytorch.org/get-started/previous-versions/#linux-and-windows-13) (tested)
- Install other depandent packages:
  ```
  pip install -r requirements.txt
  ```
- Testing the installation
  ```
  cd utils
  python visualize_env.py --env-name Cartpole --num-envs 1
  ```
  You are expected to see a passive Cartpole motion in a visualization UI.

## Testing NeRD Models Without Training

We released pretrained NeRD models for *Cartple*, *Pendulum*, *Franka*, *Ant*, and *ANYmal* for you to quickly test out neural dynamics without training from scratch. We also released the RL policies that are trained in the NeRD-integrated simulator and are used in the Experiment section in the paper. Both pretrained NeRD models and RL policies are in the `pretrained_models` folder.

We also provide evaluation scripts of the metrics used in the Experiment section of the paper.

### Passive Motion Evaluation
The script `eval/eval_passive/eval_passive_motion.py` conduct the long-horizon passive motion evaluation in Section 5.1. The script below evaluate and render the 100-step passive motion for Cartpole.
```
cd eval/eval_passive
python eval_passive_motion.py --env-name Cartpole --model-path ../../pretrained_models/NeRD_models/Cartpole/model/nn/model.pt --env-mode neural --num-envs 1 --num-rollouts 5 --rollout-horizon 100 --render
```
To get the metrics averaged from 2048 trajectories, you can run the following script:
```
python eval_passive_motion.py --env-name Cartpole --model-path ../../pretrained_models/NeRD_models/Cartpole/model/nn/model.pt --env-mode neural --num-envs 2048 --num-rollouts 2048 --rollout-horizon 100 --seed 500
```

To change the ground configuration for the double pendulum as done in Table 3 in Section C.2, you can manually specify the contact configuration ID in [`envs/warp_sim_envs/env_pendulum_with_contact.py`](envs/warp_sim_envs/env_pendulum_with_contact.py?ref_type=heads#L86), then run the `eval_passive_motion.py` script:
```
python eval_passive_motion.py --env-name PendulumWithContact --model-path ../../pretrained_models/NeRD_models/Pendulum/model/nn/model.pt --env-mode neural --num-envs 2048 --num-rollouts 2048 --rollout-horizon 100 --seed 500
```


### RL Policy Evaluation
You can test an individual RL policy using the `run_rl.py` script:
```
cd eval/eval_rl
python run_rl.py --rl-cfg ./cfg/Anymal/anymal_forward.yaml --playback ../../pretrained_models/RL_policies/Anymal/forward_walk/0/nn/AnymalPPO.pth --num-envs 1 --num-games 2 --env-mode [neural|ground-truth] [--render]
```
where `--env-mode` specifies to use NeRD dynamics or ground-truth analytical dynamcis.

To evaluate a batch of policies with different seeds in both ground-truth dynamics and NeRD dynamics (as done in Table 1 in the paper), you can run our batch evaluation script with the batch evaluation config file:
```
python batch_eval_policy.py --num-envs 2048 --num-games 2048 --eval-cfg ./eval_cfg/Anymal/eval_cfg_side.yaml
```

> [!Note]
> Some released NeRD models were trained using the data generated from an older version of Warp (Warp 1.5.1), thus the motion produced by NeRD models might not match the ground-truth dynamics generated with the latest version of Warp. This could result in different reported metrics from the numbers in the paper. To fully reproduce the numbers reported in the paper, please use Warp 1.5.1 for *Ant* and *Pendulum* and use Warp 1.8.0 for *ANYmal*, otherwise you can also train your own NeRD models by following the instructions shown in the next section.


## Train NeRD Models

Training a NeRD model involves two steps: (1) generate dataset and (2) train a NeRD model using the generated dataset. We provided data generation and training scripts, as well as training configuration files for the examples in the paper. Below shows example scripts for *Cartpole* and *Ant*, and you can try other examples using our provided scripts.

### Generate Dataset

We need to generate the training and validation datasets for NeRD training. Each dataset consists of a set of random motion trajectories. We generate datasets with 1M transitions in the examples here.

#### Cartpole
```
cd generate
python generate_dataset_contact_free.py --env-name Cartpole --num-transitions 1000000 --dataset-name trajectory_len-100_1M_train.hdf5 --trajectory-length 100 --num-envs 2048 --seed 0
python generate_dataset_contact_free.py --env-name Cartpole --num-transitions 1000000 --dataset-name trajectory_len-100_valid.hdf5 --trajectory-length 100 --num-envs 2048 --seed 10
```

#### Ant
```
cd generate
python generate_dataset_ant.py --env-name Cartpole --num-transitions 1000000 --dataset-name trajectory_len-100_1M_train.hdf5 --trajectory-length 100 --num-envs 2048 --seed 0
python generate_dataset_ant.py --env-name Cartpole --num-transitions 1000000 --dataset-name trajectory_len-100_valid.hdf5 --trajectory-length 100 --num-envs 2048 --seed 10
```

### Training

#### Cartpole
```
cd train
python train.py --cfg ./cfg/Cartpole/transformer.yaml --logdir ../../data/trained_models/Cartpole/
```

#### Ant
```
cd train
python train.py --cfg ./cfg/Ant/transformer.yaml --logdir ../../data/trained_models/Ant/
```

## Train RL Policies in a NeRD-Integrated Simulator
We use [rl-games](https://github.com/Denys88/rl_games) to train RL policies within a NeRD-Integrated Simulator. Thanks for the seamless design of the NeRD integrator within Warp simulator, we can turn on the using of NeRD dynamics models by simply call the `set_env_mode` function in `NeuralEnvironment` class to switch between using the NeRD dynamics and using the analytical dynamics. Below shows an example to train the RL policy for *ANYmal* forward walking task.
```
cd eval/eval_rl
python run_rl.py --rl-cfg ./cfg/Anymal/anymal_forward.yaml --env-mode neural --nerd-model-path ../../pretrained_models/NeRD_models/Anymal/model/nn/model.pt 
```

## Implementation Details
We implemented NeRD as a class of [Integrator](integrator/integrator_neural.py?ref_type=heads#L127) in Warp Sim, which shares the same [`simulate`](integrators/integrator_neural.py?ref_type=heads#L329) function interfaces as other default integrators such as Featherstone and XPBD integrators so that the specific integrator type is transparent to the environments implemented on top of it. The `AbstractContactEnvironment` is a wrapper around any existing robotic environment that pre-extract the contact pair lists for NeRD. `NeuralEnvironment` is a wrapper around `AbstractContactEnvironment` to construct NeRD integrator and provide a unified view of different environments to top-level applications such as RL training and NeRD evaluators.

## Citation

If you find our paper or code useful, please consider citing:
```
@inproceedings{
  xu2025neural,
  title={Neural Robot Dynamics},
  author={Jie Xu and Eric Heiden and Iretiayo Akinola and Dieter Fox and Miles Macklin and Yashraj Narang},
  booktitle={9th Annual Conference on Robot Learning},
  year={2025}
}
```