# Lab 1 - Reinforcement Learning with a Quadruped

## Overview

In this lab you will be asked complete and tune a code base for training a four legged robot in the Mujoco Simulator (the simulation environment is based on JAX, a python library by Google that is designed for numerical computing and machine learning tasks).
This includes implementing a component of the PPO algorithm, tuning the cost function to instruct the robot to walk over an elevated step, and reason about possible extensions and steps towards real-world deployment.

The lab will have three different levels, 'E', 'C' and 'A'. The level naming corresponds to the grade that is obtained if the respective level is completed. Levels E and C are code completion exercises, while level A is a small research task.
The levels are incremental, so e.g. reaching grade C is only possible if levels E and C are successfully completed, similarly level E and C and A are required for reaching an A grade.

## Lab Structure

- **Level E** (`lab1_E.ipynb`): Implement PPO loss function, and show that the robot can follow simple velocity commands
- **Level C** (`lab1_C.ipynb`): Tune rewards on more challenging environment, and show the robot succesfully climbing obstacles. 
- **Level A**: Discussion deliverable - reason about extensions and real-world deployment (see below)

## Getting Started
We will provide instructions to use our GPU cluster in the lab.
### Clone the repository

First, clone the lab repository:

```git clone https://github.com/finnBsch/eai2025_lab1_rl.git```

Then navigate to the cloned directory:

```cd eai2025_lab1_rl```

### Create a Python Virtual Environment

Creating a Python venv (or conda/mamba env) is good practice to avoid installation/dependency conflicts when working on different projects. Given that labs 1, 2 and 3 will have different dependencies we strongly recommend you doing this.

To create a venv, in a terminal (in the directory where this lab is) run:

```python3 -m venv <your_venv_name>```

For example:

```python3 -m venv venv_lab1```

Then run:

```source <path_to_your_venv>/venv_lab1/bin/activate```

This will source your venv, as a consequence, there will be an isolated python interpreter (a sourced venv is indicated by the venv name appearing in brackets at the beginning of the terminal lines).
Packages you install while the venv is sourced will be specific to your venv and not interfer with system-wide installs you might have.

In vscode/jupyter notebook, make sure that you select the interpreter of your venv before executing cells.
This way installations performed will be specific for the venv only and the notebook will find the correct dependencies installed in your venv.

A few ways to do this:
- VSCode: ctrl+p, then enter '>Python Select Interpreter' and chose your venv's interpreter (should be something like <path_to_your_venv/venv_lab1/bin/python>)
- VSCode: sometimes on the top right of the window there is a select Kernel button where you can do the same thing
- VSCode: sometimes when executing cells you will be prompted to select an interpreter, then you can select the one of your venv
- If you run into issues, please ask in the lab.

(Note: when executing the jupyter cells you will likely be prompted to install the package 'ipykernel'.
Click install, as this is required to interface Jupyter and python.)

### Install general dependencies

With your venv sourced please run the following commands in the terminal:
```
python3 -m pip install matplotlib opencv-python mediapy jax[cuda12]
```

### Install mujoco related dependencies

Furthermore install these dependencies (note: the order matters here, so please stick to it, if something seems to go wrong, just restart from the first one and run them all again):
```
python3 -m pip install git+https://github.com/finnBsch/mujoco_playground.git@lab1_rl
python3 -m pip uninstall mujoco-mjx
python3 -m pip install git+https://github.com/finnBsch/mujoco.git@lab#subdirectory=mjx
```

## Level A - Detailed Instructions
For Level A, you will need to write a **1-2 page discussion** about the policy you trained in Level C and potential extensions to improve it. Use the papers linked below as references and inspiration for your discussion.

### Required Discussion Points:
Your report must cover the following topics:

1. **Perception/Reward Function Analysis**
   - Examine the reward function setup to understand how the robot perceives its environment
   - Explain what information the policy uses for decision-making (hint: look at observation space and reward structure)
   - Discuss how this perception influences obstacle climbing behavior, and what the robot must do to climb obstacles.

2. **System Limitations and Improvements**
   - Identify at least 2 specific problems with the current setup regarding:
     - Safety considerations for real-world deployment
     - Robot efficiency (in terms of power usage)
     - Applicability of the approach to more complex environments
   - Discuss what is needed to overcome those specific problems, for instance you could suggest additional sensors and explain how they would be integrated

3. **Real-World Deployment Considerations**
   - Discuss the sim-to-real transfer challenges
   - Address robustness, safety, and reliability requirements
   - Reference insights from the provided papers to support your arguments


## Related Resources

- PPO paper (OpenAI): https://arxiv.org/pdf/1707.06347
- Perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2201.08117
- Non-perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2010.11251
