# SDWAN-MARL

A custom multi-agent environment for simulating dynamic overlay selection in SD-WAN and training agents with DQN and PPO algorithms.

## Features
- Multi-overlay request simulation
- Custom reward design
- Stable-Baselines3 integration
- Logging of rewards, congestion, and joint actions

## How to Run

### 1. Install System Dependencies

Make sure the following system packages are installed:

~~~bash
sudo apt-get update && sudo apt-get install -y swig cmake
~~~

---

### 2. Set Up Python Environment

You can use a virtual environment (recommended):

~~~bash
python3 -m venv venv
source venv/bin/activate
~~~

#### 2.1 Install Python dependencies:

~~~bash
pip install -r requirements.txt
~~~

---

### 3. Train the Agent

To start training your agent (DQN or PPO), run:

~~~bash
python train_dqn_ppo.py
~~~


