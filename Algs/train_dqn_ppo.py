import numpy as np
import random
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sdwan_env import SDWANEnv
from callbacks import EpisodeReturnLogger, CongestionLogger, JointActionLogger

SEED = 42
total_timesteps = 100_000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

env = DummyVecEnv([lambda: Monitor(SDWANEnv())])

# DQN Training
logger_dqn = EpisodeReturnLogger()
cong_dqn = CongestionLogger()
jointlog_dqn = JointActionLogger()

model_dqn = DQN(
    'MlpPolicy', env, seed=SEED,
    learning_rate=1e-4,
    buffer_size=500_000,
    batch_size=128,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    train_freq=(4, 'step'),
    target_update_interval=1000,
    verbose=1,
    tensorboard_log="./dqn_centralized_seeded/"
)

model_dqn.learn(total_timesteps, callback=[logger_dqn, cong_dqn, jointlog_dqn])
model_dqn.save('dqn_centralized_seeded')

# PPO Training
logger_ppo = EpisodeReturnLogger()
cong_ppo = CongestionLogger()
jointlog_ppo = JointActionLogger()

model_ppo = PPO(
    'MlpPolicy', env, seed=SEED,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log="./ppo_centralized_seeded/"
)

model_ppo.learn(total_timesteps, callback=[logger_ppo, cong_ppo, jointlog_ppo])
model_ppo.save('ppo_centralized_seeded')

# Print Congestion Summary
for name, logger in [("DQN", cong_dqn), ("PPO", cong_ppo)]:
    print(f"=== {name} Congestion per Episode ===")
    for i, (count, length) in enumerate(zip(logger.congested1, logger.episode_lengths), 1):
        rate = count / length
        print(f"[{name}] Episode {i:3d}: congested_steps = {count:4d}/{length:3d}, rate = {rate:.2%}")
