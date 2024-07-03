'''
 @Author: wandouyu 
 @Date: 2024-07-03 
'''

import os
import numpy as np
import torch as th
from stable_baselines3 import PPO,SAC,DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback
import matplotlib.pyplot as plt

from env_D import air_env                        #一维离散空间版本
# from env_continuous import air_env                 #连续版本环境
# from env_MD import air_env                       #多维离散空间版本


###################################测试函数#############################################
def evaluate_model(model, env, n_eval_episodes=10):
    success_count = 0
    episode_rewards = []

    for episode in range(n_eval_episodes):
        episode_reward = 0
        obs,_ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _,info = env.step(action)
            episode_reward += reward


        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
#########################################################################################
#########################################回调函数#########################################
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = "tmp/best_model"
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(f"{self.save_path}/best_model_DQN")
                  self.model.save_replay_buffer(f"{self.save_path}/best_model_DQN_replay_buffer")

        return True

#####################################################################################################
#日志文件
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

#训练环境
train_env = make_vec_env(lambda: air_env(), n_envs=16,monitor_dir=log_dir)
# train_env = VecNormalize(train_env)
#回调
callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)
# callback = CheckpointCallback(save_freq=1000, 
#                               save_path=log_dir,
#                               name_prefix="SAC_model",
#                               save_replay_buffer=True,
#                               save_vecnormalize=True,
#                               verbose=1)
##################################################连续动作空间模型###########################################################
#用于env_continuous

# model = SAC("MlpPolicy",
#             train_env,
#             tau=0.05,
#             gamma=0.96,
#             learning_starts=10000,
#             verbose=0,
#             device="cuda")

##########################################################################################################
##################################################离散动作空间###########################################################
#PPO算法，可用于env_MD和env_D

# policy_kwargs = dict(net_arch=dict(pi=[64, 256, 128], vf=[64, 256, 64]))
# model = PPO(policy = "MlpPolicy",
#             env = train_env,
#             learning_rate = 0.003,
#             n_steps = 2048,
#             batch_size = 256,
#             n_epochs = 10,
#             gamma = 0.96,
#             gae_lambda = 0.95,
#             clip_range=  0.2,
#             clip_range_vf = None,
#             normalize_advantage = True,
#             ent_coef = 0,
#             vf_coef = 0.5,
#             max_grad_norm = 0.5,
#             use_sde = False,
#             sde_sample_freq = -1,
#             target_kl = None,
#             policy_kwargs = policy_kwargs,
#             verbose = 0,
#             seed = None,
#             device = "cuda",
#             _init_setup_model = True)

#DQN算法，可用于env_D

policy_kwargs = dict(net_arch=[64,256,256,256])
model = DQN(policy = "MlpPolicy",
            env = train_env,
            learning_starts=500000,
            buffer_size=2000000,
            device = "cuda",
            batch_size=256,
            tau=0.6,
            gamma=0.96,
            train_freq=1,
            verbose=0,
            policy_kwargs=policy_kwargs)
            
######################################################################################################
# model = PPO.load("tmp/best_model/best_model_PPO",env=train_env)
# model = DQN.load("tmp/best_model/best_model_DQN",env=train_env)
# model.load_replay_buffer("tmp/best_model/best_model_DQN_replay_buffer")
# model = SAC.load("tmp/best_model/best_model_SAC",env=train_env)
# model.load_replay_buffer("tmp/best_model/best_model_SAC_replay_buffer")
############################################模型训练###################################################

timesteps = 1e7
model.learn(total_timesteps=int(timesteps), callback=callback,reset_num_timesteps=False)

######################################################################################################

#画图
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "DQN air")
plt.show()