import gymnasium as gym
import wordle_env

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
  "policy_type":"MultiInputPolicy",
  "total_timesteps":1000000,
  "env_id":"Wordle-v3"
}

run = wandb.init(
  project="Wordle_PPO",
  config=config,
  sync_tensorboard=True,
)

def make_env():
  env = gym.make(config["env_id"])
  env = Monitor(env)
  return env

model = PPO(
  config["policy_type"],
  make_env(),
  verbose=1,
  tensorboard_log=f"runs/{run.id}",
  learning_rate=0.001,
  device="auto"
)

model.learn(
  total_timesteps=config["total_timesteps"],
  callback=WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2
  ),
)

env = gym.make(config["env_id"], render_mode="ASCII")
observation, info = env.reset()

for _ in range(60):
  action, _ = model.predict(observation)
  # print(action)
  observation, reward, terminated, truncated, info = env.step(action)
  if terminated or truncated:
    observation, info = env.reset()

run.finish()
