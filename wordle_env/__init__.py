from gymnasium.envs.registration import register

register(
  "Wordle-v3",
  entry_point="wordle_env.envs:WordleEnv",
)
