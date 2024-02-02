import numpy as np
import gymnasium as gym
from gymnasium import spaces
from time import sleep

class WordleEnv(gym.Env):
  metadata = {"render_modes":["ASCII"], "render_fps":2}

  def __init__(self, render_mode=None):
    self.observation_space = spaces.Dict(
      {
        "guess":spaces.Box(0, 26, shape=(30,), dtype=int),
        "feedback":spaces.Box(-1, 5, shape=(30,), dtype=int)
      }
    )

    with open("wordle_env/envs/5-letter-words.txt", "r") as file:
      self._word_list = file.readlines()

    for i in range(len(self._word_list)):
      self._word_list[i] = self._word_list[i].strip().lower()

    self.action_space = spaces.Discrete(len(self._word_list))

    # self._action_to_letter = {
    #   0:"a",
    #   1:"b",
    #   2:"c",
    #   3:"d",
    #   4:"e",
    #   5:"f",
    #   6:"g",
    #   7:"h",
    #   8:"i",
    #   9:"j",
    #   10:"k",
    #   11:"l",
    #   12:"m",
    #   13:"n",
    #   14:"o",
    #   15:"p",
    #   16:"q",
    #   17:"r",
    #   18:"s",
    #   19:"t",
    #   20:"u",
    #   21:"v",
    #   22:"w",
    #   23:"x",
    #   24:"y",
    #   25:"z"
    # }

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def _get_obs(self, guess):
    if self._guesses == 0:
      return {"guess":self._obs["guess"].flatten(), "feedback":self._obs["feedback"].flatten()}

    secret_dict = {}
    for letter in self._word:
      if letter in secret_dict.keys():
        secret_dict[letter] += 1
      else:
        secret_dict.update({letter:1})

    index = self._guesses-1
    self._obs["guess"][index] = np.array([ord(guess[0])-ord('a'),ord(guess[1])-ord('a'),ord(guess[2])-ord('a'),ord(guess[3])-ord('a'),ord(guess[4])-ord('a')])

    # observation = {
    #   "guess":np.array([ord(guess[0])-ord('a'),ord(guess[1])-ord('a'),ord(guess[2])-ord('a'),ord(guess[3])-ord('a'),ord(guess[4])-ord('a')]),
    #   "feedback":np.array([-1, -1, -1, -1, -1])
    # }

    for i in range(len(guess)):
      if guess[i] == self._word[i]:
        self._obs["feedback"][index][i] = 5
        secret_dict[guess[i]] -= 1
      elif guess[i] not in secret_dict.keys():
        self._obs["feedback"][index][i] = 0

    for i in range(len(guess)):
      if self._obs["feedback"][index][i] == -1 and secret_dict[guess[i]] > 0:
        self._obs["feedback"][index][i] = 1
        secret_dict[guess[i]] -= 1
      elif self._obs["feedback"][index][i] == -1:
        self._obs["feedback"][index][i] = 0

    return {"guess":self._obs["guess"].flatten(), "feedback":self._obs["feedback"].flatten()}
    
  def _get_info(self):
    return {"secret_word":self._word}

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # with open("wordle_env/envs/5-letter-words.txt", "r") as file:
    #   words = file.readlines()

    self._word = self.np_random.choice(self._word_list).strip().lower()

    self._guesses = 0

    self._obs = {
      "guess":np.array([[26, 26, 26, 26, 26],
                        [26, 26, 26, 26, 26],
                        [26, 26, 26, 26, 26],
                        [26, 26, 26, 26, 26],
                        [26, 26, 26, 26, 26],
                        [26, 26, 26, 26, 26]]),
      "feedback":np.array([[-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1]])
    }

    observation = self._get_obs("")
    info = self._get_info()

    return observation, info


  def step(self, action):
    self._guesses += 1
    
    guess = self._word_list[action]

    # guess = ""
    # for letter in action:
    #   guess += self._action_to_letter[letter]

    observation = self._get_obs(guess)
    reward = sum(self._obs["feedback"][self._guesses-1])
    terminated = guess == self._word
    truncated = self._guesses >= 6 and not terminated
    info = self._get_info()

    if self.render_mode == "ASCII":
      self.render(guess)
      # print(f"Observation: {observation}")
      # print(f"Reward: {reward}")

    return observation, reward, terminated, truncated, info
  
  def render(self, guess):
    if self.render_mode == "ASCII":
      index = self._guesses-1
      print(f"Attempt #{self._guesses}")
      word = ""
      for i in range(len(guess)):
        if self._obs["feedback"][index][i] == 5:
          word += f"\033[92m{guess[i]}\033[0m"
        elif self._obs["feedback"][index][i] == 1:
          word += f"\033[93m{guess[i]}\033[0m"
        else:
          word += f"\033[91m{guess[i]}\033[0m"
      print(word)
      if self._guesses == 6:
        print(f"The secret word was {self._word}")
      sleep(1/self.metadata["render_fps"])