import gymnasium as gym
import wordle_env

action_dict = {
  "a":0,
  "b":1,
  "c":2,
  "d":3,
  "e":4,
  "f":5,
  "g":6,
  "h":7,
  "i":8,
  "j":9,
  "k":10,
  "l":11,
  "m":12,
  "n":13,
  "o":14,
  "p":15,
  "q":16,
  "r":17,
  "s":18,
  "t":19,
  "u":20,
  "v":21,
  "w":22,
  "x":23,
  "y":24,
  "z":25
}

if __name__ == "__main__":
  env = gym.make("Wordle-v2", render_mode="ASCII")

  play_game = True
  while play_game:
    observation, info = env.reset()

    while True:
      user_guess = input("Type your guess here: ").lower()
      while len(user_guess) != 5:
        user_guess = input("That was not five letters. Try again: ").lower()
      
      action = []
      for letter in user_guess:
        action.append(action_dict[letter])

      observation, reward, terminated, truncated, info = env.step(action)

      # if truncated:
      #   print(f"The correct word was {info['secret_word']}")

      if terminated or truncated:
        replay = input("Play again? (Y/N): ")
        if replay not in ["Yes", "yes", "Y", "y"]:
          play_game = False
        break