"""
Gym contains an environemnt for playing text-based games.
Each action is a command to be executed in the game, from a preset list

Observations are text sequences, so need to be tokenized and embedded
Env is also a POMDP - since the inventory is not shown, neither is previous state.
So deal with that by stacking states
TextWorld also gives intermediate rewards based on steps to solve the game

"""

from textworld import gym
from textworld.gym import register_game

env_id = register_game("games/t1.ulx")

env = gym.make(env_id)

print(env)