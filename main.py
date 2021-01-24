#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function
import vizdoom as vzd
import numpy as np

from random import choice
from time import sleep

from init_game import init_game
from reward import Reward
from model import Model
from trainer_simple import TrainerSimple
import utils
import matplotlib.pyplot as plt


def main():
    episodes = 16384
    episode_length = 1024
    game = init_game(episode_length)

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    #sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    game.new_episode()

    player_start_pos = utils.get_player_pos(game)
    print("Player start pos:", player_start_pos)

    reward_controller = Reward(player_start_pos)
    model = Model(episode_length)
    model.load_model("model")
    trainer = TrainerSimple(model, reward_controller, episode_length)

    print("Model setup complete. Starting training episodes")

    for i in range(0, episodes):
        #game.set_doom_map(choice(["map01", "map02", "map03", "map04", "map05"]))
        game.set_doom_map(choice(["map01"]))

        game.new_episode()
        reward_controller.player_start_pos = utils.get_player_pos(game)

        frame_id = 0
        while not game.is_episode_finished():
            trainer.step(game, i, frame_id)
            frame_id += 1

        if ((i+1) % 4) == 0:
            model.save_model("model")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()

print()
print("-------- starting ------------")
main()