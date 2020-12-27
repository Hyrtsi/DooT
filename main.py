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
import utils
import matplotlib.pyplot as plt


def main():
    game = init_game()
    episodes = 1000

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    game.new_episode()

    player_start_pos = utils.get_player_pos(game)
    print("Player start pos:", player_start_pos)

    reward_controller = Reward(player_start_pos)
    model = Model(reward_controller,n_channels=game.get_screen_channels())
    episode_mean_rewards = []

    print()
    print("Model setup complete. Starting training episodes")

    for i in range(episodes):
        game.new_episode()
        rewards_current_episode = []
        while not game.is_episode_finished():
            state = game.get_state()
            state_number = state.number

            reward = model.step(game)
            rewards_current_episode.append(reward)           
            
            if state_number % 50 == 0:
                print("State #" + str(state_number))
                print("Reward:", reward)
                print("=====================")

            # disable sleep unless a human wants to watch the game
            #if sleep_time > 0:
            #    sleep(sleep_time)

        
        print("Episode", i, "finished in", )
        print("Total rewards:", sum(rewards_current_episode))
        print("************************")

        # compress all the rewards of an episode into a single number
        episode_mean_rewards.append(np.mean(rewards_current_episode))

        """
        if i % 5 == 0:
            plt.plot(episode_mean_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.show()
        """

        print("i:", i, "max mean reward", max(episode_mean_rewards), "last mean reward", episode_mean_rewards[i])
        model.save_model("my_model.h5")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()

print()
print("-------- starting ------------")
main()