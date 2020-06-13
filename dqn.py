import argparse

import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import Agent

FLAGS = None

def parse_arguements(parser):

    parser.add_argument(
            "--env_dir", 
            type=str,
            default="Banana_Linux_NoVis/Banana.x86_64",
            help="Directory Path to the Environment Files"
            )
    parser.add_argument(
            "--n_epsisodes", 
            type=int,
            default=2000,
            help="Number of Episodes"
            )

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    
    for i_episode in range(1, n_episodes+1):
        #print('ckpt 2')
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        #print('ckpt 3')
        for t in range(max_t):
            #print('\rt: ' + str(t))
#             if (t % 4) == 0:
#                 action = agent.act(state, eps)
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
#             if (t % 4) == 0:
#                 agent.step(state, action, reward, next_state, done)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    return scores
    
def main():
    
    env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    scores = dqn()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Deep Q-Network for ML Agents Banana Environment')
    parse_arguments(parser)

    FLAGS, unparsed = parser.parse_known_args()

    main()
