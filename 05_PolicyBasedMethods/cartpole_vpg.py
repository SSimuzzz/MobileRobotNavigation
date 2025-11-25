import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyEstimator():
    def __init__(self, n_observations, n_actions):
        self.num_observations = n_observations
        self.num_actions = n_actions

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation))

episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


### Main script ###
env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-2

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = PolicyEstimator(n_observations, n_actions)

optimizer = optim.AdamW(policy_net.network.parameters(), lr=LR, amsgrad=True)
action_space = np.arange(env.action_space.n)

steps_done = 0


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 15000
else:
    num_episodes = 15000

total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
batch_counter = 1


for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    rewards, actions, observations = [], [], []
    observation, info = env.reset()
    #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action_probs = policy_net.predict(observation).detach().numpy()
        action = np.random.choice(action_space, p=action_probs)  # randomly select an action weighted by its probability

        # push all episodic data, move to next observation
        observations.append(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)

        done = terminated or truncated

        if done:
            print(i_episode, t)
            # apply discount to rewards
            r = np.full(len(rewards), GAMMA) ** np.arange(len(rewards)) * np.array(rewards)
            r = r[::-1].cumsum()[::-1]
            discounted_rewards = r - r.mean()

            # collect the per-batch rewards, observations, actions
            batch_rewards.extend(discounted_rewards)
            batch_observations.extend(observations)
            batch_actions.extend(actions)
            batch_counter += 1
            total_rewards.append(sum(rewards))

            if batch_counter >= BATCH_SIZE:
                # reset gradient
                optimizer.zero_grad()

                # tensorify things
                batch_rewards = torch.FloatTensor(batch_rewards)
                batch_observationss = torch.FloatTensor(batch_observations)
                batch_actions = torch.LongTensor(batch_actions)

                # calculate loss
                logprob = torch.log(policy_net.predict(batch_observations))
                batch_actions = batch_actions.reshape(len(batch_actions), 1)
                selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                loss = -selected_logprobs.mean()

                # backprop/optimize
                loss.backward()
                optimizer.step()

                # reset the batch
                batch_rewards, batch_observations, batch_actions = [], [], []
                batch_counter = 1

            # get running average of last 100 rewards, print every 100 episodes
            average_reward = np.mean(total_rewards[-100:])
            if i_episode % 100 == 0:
                print(f"average of last 100 rewards as of episode {i_episode}: {average_reward:.2f}")
            #
            # # quit early if average_reward is high enough
            # if early_exit_reward_amount and average_reward > early_exit_reward_amount:
            #     return total_rewards
            #
            # break

            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()