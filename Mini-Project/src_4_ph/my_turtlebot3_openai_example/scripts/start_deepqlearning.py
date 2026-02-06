#!/usr/bin/env python3
import gym
import numpy
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from functools import reduce
import os
import csv

import time
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(256, 128)
        self.head = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return self.head(x)


def select_action(state, eps_start, eps_end, eps_decay):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(0)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold


def optimize_model(batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, torch.squeeze(action_batch, 2))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)

    # Use Double DQN instead of DQN:
    #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states)
        .gather(1, next_actions)
        .squeeze()
        .detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()


def save_training_plot(outdir, episode_durations, batch_size, gamma, target_update,
                       epsilon_start, epsilon_end, epsilon_decay,
                       lr, opt,
                       training_time_s, max_peak, max_peak_episode, highest_reward):
    """
    Crea un'immagine con:
      - grafico della durata per episodio + media mobile (100 episodi)
      - tabella con parametri e statistiche
    e la salva in outdir/training_result.png
    """

    fig = plt.figure(figsize=(8, 6))

    # Griglia: grafico sopra, tabella sotto
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_plot = fig.add_subplot(gs[0])
    ax_tab = fig.add_subplot(gs[1])

    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    ax_plot.set_title("Result")
    ax_plot.set_xlabel("Episode")
    ax_plot.set_ylabel("Duration")

    # curva blu: durate per episodio
    ax_plot.plot(durations_t.numpy())

    # curva arancione: media mobile su 100 episodi
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1)
        means = torch.cat((torch.zeros(99), means))
        ax_plot.plot(means.numpy())

    # Tabella parametri
    ax_tab.axis("off")

    table_data = [
        ["BATCH SIZE", batch_size],
        ["GAMMA", gamma],
        ["TARGET UPDATE", target_update],
        ["EPSILON START", epsilon_start],
        ["EPSILON END", epsilon_end],
        ["EPSILON DECAY", epsilon_decay],
        ["LR", lr],
        ["Optimizer", opt],
        ["Training time (s)", training_time_s],
        ["Max peak", max_peak],
        ["Maximal Peak Episode", max_peak_episode],
        ["Highest Reward", highest_reward],
    ]

    ax_tab.table(cellText=table_data, loc="center", cellLoc="center")

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, "training_result.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    rospy.loginfo(f"Training plot salvato in: {save_path}")


# import our training environment
if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    metrics_path = os.path.join(outdir, "metrics.csv")
    metrics_f = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow([
        "episode","reward","success","collision","steps",
        "goal_x","goal_y","dist","min_scan",
        "r_progress","r_time","r_collision_avoid","r_yaw","r_terminal",
        "epsilon"
    ])


    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_start = rospy.get_param("/turtlebot3/epsilon_start")
    epsilon_end = rospy.get_param("/turtlebot3/epsilon_end")
    epsilon_decay = rospy.get_param("/turtlebot3/epsilon_decay")
    n_episodes = rospy.get_param("/turtlebot3/n_episodes")
    batch_size = rospy.get_param("/turtlebot3/batch_size")
    target_update = rospy.get_param("/turtlebot3/target_update")
    lr = float(rospy.get_param("/turtlebot3/lr"))
    optimizer_name = rospy.get_param("/turtlebot3/optimizer")

    max_step = rospy.get_param("/turtlebot3/max_step")
    running_step = rospy.get_param("/turtlebot3/running_step")

    learning_starts = rospy.get_param("/turtlebot3/learning_starts", 5000)

    # Initialises the algorithm that we are going to use for learning
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    n_observations = rospy.get_param("/turtlebot3/new_ranges") + 2

    # initialize networks with input and output sizes
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)

    target_net_path = os.path.join(outdir, "target_net.pth")
    policy_net_path = os.path.join(outdir, "policy_net.pth")

    if os.path.exists(target_net_path):
        target_net.load_state_dict(
            torch.load(target_net_path, map_location=device, weights_only=True)
            )
        print("Target Net loaded")

    if os.path.exists(policy_net_path):
        policy_net.load_state_dict(
            torch.load(policy_net_path, map_location=device, weights_only=True)
            )
        print("Policy Net Loaded")
    
    target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)

    memory = ReplayMemory(100000)
    episode_durations = []
    steps_done = 0

    start_time = time.time()
    highest_reward = 0

    max_peak = 0
    max_peak_episode = 0

    # Starts the main training loop: the one about the episodes to do
    for i_episode in range(n_episodes):
        rospy.logdebug("############### START EPISODE=>" + str(i_episode))

        cumulated_reward = 0
        done = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = torch.tensor(observation, device=device, dtype=torch.float)
        #state = ''.join(map(str, observation))

        for t in range(max_step):
            rospy.logwarn("############### Start Step=>" + str(t))
            # Select and perform an action
            action, epsilon = select_action(state, epsilon_start, epsilon_end, epsilon_decay)
            rospy.logdebug("Next action is:%d", action)

            observation, reward, done, info = env.step(action.item())

            last_info = info


            cumulated_reward += float(reward)

            if done:
                print("\n=== EPISODE DONE ===")
                print(f"success={info.get('success')} collision={info.get('collision')} steps={info.get('steps')}")
                print(f"goal=({info.get('goal_x')}, {info.get('goal_y')}) dist={info.get('dist')} min_scan={info.get('min_scan')}")
                print("cumulated_reward_terms:",
                      {k: info.get(k) for k in ["cum_r_progress","cum_r_time","cum_r_smooth","cum_r_collision_avoid","cum_r_terminal"]})
                print("Last step reward:", reward)
                print("Cumulated reward:", cumulated_reward)
                print("====================\n")

                metrics_writer.writerow([
                    i_episode+1,
                    float(cumulated_reward),
                    int(last_info.get("success", 0)),
                    int(last_info.get("collision", 0)),
                    int(last_info.get("steps", t+1)),
                    last_info.get("goal_x",""),
                    last_info.get("goal_y",""),
                    last_info.get("dist",""),
                    last_info.get("min_scan",""),
                    last_info.get("cum_r_progress",""),
                    last_info.get("cum_r_time",""),
                    last_info.get("cum_r_collision_avoid",""),
                    last_info.get("yaw_reward",""),
                    last_info.get("cum_r_terminal",""),
                    float(epsilon),
                ])
                metrics_f.flush()



            rospy.logdebug(str(observation) + " " + str(reward))
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            reward = torch.tensor([reward], device=device)

            #next_state = ''.join(map(str, observation))
            next_state = torch.tensor(observation, device=device, dtype=torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(next_state))

            if steps_done >= learning_starts:
                optimize_model(batch_size, gamma)

            if done:
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])

                episode_durations.append(t + 1)
                if t > max_peak:
                    max_peak = t+1
                    max_peak_episode = i_episode
                break
            else:
                rospy.logdebug("NOT DONE")
                state = next_state

            rospy.logwarn("############### END Step=>" + str(t))
            # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(i_episode + 1) + " - gamma: " + str(
            round(gamma, 2)) + " - epsilon: " + str(round(epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    metrics_f.close()

    torch.save(target_net.state_dict(), target_net_path)
    torch.save(policy_net.state_dict(), policy_net_path)

    rospy.loginfo(("\n|" + str(n_episodes) + "|" + str(gamma) + "|" + str(epsilon_start) + "*" +
                   str(epsilon_decay) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # ---- Calcolo statistiche per il grafico / tabella ----
    training_time_s = int(time.time() - start_time)

    # ---- Salvataggio dell'immagine ----
    save_training_plot(
        outdir=outdir,
        episode_durations=episode_durations,
        batch_size=batch_size,
        gamma=gamma,
        target_update=target_update,
        epsilon_start = epsilon_start,
        epsilon_end = epsilon_end, 
        epsilon_decay = epsilon_decay,
        lr=lr, 
        opt=optimizer_name,
        training_time_s=training_time_s,
        max_peak=max_peak,
        max_peak_episode=max_peak_episode,
        highest_reward=highest_reward
    )

    env.close()
