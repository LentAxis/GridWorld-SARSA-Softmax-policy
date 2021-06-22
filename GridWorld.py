
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib.patches as patches
import random


class Environment:

    def __init__(self, size=(5, 4), start=(0, 0), goal=(4, 3), step_penalty=-0.1, goal_reward=10):

        # Set board dimensions
        self.size = size

        # Set state to start
        self.state = start

        # Set start- and endpoint
        self.start = start
        self.goal = goal

        # Set reward parameters
        self.R = np.ones([size[0], size[1]]) * step_penalty
        self.R[goal[0]][goal[1]] = goal_reward

        # Set termination conditions
        self.final_mask = np.zeros((size[0], size[1]), dtype=bool)
        self.final_mask[self.R != step_penalty] = True

        # Set action parameters (up, down, left, right)
        self.action_coordinates = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def add_obstacle(self, loc, penalty):

        # We will set the reward on the location "loc" in our reward matrix to the value "penalty"
        self.R[loc[0]][loc[1]] = penalty

        # Finally we will say, that this state has an termination condition
        self.final_mask[loc[0], loc[1]] = True

    def step(self, action):

        # Move state in direction of the action and save the new state as "observation"
        state_x = self.state[0] + self.action_coordinates[action][0]
        state_y = self.state[1] + self.action_coordinates[action][1]
        observation = (state_x, state_y)

        # We can now get the reward for the next state
        reward = self.R[observation]

        # If the step results in a termination condition, we will return true with "done"
        if self.final_mask[observation[0]][observation[1]]:
            done = True
        else:
            done = False

        # Now we can update the current state for the environment
        self.state = observation

        # If you want to see some information while running
        info = "State:", self.state, ", Reward:", reward

        return observation, reward, done, info

    def reset(self):

        # Reset state to start
        self.state = self.start

    def get_valid_actions(self, state):

        valid_actions = []

        # Just for better visualization
        x = state[0]
        y = state[1]

        # We will create an array with all possible actions from the state.
        # If the state is at a border, the action, which will lead our player outside
        # the environment, will not be added.
        if y > 0:
            valid_actions.append(0)
        if y < (self.size[1] - 1):
            valid_actions.append(1)

        if x > 0:
            valid_actions.append(2)
        if x < (self.size[0] - 1):
            valid_actions.append(3)

        return valid_actions


class Agent:

    def __init__(self, environment, alpha=0.1, gamma=0.9, printinfo=False):

        # Set agent parameters
        self.env = environment
        self.printinfo = printinfo
        self.alpha = alpha
        self.gamma = gamma
        self.size = environment.size
        self.action_coordinates = environment.action_coordinates
        self.tau_array = None

        # Q_Values for every state and action
        self.Q_values = np.zeros([environment.size[0], environment.size[1], 4])

    def softmax(self, state, tau, e=np.exp(1)):

        qs = {}
        valid_actions = self.env.get_valid_actions(state)

        for a in valid_actions:
            qs[a] = self.Q_values[(state[0], state[1], a)]  # all the possible Q-values from our state

        # Softmax equation
        sum_q = sum([e ** (qs[a] / tau) for a in valid_actions])
        probabilities = [(a, e ** (qs[a] / tau) / sum_q) for a in valid_actions]

        r = random.random()
        accumulator = 0

        for (action, p) in probabilities:
            accumulator += p
            if accumulator >= r:
                return action

        return np.random.choice(valid_actions)

    def run(self, episodes=1000, constant_episodes=100, steps=100, tau=1.5, tau_min=1.0):

        episode = 0
        episodes_with_decay = episodes - constant_episodes
        tau_decay = (tau - tau_min) / episodes_with_decay
        self.tau_array = []

        # We user the SARSA algorithm with a Softmax policy
        while episode < episodes:

            step = 0
            done = False

            # Reset and get the starting state from the environment
            self.env.reset()
            current_state = self.env.state

            # Get the first action before we start with the loop
            current_action = self.softmax(current_state, tau)
            
            while step < steps and not done:

                # Get next state and action
                observation, reward, done, info = self.env.step(current_action)
                next_action = self.softmax(observation, tau)

                # Calculate the Reward prediction error
                rpe = reward + self.gamma * self.Q_values[observation[0], observation[1], next_action] - self.Q_values[current_state[0], current_state[1], current_action]

                # Update Q-Values
                self.Q_values[current_state[0], current_state[1], current_action] += self.alpha * rpe

                current_state = observation
                current_action = next_action

                if self.printinfo:
                    print(info)

                step += 1

            # Handle the tau decay over time
            if episode <= episodes_with_decay:
                tau -= tau_decay
            else:
                self.gamma = 1.0

            episode += 1

            # Used for visualization of the tau decay
            self.tau_array.append(tau)
            
            if self.printinfo:
                print("-------------------")

    def plot_tau_decay(self):

        plt.plot(self.tau_array)
        plt.title("Softmax τ Decay")
        plt.xlabel("Episode")
        plt.ylabel("τ value")
        plt.grid()
        plt.show()

    def plot_grid(self):

        # This is a little messy.
        # We create via Triangulation a heatmap-like plot and visualize the Points of interest
        M = self.size[0]
        N = self.size[1]

        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers

        trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]

        valuesN = np.ravel(np.swapaxes(self.Q_values[:, :, 0], 0, 1))
        valuesE = np.ravel(np.swapaxes(self.Q_values[:, :, 3], 0, 1))
        valuesS = np.ravel(np.swapaxes(self.Q_values[:, :, 1], 0, 1))
        valuesW = np.ravel(np.swapaxes(self.Q_values[:, :, 2], 0, 1))

        f_mask = self.env.final_mask.copy()
        f_mask[self.env.start[0], self.env.start[1]] = True
        mask = np.argwhere(np.ravel(np.swapaxes(f_mask, 0, 1)))

        for j, i in zip(mask[:, 0], np.arange(len(mask))):
            trianglesN.pop(j - i)
            trianglesE.pop(j - i)
            trianglesS.pop(j - i)
            trianglesW.pop(j - i)

            valuesN = np.delete(valuesN, j - i)
            valuesE = np.delete(valuesE, j - i)
            valuesS = np.delete(valuesS, j - i)
            valuesW = np.delete(valuesW, j - i)

        triangul = [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

        values = [valuesN, valuesE, valuesS, valuesW]
        cmaps = ['Oranges', 'Oranges', 'Oranges', 'Oranges']
        norms = [plt.Normalize(0, self.env.R[self.env.goal[0]][self.env.goal[1]]) for _ in range(4)]

        fig, ax = plt.subplots(figsize=(self.size[0], self.size[1]))
        ax.set_facecolor('grey')
        [ax.tripcolor(t, val, cmap=cmap, alpha=0.75, norm=norm, ec='darkslategrey') for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]

        valuesN = np.swapaxes(self.Q_values[:, :, 0], 0, 1)
        valuesE = np.swapaxes(self.Q_values[:, :, 3], 0, 1)
        valuesS = np.swapaxes(self.Q_values[:, :, 1], 0, 1)
        valuesW = np.swapaxes(self.Q_values[:, :, 2], 0, 1)
        values = [valuesN, valuesE, valuesS, valuesW]

        for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
            for i in range(M):
                for j in range(N):
                    if not f_mask[i, j]:
                        v = val[j, i]
                        if v > (self.env.R[self.env.goal[0]][self.env.goal[1]] / 2):
                            ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.1f}', color='white', ha='center', va='center')
                        else:
                            ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.1f}', color='black', ha='center', va='center')

        # Create a Rectangle patch
        start_field = patches.Rectangle((self.env.start[0] - 0.5, self.env.start[1] - 0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='tab:blue')
        goal_field = patches.Rectangle((self.env.goal[0] - 0.5, self.env.goal[1] - 0.5), 1, 1, linewidth=1, edgecolor='k', facecolor='tab:green')

        # Add the patch to the Axes
        ax.add_patch(start_field)
        ax.add_patch(goal_field)
        ax.text(self.env.start[0], self.env.start[1], "START", color='black', ha='center', va='center')
        ax.text(self.env.goal[0], self.env.goal[1], "GOAL", color='black', ha='center', va='center')

        ax.set_xticks(range(M))
        ax.set_yticks(range(N))
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        plt.show()