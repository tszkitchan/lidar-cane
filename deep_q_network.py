import gymnasium as gym
import torch
import numpy as np
from collections import deque
import random
import os

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=360, shape=(2, 360), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  # Example: 4 possible actions

    def reset(self):
        # Define the path to the data folder
        data_folder = "data/"
        # Get a list of all files in the folder
        files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
        # Randomly select a file
        selected_file = np.random.choice(files)
        # Load the data from the selected file
        data = np.loadtxt(os.path.join(data_folder, selected_file), delimiter=',')

        # Initialize the state with zeros
        self.state = np.zeros((2, 360), dtype=np.float32)
        for angle in range(360):
            self.state[0, angle] = int(angle)
        for angle, radius in data:
            self.state[1, int(angle)] = radius
        return self.state, {}

    def step(self, action):
        action = 90 * action + 45  # Example action mapping
        # reward = np.random.rand()  # Example reward
        # Check if any value in the range [action - 90, action + 90] in the second index of the state is > 0
        # if np.any(self.state[1, max(0, action - 44):min(360, action + 44)] > 0):
        #     reward = 0
        # else:
        #     # reward = np.random.rand()  # Example reward
        #     reward = 100
        if self.state[1, action] > 0:
            reward = -100
        else:
            reward = 100
        self.counter = getattr(self, 'counter', 0) + 1  # Increment counter or initialize it
        done = self.counter >= 360  # Termination condition when counter reaches 360
        if done:
            self.counter = 0  # Reset counter when done
        self.state, _ = self.reset()  # Reset state for the next step 
        self.last_action = action / 90
        return self.state, reward, done, False, {}

    def render(self):
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        theta = np.deg2rad(self.state[0, :])  # Convert angles to radians
        radius = self.state[1, :]  # Radii from the state
        # ax.plot(theta, radius, label="State")
        ax.scatter(theta, radius, s=10, c='blue', alpha=0.7)
        ax.scatter(np.deg2rad(self.last_action * 90), 1, color='red', label="Action", s=100)  # Plot the action
        ax.set_rmax(12)  # Fix the distance axis to a maximum value of 12
        ax.legend()
        plt.draw()  # Draw the plot
        plt.pause(0.5)  # Pause briefly to allow the plot to update
        plt.clf()  # Clear the figure for the next frame
        plt.close()

    def close(self):
        pass

# Define the CNN+LSTM+Attention model
class DQNModel(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        # CNN expects input shape (batch_size, channels, sequence_length)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Change to (batch_size, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc(context_vector)

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, input_dim, num_actions, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQNModel(input_dim, num_actions)
        self.target_model = DQNModel(input_dim, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(10000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        print(f"Epsilon: {self.epsilon:.4f}")

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Main training loop
def train_dqn(env_name, num_episodes=500, batch_size=64, target_update=10):
    # env = gym.make(env_name)
    env = env_name
    input_dim = (2, 360)  # (channels, sequence_length)
    num_actions = env.action_space.n
    agent = DQNAgent(input_dim, num_actions, 0.1, 0.99, 1.0, 0.9, 0.01)
    # agent.model.load_state_dict(torch.load("dqn_model_weights-5.pth")) # init weights

    for episode in range(0, num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train(batch_size)
            env.render()

            if done:
                break

        if episode % target_update == 0:
            agent.update_target_model()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        # Save the model weights
        torch.save(agent.model.state_dict(), f"dqn_model_weights-{episode+1}.pth")

    env.close()

train_dqn(CustomEnv())

def test_dqn(env_name, model_weights_path, num_episodes=1):
    env = env_name
    input_dim = (2, 360)  # (channels, sequence_length)
    num_actions = env.action_space.n
    model = DQNModel(input_dim, num_actions)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward

            env.render()

            if done:
                break

        print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

# test_dqn(CustomEnv(), "dqn_model_weights-1.pth")