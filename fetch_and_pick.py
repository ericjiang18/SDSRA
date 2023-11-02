'''
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
import matplotlib.pyplot as plt

# Define the neural network for Q-values approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Parameters
num_episodes = 5000
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
REPLAY_MEMORY_SIZE = 10000
UPDATE_TARGET_EVERY = 100

# Create environment
env = gym.make('CartPole-v1')

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Q and target networks
q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

replay_memory = []
episode_rewards = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = torch.argmax(q_net(state_tensor)).item()

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))
        if len(replay_memory) > REPLAY_MEMORY_SIZE:
            replay_memory.pop(0)
        
        # Train the network
        if len(replay_memory) >= BATCH_SIZE:
            batch = random.sample(replay_memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            not_dones = torch.FloatTensor([not done for done in dones]).to(device)
            
            q_values = q_net(states).gather(1, actions)
            with torch.no_grad():
                target_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * target_q_values * not_dones)
            
            loss = criterion(q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if episode % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

        state = next_state

    # Decrease epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(episode_reward)
    
    print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

    if episode_reward > 200:
        print("Stopping training as we've reached a reward greater than 200!")
        break

env.close()

# Plotting the rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward Over Time')
plt.show()
'''

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac.policies import MlpPolicy

class CustomMLPPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMLPPolicy, self).__init__(*args, **kwargs)

        # Define a custom network architecture
        self.features_extractor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def _forward(self, obs, deterministic=False):
        # Custom forward pass
        return super()._forward(obs, deterministic)

# Create the environment
env = gym.make('HalfCheetah-v2')
env = DummyVecEnv([lambda: env])  # Wrap it for Stable Baselines3

# Create the agent
model = SAC(CustomMLPPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=1000000)

# Save the agent
model.save("sac_halfcheetah")

# Test the trained agent
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
