import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from PIL import Image
from torchvision import transforms


class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)

        # Fully connected layers
        self.fc1 = nn.Linear(9 * 9 * 512, 1024)  # Adjusted input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, action_size)

    def forward(self, state):
        #print("Input shape:", state.shape)

        x = F.relu(self.bn1(self.conv1(state)))
        #print("After conv1 and bn1:", x.shape)

        x = F.relu(self.bn2(self.conv2(x)))
        #print("After conv2 and bn2:", x.shape)

        x = F.relu(self.bn3(self.conv3(x)))
        #print("After conv3 and bn3:", x.shape)

        x = F.relu(self.bn4(self.conv4(x)))
        #print("After conv4 and bn4:", x.shape)

        x = F.relu(self.bn5(self.conv5(x)))
        #print("After conv5 and bn5:", x.shape)

        x = F.relu(self.bn6(self.conv6(x)))
        #print("After conv6 and bn6:", x.shape)

        x = F.relu(self.bn7(self.conv7(x)))
        #print("After conv7 and bn7:", x.shape)

        x = F.relu(self.bn8(self.conv8(x)))
        #print("After conv8 and bn8:", x.shape)

        x = F.relu(self.bn9(self.conv9(x)))
        #print("After conv9 and bn9:", x.shape)

        x = F.relu(self.bn10(self.conv10(x)))
        #print("After conv10 and bn10:", x.shape)

        x = x.view(x.size(0), -1)
        #print("After flattening:", x.shape)

        x = F.relu(self.fc1(x))
        #print("After fc1:", x.shape)

        x = F.relu(self.fc2(x))
        #print("After fc2:", x.shape)

        x = F.relu(self.fc3(x))
        #print("After fc3:", x.shape)

        x = F.relu(self.fc4(x))
        #print("After fc4:", x.shape)

        x = F.relu(self.fc5(x))
        #print("After fc5:", x.shape)

        output = self.fc6(x)
        #print("Output shape:", output.shape)

        return output


    
# Setting up the enviroment
env = gym.make('MsPacmanDeterministic-v0', full_action_space = False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print("State shape:", state_shape)
print("State shape:", state_size)
print("State shape:", number_actions)

# Initializing the hyperparameters
learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99

# Preporcessing the frames
def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)

# Implementing the DCQN class
class Agent():

  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnetwork = Network(action_size).to(self.device)
    self.target_qnetwork = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = deque(maxlen = 10000)

  def step(self, state, action, reward, next_state, done):
    state = preprocess_frame(state)
    next_state = preprocess_frame(next_state)
    self.memory.append((state, action, reward, next_state, done))
    if len(self.memory) > minibatch_size:
       experiences = random.sample(self.memory, k = minibatch_size)
       self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = preprocess_frame(state).to(self.device)
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state)
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, discount_factor):
    states, actions, rewards, next_states, dones = zip(*experiences)
    states = torch.from_numpy(np.vstack(states)).float().to(self.device)
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

# Initializing the DCQN agent
agent = Agent(number_actions)

# Training the DCQN agent
number_episodes = 6000 # OG: 2000
maximum_number_timesteps_per_episode = 10000 # OG: 10000
epsilon_starting_value = 1.0 # OG: 1.0
epsilon_ending_value = 0.01 # OG: 0.01
epsilon_decay_value = 0.995 # OG: 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)

for episode in range(1, number_episodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
  print("\rEpisode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print("\rEpisode {}\tAverage Score: {:.2f}".format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 1000.0: # OG: 500
    print("\nEnviroment solved in {:d} episodes!\tAverage Score: {:.2f}".format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), "checkpoint.pth")
    break

# Visualizing
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanDeterministic-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()