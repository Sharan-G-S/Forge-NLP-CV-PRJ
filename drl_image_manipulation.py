"""
Deep Reinforcement Learning for Image Manipulation

This module implements a Deep Q-Network (DQN) agent that learns to perform
image manipulation tasks such as adjusting brightness, contrast, saturation,
and applying filters to optimize image quality based on a reward signal.

Author: AI Assistant
Date: September 23, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.BoolTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network for image manipulation actions."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ImageManipulationEnvironment:
    """Environment for image manipulation tasks."""
    
    def __init__(self, target_image_path: Optional[str] = None):
        self.actions = {
            0: 'brightness_up',
            1: 'brightness_down', 
            2: 'contrast_up',
            3: 'contrast_down',
            4: 'saturation_up',
            5: 'saturation_down',
            6: 'blur',
            7: 'sharpen',
            8: 'no_action'
        }
        
        self.action_dim = len(self.actions)
        self.state_dim = 10  # Image features: histogram stats, brightness, contrast, etc.
        
        self.original_image = None
        self.current_image = None
        self.target_image = None
        self.step_count = 0
        self.max_steps = 20
        
        if target_image_path and os.path.exists(target_image_path):
            self.target_image = Image.open(target_image_path).convert('RGB')
    
    def reset(self, image_path: str = None, image_array: np.ndarray = None):
        """Reset the environment with a new image."""
        if image_path:
            self.original_image = Image.open(image_path).convert('RGB')
        elif image_array is not None:
            self.original_image = Image.fromarray(image_array.astype('uint8')).convert('RGB')
        else:
            # Generate a random noisy image for demonstration
            noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            self.original_image = Image.fromarray(noise)
        
        self.current_image = self.original_image.copy()
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Extract features from the current image to create state representation."""
        # Convert PIL to numpy array
        img_array = np.array(self.current_image)
        
        # Calculate various image statistics as features
        features = []
        
        # Brightness (mean of all channels)
        features.append(np.mean(img_array) / 255.0)
        
        # Contrast (standard deviation)
        features.append(np.std(img_array) / 255.0)
        
        # Color channel means
        for channel in range(3):
            features.append(np.mean(img_array[:, :, channel]) / 255.0)
        
        # Histogram features (simplified)
        hist = cv2.calcHist([cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)], 
                           [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_features = hist.flatten()[:5]  # Take first 5 histogram bins
        features.extend(hist_features / np.sum(hist_features))  # Normalize
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on image quality metrics."""
        img_array = np.array(self.current_image)
        
        # Basic quality metrics
        brightness = np.mean(img_array) / 255.0
        contrast = np.std(img_array) / 255.0
        
        # Reward for balanced brightness (around 0.5)
        brightness_reward = 1.0 - abs(brightness - 0.5) * 2
        
        # Reward for good contrast (not too low, not too high)
        contrast_reward = min(contrast * 4, 1.0) - max(0, contrast - 0.5) * 2
        
        # Penalty for over-processing (too many steps)
        step_penalty = -0.1 * (self.step_count / self.max_steps)
        
        # If we have a target image, compare similarity
        target_reward = 0
        if self.target_image:
            target_array = np.array(self.target_image.resize(self.current_image.size))
            mse = np.mean((img_array - target_array) ** 2) / (255.0 ** 2)
            target_reward = 1.0 - mse  # Higher reward for lower MSE
        
        total_reward = brightness_reward + contrast_reward + step_penalty + target_reward * 2
        return total_reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute an action and return the new state, reward, done flag, and info."""
        self.step_count += 1
        
        # Apply the action to modify the image
        if action == 0:  # brightness_up
            enhancer = ImageEnhance.Brightness(self.current_image)
            self.current_image = enhancer.enhance(1.1)
        elif action == 1:  # brightness_down
            enhancer = ImageEnhance.Brightness(self.current_image)
            self.current_image = enhancer.enhance(0.9)
        elif action == 2:  # contrast_up
            enhancer = ImageEnhance.Contrast(self.current_image)
            self.current_image = enhancer.enhance(1.1)
        elif action == 3:  # contrast_down
            enhancer = ImageEnhance.Contrast(self.current_image)
            self.current_image = enhancer.enhance(0.9)
        elif action == 4:  # saturation_up
            enhancer = ImageEnhance.Color(self.current_image)
            self.current_image = enhancer.enhance(1.1)
        elif action == 5:  # saturation_down
            enhancer = ImageEnhance.Color(self.current_image)
            self.current_image = enhancer.enhance(0.9)
        elif action == 6:  # blur
            self.current_image = self.current_image.filter(ImageFilter.BLUR)
        elif action == 7:  # sharpen
            self.current_image = self.current_image.filter(ImageFilter.SHARPEN)
        # action == 8 is no_action, do nothing
        
        # Calculate reward and check if episode is done
        reward = self._calculate_reward()
        done = self.step_count >= self.max_steps
        
        next_state = self._get_state()
        info = {'action_taken': self.actions[action], 'step_count': self.step_count}
        
        return next_state, reward, done, info
    
    def render(self):
        """Display the current image."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.current_image)
        plt.title(f'Current Image (Step {self.step_count})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


class DQNAgent:
    """Deep Q-Learning agent for image manipulation."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.memory_size = 10000
        self.target_update_freq = 100
        
        # Neural networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Training tracking
        self.training_step = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


def train_agent(episodes: int = 1000, render_freq: int = 100):
    """Train the DQN agent on image manipulation tasks."""
    
    # Initialize environment and agent
    env = ImageManipulationEnvironment()
    agent = DQNAgent(env.state_dim, env.action_dim)
    
    # Training metrics
    episode_rewards = []
    losses = []
    
    print(f"Starting training for {episodes} episodes...")
    
    for episode in range(episodes):
        # Reset environment with random image or specific image
        state = env.reset()  # This will create a random noisy image
        total_reward = 0
        episode_losses = []
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if episode_rewards else 0
            avg_loss = np.mean(losses[-50:]) if losses else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, "
                  f"Avg Loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.3f}")
        
        # Render occasionally
        if episode % render_freq == 0 and episode > 0:
            print(f"\nRendering episode {episode}...")
            env.render()
    
    return agent, episode_rewards, losses


def test_agent(agent: DQNAgent, test_image_path: str = None, num_steps: int = 20):
    """Test the trained agent on image manipulation."""
    
    env = ImageManipulationEnvironment()
    
    if test_image_path:
        state = env.reset(test_image_path)
    else:
        state = env.reset()  # Random image
    
    print("Testing trained agent...")
    env.render()
    
    total_reward = 0
    for step in range(num_steps):
        action = agent.act(state, training=False)  # No exploration
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step + 1}: Action = {info['action_taken']}, Reward = {reward:.3f}")
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    print(f"\nFinal result after {step + 1} steps:")
    print(f"Total reward: {total_reward:.3f}")
    env.render()
    
    return total_reward


def create_sample_images():
    """Create sample images for testing."""
    
    # Create a dark image
    dark_img = np.ones((256, 256, 3), dtype=np.uint8) * 50
    dark_pil = Image.fromarray(dark_img)
    dark_pil.save('/Users/sharan/TEST/dark_sample.jpg')
    
    # Create a bright image  
    bright_img = np.ones((256, 256, 3), dtype=np.uint8) * 200
    bright_pil = Image.fromarray(bright_img)
    bright_pil.save('/Users/sharan/TEST/bright_sample.jpg')
    
    # Create a low contrast image
    low_contrast = np.ones((256, 256, 3), dtype=np.uint8) * 128
    noise = np.random.normal(0, 10, (256, 256, 3))
    low_contrast_img = np.clip(low_contrast + noise, 0, 255).astype(np.uint8)
    low_contrast_pil = Image.fromarray(low_contrast_img)
    low_contrast_pil.save('/Users/sharan/TEST/low_contrast_sample.jpg')
    
    print("Sample images created!")


def train_and_test_with_image(image_path: str, episodes: int = 500):
    """Train and test the agent with a specific uploaded image."""
    
    print("Deep Reinforcement Learning for Image Manipulation")
    print("=" * 50)
    print(f"Using uploaded image: {image_path}")
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Train the agent
    print("\nTraining DQN agent...")
    agent, rewards, losses = train_agent(episodes=episodes, render_freq=max(episodes//5, 50))
    
    # Save the trained model
    model_path = '/Users/sharan/TEST/dqn_image_manipulation.pth'
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Test the agent on the uploaded image
    print(f"\nTesting on uploaded image: {image_path}")
    total_reward = test_agent(agent, image_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/sharan/TEST/training_curves.png')
    plt.show()
    
    print(f"\nTraining complete! Final reward on uploaded image: {total_reward:.3f}")
    return agent, total_reward


def load_and_test_image(image_path: str, model_path: str = '/Users/sharan/Downloads/Lewis Hamilton.jpeg'):
    """Load a pre-trained model and test it on an uploaded image."""
    
    print(f"Loading pre-trained model from {model_path}")
    
    # Verify files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train a model first using train_and_test_with_image()")
        return
    
    # Initialize environment and agent
    env = ImageManipulationEnvironment()
    agent = DQNAgent(env.state_dim, env.action_dim)
    
    # Load the trained model
    agent.load(model_path)
    
    # Test on the uploaded image
    print(f"Testing pre-trained agent on: {image_path}")
    total_reward = test_agent(agent, image_path)
    
    print(f"Final reward: {total_reward:.3f}")
    return total_reward


def quick_enhance_image(image_path: str, save_path: str = None):
    """Quickly enhance an image using a simple rule-based approach (no training needed)."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    
    # Analyze image properties
    img_array = np.array(img)
    brightness = np.mean(img_array) / 255.0
    contrast = np.std(img_array) / 255.0
    
    print(f"Original image stats - Brightness: {brightness:.3f}, Contrast: {contrast:.3f}")
    
    # Apply enhancements based on analysis
    if brightness < 0.4:  # Too dark
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.3)
        print("Applied brightness enhancement")
    elif brightness > 0.7:  # Too bright
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
        print("Applied brightness reduction")
    
    if contrast < 0.15:  # Low contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.4)
        print("Applied contrast enhancement")
    elif contrast > 0.4:  # Too high contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.8)
        print("Applied contrast reduction")
    
    # Always apply slight sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    print("Applied sharpening")
    
    # Display results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('Enhanced Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save enhanced image if path provided
    if save_path:
        img.save(save_path)
        print(f"Enhanced image saved to: {save_path}")
    
    # Show final stats
    enhanced_array = np.array(img)
    final_brightness = np.mean(enhanced_array) / 255.0
    final_contrast = np.std(enhanced_array) / 255.0
    print(f"Enhanced image stats - Brightness: {final_brightness:.3f}, Contrast: {final_contrast:.3f}")
    
    return img


def display_uploaded_image(image_path: str):
    """Display an uploaded image and show its properties."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Load and display image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Calculate image properties
    brightness = np.mean(img_array) / 255.0
    contrast = np.std(img_array) / 255.0
    
    print(f"\nImage: {image_path}")
    print(f"Size: {img.size[0]} x {img.size[1]} pixels")
    print(f"Brightness: {brightness:.3f} (0=dark, 1=bright)")
    print(f"Contrast: {contrast:.3f} (higher = more contrast)")
    
    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Uploaded Image\nBrightness: {brightness:.3f}, Contrast: {contrast:.3f}')
    plt.axis('off')
    plt.show()
    
    return img


def check_for_images():
    """Check for uploaded images in the TEST directory."""
    
    test_dir = '/Users/sharan/Downloads/Lewis Hamilton.jpeg'
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    found_images = []
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                if not file.startswith('sample') and not file.startswith('.'):
                    found_images.append(os.path.join(test_dir, file))
    
    if found_images:
        print("\n📸 Found uploaded images:")
        for i, img_path in enumerate(found_images, 1):
            print(f"  {i}. {os.path.basename(img_path)}")
        
        print("\n🚀 Ready to use! Try these commands:")
        for img_path in found_images[:2]:  # Show first 2 examples
            img_name = os.path.basename(img_path)
            print(f"  quick_enhance_image('{img_path}')  # Fast enhancement")
            print(f"  display_uploaded_image('{img_path}')  # View image properties")
        
        return found_images
    else:
        print("\n📋 No uploaded images found in /Users/sharan/TEST/")
        print("   Upload your image there and run the script again!")
        return []


def main():
    """Main function with options for different workflows."""
    
    print("Deep Reinforcement Learning for Image Manipulation")
    print("=" * 50)
    
    # Check for uploaded images first
    found_images = check_for_images()
    
    print("\n🔧 Available functions:")
    print("  • display_uploaded_image(path) - View image and properties")
    print("  • quick_enhance_image(path) - Fast AI enhancement (no training)")
    print("  • train_and_test_with_image(path) - Full DRL training")
    print("  • load_and_test_image(path) - Use pre-trained model")
    
    # Create sample images for demonstration
    print("\n⚡ Creating sample images for demonstration...")
    create_sample_images()
    
    # If images found, show the first one
    if found_images:
        print(f"\n🖼️  Displaying first uploaded image...")
        display_uploaded_image(found_images[0])
        print(f"\n💡 Try: quick_enhance_image('{found_images[0]}')")


if __name__ == "__main__":
    main()