import numpy as np
import random

class GridWorld:
    """Represents a grid world environment for reinforcement learning.

    Attributes:
        size (int): The size of the grid (e.g., 5x5).
        state (list): The current state of the agent as a 2D coordinate.
        goal (list): The goal state coordinates.
        done (bool): Indicates whether the episode is finished.
    """

    def __init__(self, size=5):
        """Initializes the grid world environment.

        Args:
            size (int, optional): The size of the grid. Defaults to 5.
        """
        self.size = size
        self.state = [0, 0]  # Start at the top-left corner
        self.goal = [size - 1, size - 1]  # Goal is at the bottom-right corner
        self.done = False

    def reset(self):
        """Resets the environment to the starting state.

        Returns:
            list: The initial state of the agent.
        """
        self.state = [0, 0]
        self.done = False
        return self.state

    def step(self, action):
        """Takes a step in the environment based on the given action.

        Args:
            action (int): The action to take (0: up, 1: right, 2: down, 3: left).

        Returns:
            tuple: A tuple containing the next state, reward, and done flag.
        """
        # Update the state based on the action, ensuring it stays within the grid
        if action == 0:
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1:
            self.state[1] = min(self.size - 1, self.state[1] + 1)
        elif action == 2:
            self.state[0] = min(self.size - 1, self.state[0] + 1)
        elif action == 3:
            self.state[1] = max(0, self.state[1] - 1)

        # Calculate the reward and check if the goal is reached
        reward = -1  # Small negative reward for each step
        if self.state == self.goal:
            reward = 100  # Big positive reward for reaching the goal
            self.done = True

        return self.state, reward, self.done

class QLearningAgent:
    """Represents a Q-learning agent for reinforcement learning.

    Attributes:
        q_table (numpy.ndarray): The Q-table storing estimated action values.
        learning_rate (float): The learning rate for updating the Q-table.
        discount_factor (float): The discount factor for future rewards.
        epsilon (float): The exploration rate for epsilon-greedy action selection.
        action_size (int): The number of possible actions.
    """

    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """Initializes the Q-learning agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The number of possible actions.
            learning_rate (float, optional): The learning rate. Defaults to 0.1.
            discount_factor (float, optional): The discount factor. Defaults to 0.95.
            epsilon (float, optional): The exploration rate. Defaults to 0.1.
        """
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.action_size = action_size

    def get_action(self, state):
        """Chooses an action based on the current state using epsilon-greedy policy.

        Args:
            state (list): The current state of the agent.

        Returns:
            int: The chosen action.
        """
        if random.random() < self.epsilon:
            # Exploration: Choose a random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: Choose the action with the highest Q-value
            return np.argmax(self.q_table[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        """Updates the Q-table based on the current experience.

        Args:
            state (list): The current state.
            action (int): The chosen action.
            reward (float): The received reward.
            next_state (list): The next state.
        """
        current_q = self.q_table[state[0], state[1], action]
        next_max_q = np.max(self.q_table[next_state[0], next_state[1]])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state[0], state[1], action] = new_q

def train(episodes=1000):
    """Trains the Q-learning agent on the grid world environment.

    Args:
        episodes (int, optional): The number of training episodes. Defaults to 1000.

    Returns:
        QLearningAgent: The trained Q-learning agent.
    """
    env = GridWorld()
    agent = QLearningAgent(env.size, 4)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while not env.done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return agent

def play_game(agent):
    """Plays a game with the trained Q-learning agent.

    Args:
        agent (QLearningAgent): The trained Q-learning agent.
    """
    env = GridWorld()
    state = env.reset()
    total_reward = 0

    while not env.done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        print(f"State: {state}, Action: {action}, Reward: {reward}")

    print(f"Game finished! Total Reward: {total_reward}")

if __name__ == "__main__":
    trained_agent = train()
    play_game(trained_agent)
