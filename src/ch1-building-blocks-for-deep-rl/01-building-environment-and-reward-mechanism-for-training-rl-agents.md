# 1.01 Building Environment and Reward mechanism for training RL agents

This recipe notebook will walk you through the steps to build a learning environment to train RL agents. These steps will help you build custom RL learning environments for your problems that you wish to solve using RL. Implementing a learning environment with a reward mechanism for reinforcement learning. As a concrete example, this recipe will walk through the steps to build a Gridworld RL learning environment. The Gridworld is a simple environment where the world is represented as a grid and each cell in the grid can be referenced using a unique coordinate location. The goal of an agent in this environment is to find it's way to the goal state. A sample environment state of the Gridworld is shown below:

![fig-1.1-screenshot-of-the-gridworld-env.png](attachment:fig-1.1-screenshot-of-the-gridworld-env.png)

The agent's current location is represented by the blue-colored cell in the grid, while the goal cell is colored green and the red-colored cell in the grid represents a bomb/mine/obstacle that the agent is supposed to avoid stepping.

```{.python .input  n=3}
# Create a sample environment
```

```{.python .input}

```
