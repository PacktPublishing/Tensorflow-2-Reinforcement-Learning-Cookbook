from gym.envs.registration import register

register(
    id="Maze-v0", entry_point="envs.maze:MazeEnv",
)
register(
    id="Gridworld-v2", entry_point="envs.gridworldv2:GridworldV2Env",
)
