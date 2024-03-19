import ipdb
import torch
import numpy as np

from envs.maze.maze_env import Env

from dm_env import specs
import numpy as np



class MakeTimestep:
    def __init__(self, maze_parent, state, maximum_timestep, 
                timesteps_so_far=None, prev_obs=None, action=None, reward=None, goal=None):
        self.maze_parent = maze_parent
        self._state = state
        self.maximum_timestep = maximum_timestep
        self.final = None
        self.discount = 0.99
        self.action = action

        self.timesteps_so_far = self._state['n']
        self.observation = self._state['state'].numpy()
        self.prev_observation = self._state['prev_state'].numpy()
        self.reward = 0.0

    def last(self):

        if self.timesteps_so_far != 0 and (self.timesteps_so_far % self.maximum_timestep == 0):
            return True
        else:
            return False


class DMCStyleWrapper:
    def __init__(self, env=None, maximum_timestep=None, maze_parent=None, obs_dtype=np.float32, maze_type=None):
        self._env = env
        self.maximum_timestep = maximum_timestep
        self.maze_parent = maze_parent
        self.obs_dtype = obs_dtype
        self.maze_type = maze_type

        self.obs_space = (self._env.state.shape[0],)
        self.action_space = (self._env.action_size, )
        self.act_range = self._env.action_range

    def observation_spec(self):
        return specs.Array(self.obs_space, self.obs_dtype, 'observation')
    
    def final_spec(self):
        return specs.Array(self.obs_space, self.obs_dtype, 'final')

    def action_spec(self):
        return specs.BoundedArray(self.action_space, self.obs_dtype, -self.act_range, self.act_range, 'action')

    def action_range(self):
        return self.act_range

    def reset(self, state=None):
        self._env.reset()
        return MakeTimestep(self.maze_parent, self._env._state, self.maximum_timestep)
    
    def step(self, action):
        self._env.step(action)
        return MakeTimestep(self.maze_parent, self._env._state, self.maximum_timestep)

    def plot_trajectory(self, trajectory, save_dir, step, use_wandb, guide_idx = None, delta_dict = None):
        self._env.maze.plot_trajectory(trajectory_all=trajectory, save_dir = save_dir, step = step, use_wandb=use_wandb, guide_idx = guide_idx, maze_type = self.maze_type, delta_dict = delta_dict)

    def plot_guide(self, trajectory, save_dir, step, use_wandb, guide_idx = None, terminal_idx = None):
        self._env.maze.plot_guide(trajectory_all=trajectory, save_dir = save_dir, step = step, use_wandb=use_wandb, guide_idx = guide_idx, terminal_idx = terminal_idx, maze_type = self.maze_type)

    def plot_knn(self, top1, observations, all_samples, save_dir, use_wandb=False, skill_dim = None):
        self._env.maze.plot_knn(top1=top1, obs = observations, all_samples = all_samples, save_dir = save_dir, use_wandb=use_wandb, skill_dim = skill_dim, maze_type = self.maze_type)
        
    def state_coverage(self, trajectory_all, skill_dim):

        state_cov_avg = self._env.maze.state_coverage(trajectory_all=trajectory_all, skill_dim=skill_dim, maze_type = self.maze_type)
        
        return state_cov_avg

    def plot(self):
        self._env.maze.plot()


def make(maze_parent=None,
        maze_type=None,
        maximum_timestep=None,
        dtype=None,
        is_pretrain=True,
        random=False,
        num_skills=None,
        train_random=False,
        goal_idx=None,
        seed=0,
        guide_env=False,
        task=None):

    train_env = Env(n = maximum_timestep, maze_type = maze_type, random=random, num_skills=num_skills, train_random = train_random)
    train_env = DMCStyleWrapper(train_env, maximum_timestep, maze_parent=maze_parent, maze_type=maze_type)
    test_env = Env(n = maximum_timestep, maze_type = maze_type, random=random, num_skills=num_skills, train_random = train_random)
    test_env = DMCStyleWrapper(test_env, maximum_timestep, maze_parent=maze_parent, maze_type=maze_type)

    return train_env, test_env