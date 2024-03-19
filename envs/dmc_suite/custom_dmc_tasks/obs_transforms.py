import numpy as np

def obs_transform_cheetah(env, obs=None):
    t_obs = env._env.physics._named.data.xpos['torso'][[0,2]]
    return np.copy(t_obs).astype(np.float32)

def obs_transform_walker(env, obs=None):
    t_obs = env._env.physics._named.data.xpos['torso'][[0,2]]
    return np.copy(t_obs).astype(np.float32)

def obs_transform_hopper(env, obs=None):
    t_obs = env._env.physics._named.data.xpos['torso'][[0,2]]
    return np.copy(t_obs).astype(np.float32)

def obs_transform_quadruped(env, obs=None):
    t_obs = env._env.physics._named.data.xpos['torso'][[0,1]]
    return np.copy(t_obs).astype(np.float32)

def obs_transform_default(env, obs):
    return obs

OBS_TRANSFORMS = {
    "cheetah": obs_transform_cheetah,
    "hopper": obs_transform_hopper,
    "walker": obs_transform_walker,
    "quadruped": obs_transform_quadruped
}