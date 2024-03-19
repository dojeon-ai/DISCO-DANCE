import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import imageio
import omegaconf
import hydra
import numpy as np
import torch
import wandb
from pathlib import Path
from collections import OrderedDict, deque

from dm_env import specs
import common.utils as utils
from common.logger import Logger
from common.replay_buffer import ReplayBufferStorage, make_replay_loader
from common.video import VideoRecorder
import envs.make_maze as make_maze
        
torch.backends.cudnn.benchmark = True

list_2d_maze = ['square_bottleneck', 'square_a', 'square_b', 'square_c', 'square_empty']


def make_agent(obs_type, obs_spec, action_spec, action_range, num_expl_steps, 
               maze_parent, save_dir, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.action_range = action_range  
    cfg.num_expl_steps = num_expl_steps 
    cfg.maze_parent = maze_parent
    cfg.save_dir = save_dir
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):

        #########################
        # 1. Experimental Setting
        #########################
        self.cfg = cfg
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        # gpu setting
        self.device = torch.device(cfg.device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device_id)

        utils.set_seed_everywhere(cfg.seed)

        ################
        # 2. Environment
        ################
        # init train, test env
        self.maze_type = cfg.maze_type
        if self.maze_type in list_2d_maze:
            self.maze_parent = '2dmaze'
        else:
            raise NameError('Wrong maze_type')
        self.train_env, self.test_env = make_maze.make(self.maze_parent, self.maze_type,
                                        cfg.maximum_timestep, cfg.dtype, is_pretrain=True)

        ##########
        # 3. Agent
        ###########
        # init agent
        self.save_path = '_'.join([cfg.agent.name, cfg.maze_type, 
                            str(cfg.agent.skill_dim),
                            str(cfg.seed)
                            ])
        save_dir = self.get_dir(f'{self.save_path}')
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.train_env.action_range(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                self.maze_parent,
                                str(save_dir),
                                cfg.agent
                                )
        # data_spec 
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array((2,), np.float32, 'xy_p'))
        # meta_spec 
        meta_specs = self.agent.get_meta_specs()
        # init replay buffer
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs, maze_parent=self.maze_parent,
                                                  replay_dir = self.work_dir / 'buffer', use_prior = self.agent.use_prior)
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None
        
        ###########
        # 4. Logger
        ###########
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        self.exp_name = '_'.join([cfg.agent.name, cfg.maze_type, 
                            str(cfg.agent.skill_dim),
                            str(cfg.seed)
                            ])
        # wandb logger
        if cfg.use_wandb:
            # hydra -> wandb config
            config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            wandb.init(project="DISCO", group=cfg.agent.name, name=self.exp_name, config=config)
            wandb.init(settings=wandb.Settings(start_method="thread"))  

        ########
        # 5. etc
        ########
        # timer
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    # Do not touch property ft
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter


    def init_train_env(self):
        '''
        num_train_frames: # of pretraining step
        action_repeat: repeat action_repeat times for selected action
        '''
        # total_train_step / exploration_step / eval_period
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # reset env
        init_ts = self.train_env.reset()
        meta = self.agent.init_meta()

        return train_until_step, seed_until_step, eval_every_step, init_ts, meta


    def train(self):
        '''
        ts: class {
            'observation': np.array,
            'action': np.array,
            'reward': 0.0,
            'discount': 1.0,
            'final': bool,
            'maximum_timestep': int,
            'timestpes_so_far': episode에서의 step
        }
        meta: dict {
            'skill': np.array
        }
        '''
        train_until_step, seed_until_step, eval_every_step, ts, meta = self.init_train_env()

        ts.action = np.zeros(self.agent.action_dim, dtype=ts.observation.dtype)
        self.replay_storage.add(ts, meta)
        
        episode_step, episode_reward = 0, 0
        metrics = None

        while train_until_step(self.global_step):

            # 1. sample action and step 
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(ts.observation, meta, self.global_step, eval_mode=False)
            ts = self.train_env.step(action)
            
            # 2. store transition 
            ts.action = action
            self.replay_storage.add(ts, meta)

            episode_reward += ts.reward

            # 3. update agent 
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.train_env, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')  # log every ts

            # 4. when episode ends, 0) add to buffer, 1) log, 2) reset env 
            if ts.last(): 
                ### 1) log ###
                self._global_episode += 1

                D_delta, _ = self.agent.compute_delta(
                                            ts.observation, 
                                            meta['skill']
                                        )
                self.agent.update_D_delta(meta['skill'], D_delta, self.global_step)
            
                # log to wandb
                if not seed_until_step(self.global_step):
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)  
                        log('total_time', total_time)  
                        log('episode_reward', episode_reward)  
                        log('episode', self.global_episode)  
                        log('buffer_size', len(self.replay_storage))

                episode_step = 0
                episode_reward = 0

                ### 2) reset ###
                # reset env
                ts = self.train_env.reset()

                meta = self.agent.init_meta(self.agent.delta_dict)

                ts.action = np.zeros(self.agent.action_dim, dtype=ts.observation.dtype)
                self.replay_storage.add(ts, meta)

            # 6. evaluate
            if eval_every_step(self.global_step):
                self.pretrain_evaluate()
            
            episode_step += 1
            self._global_step += 1

            # 7. save model
            if self.global_frame % self.cfg.snapshots_interval == 0:
                self.save_snapshot()

    def pretrain_evaluate(self):
        '''
        iterate all (for discrete z) skills.
        calculate:
            1. video (for mujoco)
            2. state_coverage (value, 2d plot) 
            3. num_learned_skill 
            4. (should do) downstream task performance

        meta: dict {
            'skill': np.array
        }
        '''
        episode, total_reward = 0, 0
        meta_all = self.agent.init_all_meta()
        meta, trajectory_all = OrderedDict(), OrderedDict()

        num_eval_each_skill = 5
        num_eval_skills = self.agent.skill_dim

        for episode in range(num_eval_skills):
            # iterate all skills
            meta['skill'] = meta_all['skill'][episode]
            trajectory = list()

            for idx in range(num_eval_each_skill): 
                ts = self.test_env.reset()

                while not ts.last():
                    # 1. sample action and step
                    prev_obs = ts.observation
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(ts.observation, meta, self.global_step, eval_mode=True)    
                    
                    ts = self.test_env.step(action)
                    
                    # 2. save trajectory for (2)state_coverage_plot
                    trajectory.append([[prev_obs[0].item(), ts.observation[0].item()], 
                                    [prev_obs[1].item(), ts.observation[1].item()]])

                    # 3. save rw
                    total_reward += ts.reward

            # for (2)state_coverage_plot 
            trajectory_all[episode] = trajectory 

        save_dir = self.get_dir(f'{self.exp_name}/{self.exp_name}_{self.global_frame}.png')
        # (2) state_coverage_plot
        self.agent.plot(env = self.test_env,
                        trajectory = trajectory_all, 
                        save_dir = save_dir,
                        step = self.global_step,
                        use_wandb = self.cfg.use_wandb)

        state_coverage = self.test_env.state_coverage(trajectory_all=trajectory_all,
                                                      skill_dim=self.agent.skill_dim)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / (episode*num_eval_each_skill))
            log('episode', self.global_episode)
            log('step', self.global_step)
            log(f'state_coveraged(out of 100 bucekts)', state_coverage)


    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def get_dir(self, file_name):
        save_dir = self.work_dir / 'eval_video'
        dir_name = file_name.split('/')[0]
        if not os.path.exists(save_dir / dir_name):
            os.makedirs(save_dir / dir_name)
        path = save_dir / file_name

        return path


@hydra.main(config_path='.', config_name='pretrain_maze')
def main(cfg):
    from pretrain_maze import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
