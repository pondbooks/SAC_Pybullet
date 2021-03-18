import trainer
import sac
import fixed_seed

import numpy as np 
import torch
import random

import gym
gym.logger.set_level(40)
import pybullet_envs


def main():
    ENV_ID = 'ReacherBulletEnv-v0'
    SEED = 0
    REWARD_SCALE = 1.0
    NUM_STEPS =  10 ** 6
    EVAL_INTERVAL = 10 ** 4

    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)
    
    # シードを設定する．
    fixed_seed.fix_seed(SEED)
    # 環境の乱数シードを設定する．
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    env_test.seed(2**31-SEED)
    env_test.action_space.seed(2**31-SEED)
    env_test.observation_space.seed(2**31-SEED)
    
    env_test.render(mode="human")

    algo = sac.SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        seed=SEED,
        reward_scale=REWARD_SCALE,
    )

    SACtrainer = trainer.Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        seed=SEED,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL,
    )

    SACtrainer.train()
    SACtrainer.plot()

if __name__ == "__main__":
    main()