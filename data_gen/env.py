"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

# Get env directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import time
import argparse
from multiprocessing import Pool

# noinspection PyUnresolvedReferences
import envs

import utils

import gym
from gym import logger

import numpy as np
from PIL import Image


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.ANTIALIAS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255


def rollout(**kwargs):
    reward = 0
    done = False
    crop = kwargs.get('crop')

    trace = {
        'obs': [],
        'action': [],
            'next_obs': [],
    }

    ob = env.reset()

    if args.atari:
        # Burn-in steps
        for _ in range(warmstart):

            action = agent.act(ob, reward, done)
            ob, _, _, _ = env.step(action)
        prev_ob = crop_normalize(ob, crop)
        ob, _, _, _ = env.step(0)
        ob = crop_normalize(ob, crop)

        while True:
            trace['obs'].append(
                np.concatenate((ob, prev_ob), axis=0))
            prev_ob = ob

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            ob = crop_normalize(ob, crop)

            trace['action'].append(action)
            trace['next_obs'].append(
                np.concatenate((ob, prev_ob), axis=0))

            if done:
                break
    else:

        while True:
            trace['obs'].append(ob[1])

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            trace['action'].append(action)
            trace['next_obs'].append(ob[1])

            if done:
                break
    
    return trace


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='ShapesTrain-v0',
                        help='Select the environment to run.')
    parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='Run atari mode (stack multiple frames).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of rollout workers')
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    env.seed(args.seed)

    agent = RandomAgent(env.action_space)

    episode_count = args.num_episodes

    crop = None
    warmstart = None
    if args.env_id == 'PongDeterministic-v4':
        crop = (35, 190)
        warmstart = 58
    elif args.env_id == 'SpaceInvadersDeterministic-v4':
        crop = (30, 200)
        warmstart = 50

    if args.atari:
        env._max_episode_steps = warmstart + 11

    def do_episode(i):
        return rollout(crop=crop)
        
    replay_buffer = []
    start = time.time()
    with Pool(args.num_workers) as p:
        for run in p.imap(do_episode, range(episode_count)):
            replay_buffer.append(run)
            display_data = len(replay_buffer), episode_count, int(time.time() - start)
            print('{}/{} episodes complete in {}s ({}s to go!)'.format(
                *display_data, int(utils.time_remaining(*display_data))))

    env.close()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname)
