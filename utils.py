import os
import json
from collections import defaultdict
from typing import List
import numpy as np
import torch
import gym


def defaultdict_gym_envs() -> defaultdict:
    envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        envs[env_type].add(env.id)
    return envs


def list_gym_envs() -> List[str]:
    envs = []
    for k, v in defaultdict_gym_envs().items():
        for x in v:
            envs.append(x)
    return envs


def np2torch_obs(obs: np.ndarray, low: np.ndarray, high: np.ndarray) -> torch.tensor:
    obs = (obs - low) / (high - low)
    obs = obs.astype(np.float32)
    if len(obs.shape) == 4:
        obs = np.transpose(obs, (0, 3, 1, 2))
    elif len(obs.shape) == 3:
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.expand_dims(obs, 0)
    obs = torch.from_numpy(obs)

    return obs


def general_log_msg(**kwargs):
    messages = []
    for key, value in kwargs.items():
        if isinstance(value, float):
            messages.append("{0}={1:.6f}".format(key, value))
        else:
            messages.append("{}={}".format(key, value))

    frame_width = max([len(m) for m in messages])

    msg = '\n|' + '-' * frame_width + '|\n'
    for m in messages:
        msg += '|'
        msg += m
        msg += ' ' * (frame_width - len(m)) + '|\n'
    msg += '|' + '-' * frame_width + '|\n'

    return msg


def save_checkpoint(model, optimizer, epoch, critic_loss, actor_loss, policy_entropy, average_episode_reward, average_episode_length, path, args, first):

    try:
        if first:
            with open(os.path.join(os.path.dirname(path), 'config.json'), 'w', encoding='utf-8') as f:
                json.dump({k: str(v) for k, v in vars(args).items()}, f, ensure_ascii=False, indent=4)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'critic_loss': critic_loss,
            'actor_loss' : actor_loss,
            'policy_entropy' : policy_entropy,
            'average_episode_reward' : average_episode_reward,
            'average_episode_length': average_episode_length,
        }, path)
    except FileNotFoundError as fe:
        os.makedirs(os.path.dirname(path))
        if first:
            with open(os.path.join(os.path.dirname(path), 'config.json'), 'w', encoding='utf-8') as f:
                json.dump({k: str(v) for k,v in vars(args).items()}, f, ensure_ascii=False, indent=4)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'policy_entropy': policy_entropy,
            'average_episode_reward': average_episode_reward,
            'average_episode_length': average_episode_length,
        }, path)


def load_checkpoint(model, optimizer, args):
    checkpoint = os.path.expanduser(args.load)
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


if __name__ == '__main__':
    general_log_msg(bla='blob', bli='blib')
