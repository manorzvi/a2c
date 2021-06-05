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


def training_log_msg(update, actor_loss, critic_loss, policy_entroy, episode_reward, episode_length, args):
    update_msg = "update#{}".format(update+1,)
    timestep_msg = "timestep#{}".format((update+1)*args.n_steps)
    actor_loss_msg = "actor_loss={0:.6f}".format(actor_loss)
    critic_loss_msg = "critic_loss={0:.6f}".format(critic_loss)
    policy_entroy_msg = "policy_entropy={0:.6f}".format(policy_entroy)
    episode_reward_msg = "average_episode_reward={0:.2f}".format(episode_reward) if episode_reward is not None else ''
    episode_length_msg = "average_episode_length={0:.2f}".format(episode_length) if episode_length is not None else ''

    frame_width = max([len(update_msg),
                       len(timestep_msg),
                       len(actor_loss_msg),
                       len(critic_loss_msg),
                       len(policy_entroy_msg),
                       len(episode_length_msg),
                       len(episode_reward_msg)])

    msg = '\n|' + '-' * frame_width + '|\n' +\
          '|' + update_msg + ' ' * (frame_width-len(update_msg)) + '|\n' + \
          '|' + timestep_msg + ' ' * (frame_width-len(timestep_msg)) + '|\n' + \
          '|' + actor_loss_msg + ' ' * (frame_width-len(actor_loss_msg)) + '|\n' + \
          '|' + critic_loss_msg + ' ' * (frame_width-len(critic_loss_msg)) + '|\n' + \
          '|' + policy_entroy_msg + ' ' * (frame_width-len(policy_entroy_msg)) + '|\n' + \
          '|' + episode_reward_msg + ' ' * (frame_width - len(episode_reward_msg)) + '|\n' + \
          '|' + episode_length_msg + ' ' * (frame_width - len(episode_length_msg)) + '|\n' + \
          '|' + '-' * frame_width + '|'

    return msg


def playing_log_msg(total_rewards):
    msg_ = "Total Reward per Agent: {}".format(str([total_rewards])[2:-2])
    msg = '\n|' + '-' * len(msg_) + '|\n' + \
          '|' + msg_ + '|\n' + \
          '|' + '-' * len(msg_) + '|'

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
