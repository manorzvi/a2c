import os
from timeit import default_timer as timer
import torch
from utils import np2torch_obs
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from loguru import logger

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from utils import save_checkpoint, general_log_msg
from play import trained_agent_play


def train(model, env, value_loss_func, optimizer, args):
    model.train()

    if env.observation_space.__class__.__name__ == 'Box':
        obs_shape = env.observation_space.shape
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    else:
        raise ValueError(f"env.observation_space.__class__.__name__ == {env.observation_space.__class__.__name__}.\n"
                         f"Only 'Box' currently supported.")

    trace_obs = torch.zeros(args.n_steps + 1, args.n_env, *obs_shape).to(args.device)  # TODO: check the impact on performance of moving the tracers to 'registered buffers' of the model (saved on gpu memory)
    trace_reward = torch.zeros(args.n_steps, args.n_env).to(args.device)
    trace_return = torch.zeros(args.n_steps + 1, args.n_env).to(args.device)
    trace_action = torch.zeros(args.n_steps, args.n_env).to(args.device)
    trace_mask = torch.zeros(args.n_steps, args.n_env).to(args.device)

    obs = env.reset()
    # env.render()
    obs = np2torch_obs(obs, env.observation_space.low, env.observation_space.high)
    obs.to(args.device)

    trace_obs[0] = obs

    average_episode_reward = []
    average_episode_length = []

    episode_reward = [0 for _ in range(args.n_env)]
    episode_length = [0 for _ in range(args.n_env)]

    if args.timer:
        update_timers = []
        step_timers = []

    for update in tqdm(range(args.n_updates)):

        if args.timer:
            update_loop_start = timer()

        for step in range(args.n_steps):

            episode_length = [x+1 for x in episode_length]

            with torch.no_grad():
                policies = model.forward_actor(trace_obs[step])

            m = Categorical(logits=policies)
            actions = m.sample()  # TODO: maybe need here to move actions (before env.step) back to cpu (need to check on gpu)

            obs, rewards, dones, infos = env.step(actions.tolist())
            # env.render()
            obs = np2torch_obs(obs, env.observation_space.low, env.observation_space.high)
            obs.to(args.device)

            episode_reward = [x+r for x, r in zip(episode_reward, rewards)]

            trace_obs[step + 1] = obs
            trace_action[step] = actions.to(args.device)
            trace_reward[step] = torch.from_numpy(rewards).to(args.device)
            trace_mask[step] = torch.from_numpy(1 - dones.astype(np.float32)).to(args.device)

            done = dones if not isinstance(env, SubprocVecEnv) else any(dones)
            if done:
                average_episode_length.extend([l for l, d in zip(episode_length, dones) if d])
                average_episode_reward.extend([r for r, d in zip(episode_reward, dones) if d])
                episode_length = [l * (1 - d.astype(int)) for l, d in zip(episode_length, dones)]
                episode_reward = [r * (1 - d.astype(int)) for r, d in zip(episode_reward, dones)]

        if args.timer:
            step_loop_end = timer()
            step_timers.append(step_loop_end-update_loop_start)

        del step

        with torch.no_grad():
            next_value = model.forward_critic(trace_obs[-1])

        trace_return[-1] = next_value.squeeze(1)
        for step in reversed(range(args.n_steps)):
            trace_return[step] = trace_reward[step] + args.gamma * trace_return[step + 1] * trace_mask[step]

        policies, values = model(trace_obs[:-1].view(-1, *obs_shape))
        m = Categorical(logits=policies)
        log_probs = m.log_prob(trace_action.view(-1))
        entropy_probs = m.entropy().mean()  # TODO: Consider to subtract the action entropy from the action loss, as in: 'Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration' (manorz, 06/03/21)

        values = values.view(args.n_steps, args.n_env)
        log_probs = log_probs.view(args.n_steps, args.n_env)
        advantage = trace_return[:-1] - values

        value_loss = value_loss_func(trace_return[:-1], values)
        action_loss = -(advantage.detach() * log_probs).mean()

        optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        action_loss.backward()
        optimizer.step()

        trace_obs[0]    = trace_obs[-1]
        trace_mask[0]   = trace_mask[-1]

        if args.timer:
            update_loop_end = timer()
            update_timers.append(update_loop_end-update_loop_start)

        if args.log:
            if (update * args.n_steps) % args.log_interval == 0:
                if args.timer:
                    log_msg = general_log_msg(update=update,
                                              policy_loss=action_loss.item(),
                                              critic_loss=value_loss.item(), policy_entropy=entropy_probs.item(),
                                              average_episode_reward=(sum(average_episode_reward)/len(average_episode_reward) if
                                                                      len(average_episode_reward) > 0 else None),
                                              average_episode_length=(sum(average_episode_length)/len(average_episode_length) if
                                                                      len(average_episode_length) > 0 else None),
                                              s_per_update=(sum(update_timers) / len(update_timers)),
                                              update_iter_per_s=(1/(sum(update_timers)/len(update_timers))),
                                              s_per_step=((sum(step_timers) / len(step_timers))/args.n_steps),
                                              step_iter_per_s=(args.n_steps/(sum(step_timers)/len(step_timers)))
                                              )
                else:
                    log_msg = general_log_msg(update=update,
                                              policy_loss=action_loss.item(),
                                              critic_loss=value_loss.item(), policy_entropy=entropy_probs.item(),
                                              average_episode_reward=(sum(average_episode_reward)/len(average_episode_reward) if
                                                                      len(average_episode_reward) > 0 else None),
                                              average_episode_length=(sum(average_episode_length)/len(average_episode_length) if
                                                                      len(average_episode_length) > 0 else None)
                                              )
                logger.info(log_msg)

        if args.save:
            if (update * args.n_steps) % args.save_interval == 0:
                checkpoint_name = '{}.pth'.format(update * args.n_steps)
                checkpoint_name = os.path.expanduser(os.path.join(args.save_path, checkpoint_name))
                logger.info('Saving checkpoint {} at {} ...'.format(checkpoint_name, update * args.n_steps))
                save_checkpoint(model, optimizer, update, value_loss.item(), action_loss.item(),
                                entropy_probs.item(), average_episode_reward, average_episode_length, checkpoint_name,
                                args, first=(update == 0))
                logger.info('Done.')

        if args.evaluate:
            if (update * args.n_steps) % args.evaluate_interval == 0:
                logger.info(general_log_msg(total_reward=trained_agent_play(model, args)))

    checkpoint_name = '{}.pth'.format(update * args.n_steps)
    checkpoint_name = os.path.expanduser(os.path.join(args.save_path, checkpoint_name))
    logger.info('Saving final checkpoint {} at {} ...'.format(checkpoint_name, update * args.n_steps))
    save_checkpoint(model, optimizer, update, value_loss.item(), action_loss.item(),
                    entropy_probs.item(), average_episode_reward, average_episode_length, checkpoint_name,
                    args, first=(update == 0))
    logger.info('Done.')

    model.eval()
