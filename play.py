import time
import gym
import torch
from torch.distributions import Categorical
from loguru import logger
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from wrappers import WarpFrame, FrameStack
from utils import np2torch_obs


def random_walk_play(env, epochs, args):
    logger.info(f"Hi! Welcome to Random Walk Play =D The Place where too much Alcohol and not enough sleep hours meet")

    if args.timer:
        from timeit import default_timer as timer
        avg_time = 0
        counter = 0
        logger.info("env.step execution time will be measured.")

    for epoch in range(epochs):
        done = False

        _ = env.reset()
        if args.render:
            env.render()
            time.sleep(0.04)
        epoch_counter = 0

        while not done:
            actions = [env.action_space.sample() for _ in range(env.num_envs)] if isinstance(env, SubprocVecEnv) else env.action_space.sample()
            if args.timer:
                start = timer()
                _, rewards, dones, info = env.step(actions)
                end = timer()
                avg_time = avg_time * (counter / (counter + 1)) + (end - start) / (counter + 1)
                counter += 1
            else:
                _, rewards, dones, info = env.step(actions)
            if args.render:
                env.render()
                time.sleep(0.004)
            epoch_counter += 1

            done = dones if not isinstance(env, SubprocVecEnv) else any(dones)
            if done:
                if args.timer:
                    logger.info("Epoch {}, Agents {} have finished after {}. "
                                "Now everyone most die. "
                                "Average Env step time: {}[s]. "
                                "Bye Bye".format(epoch, str([i for i, x in
                                                             enumerate([dones] if not isinstance(env, SubprocVecEnv) else dones) if
                                                             x])[1:-1], epoch_counter, avg_time))
                else:
                    logger.info("Epoch {}, Agents {} have finished after {}. "
                                "Now everyone most die. "
                                "Bye Bye".format(epoch, str([i for i, x in
                                                             enumerate([dones] if not isinstance(env, SubprocVecEnv) else dones) if
                                                             x])[1:-1], epoch_counter))


def trained_agent_play(model, args):

    model.eval()

    if args.n_env > 1:
        env = [lambda: FrameStack(WarpFrame(gym.make(args.env), width=args.frame_width,
                                                                height=args.frame_height,
                                                                grayscale=(args.frame_channel == 1)), args.frame_stack)
               for i in range(args.n_env)]
        env = SubprocVecEnv(env)
    else:
        env = gym.make(args.env)
        env = WarpFrame(env, width=args.frame_width, height=args.frame_height, grayscale=(args.frame_channel == 1))
        env = FrameStack(env, args.frame_stack)

    done = False
    obs = env.reset()
    if args.render:
        env.render()
        time.sleep(0.004)
    obs = np2torch_obs(obs, env.observation_space.low, env.observation_space.high).to(args.device)

    total_rewards = [0 for _ in range(args.n_env)]

    while not done:
        with torch.no_grad():
            policies = model.forward_actor(obs)
        m = Categorical(logits=policies)
        actions = m.sample()

        obs, r, dones, info = env.step(actions.tolist())
        if args.render:
            env.render()
            time.sleep(0.004)
        obs = np2torch_obs(obs, env.observation_space.low, env.observation_space.high).to(args.device)

        total_rewards = [x+y for x, y in zip(total_rewards, r if isinstance(r, list) else [r])]

        done = dones if not isinstance(env, SubprocVecEnv) else any(dones)

    env.close()

    model.train()

    return total_rewards
