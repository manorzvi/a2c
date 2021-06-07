import os
from loguru import logger
import gym
import torch

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from arg_parser import common_arg_parser, make_auto_args, check_args
from play import random_walk_play, trained_agent_play
from model import A2CModel
from wrappers import WarpFrame, FrameStack, EpisodicLifeEnv
from train import train
from utils import load_checkpoint, general_log_msg


def main():
    args = common_arg_parser()
    args = make_auto_args(args)
    args = check_args(args)
    logger.info('\n' + str(args)[len('Namespace')+1:-1].replace(', ', '\n'))

    if args.n_env > 1:
        env = [lambda: FrameStack(WarpFrame(EpisodicLifeEnv(gym.make(args.env)),
                                            width=args.frame_width,
                                            height=args.frame_height,
                                            grayscale=(args.frame_channel == 1)),
                                  args.frame_stack)
               for i in range(args.n_env)]
        # TODO: Add VecActionSpaceWrapper: Vectorized Action Space for clean and easy action sampling (manorz, 05/30/21)
        # TODO (2): Add support In-Process Vectorized envs (similar to: 'Accelerated Methods for Deep Reinforcement Learning').
        #  Improve sample efficiency by coalescing groups of envs into the same process.
        env = SubprocVecEnv(env)
    else:
        env = gym.make(args.env)
        env = EpisodicLifeEnv(env)
        env = WarpFrame(env, width=args.frame_width, height=args.frame_height, grayscale=(args.frame_channel == 1))
        env = FrameStack(env, args.frame_stack)

    if args.play:
        random_walk_play(env, 1, args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = A2CModel(shape_in=env.observation_space.shape, n_action=env.action_space.n, args=args).to(args.device)
    logger.info('\n|---------|'
                '\n|The Model|'
                '\n|---------|\n'
                '{}'.format(str(model)))
    try:
        model.check(env, args)
        logger.info(f"FeedForward check passed")
    except Exception as e:
        logger.error("FeedForward check failed. Error message: {}".format(str(e)))
        exit("FeedForward check failed. Error message: {}".format(str(e)))

    try:
        optimizer_func = torch.optim.__dict__[args.optimizer]
    except KeyError as e:
        logger.error(f'Optimizer {args.optimizer} does not found in torch.optim.')
        exit()

    if args.optimizer == 'RMSprop':
        optimizer = optimizer_func(model.parameters(), lr=args.lr, eps=args.rmsprop_eps, alpha=args.rmsprop_alpha)

    logger.info("\n|-------------|"
                "\n|The Optimizer|"
                "\n|-------------|\n"
                "{}".format(optimizer))

    try:
        value_loss_func = torch.nn.__dict__[args.value_loss]()
    except KeyError as e:
        logger.error(f'Value Loss {args.value_loss} does not found in torch.nn.')
        exit()

    logger.info("\n|--------------|"
                "\n|The Value Loss|"
                "\n|--------------|\n"
                "{}".format(value_loss_func))

    if args.load is not None:
        logger.info('Loading Checkpoint ... ')
        model, optimizer = load_checkpoint(model, optimizer, args)
        logger.info('Done.')

    if args.train:
        try:
            train(model, env, value_loss_func, optimizer, args)
        except ValueError as ve:
            logger.error(ve)
            exit()

    if args.play:
        logger.info(general_log_msg(total_reward=trained_agent_play(model, args)))


if __name__ == '__main__':
    main()
