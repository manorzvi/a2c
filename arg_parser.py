import os
from pathlib import Path
import shutil
from argparse import ArgumentParser, Namespace
import multiprocessing as mp
from loguru import logger
import torch
from utils import list_gym_envs, defaultdict_gym_envs


def get_env_type(env):
    env_type = None
    for g, e in defaultdict_gym_envs().items():
        if env in e:
            env_type = g
            break
    return env_type


def check_args(args: Namespace) -> Namespace:
    if args.log is False and args.log_path is not None:
        logger.error(f"args.log={args.log} and args.log_path={args.log_path}")
        exit(1)
    if args.save is False and args.save_path is not None:
        logger.error(f"args.save={args.save} and args.save_path={args.save_path}")
        exit(1)

    if args.log:
        if os.path.isdir(args.log_path) and not args.override_log:
            logger.error(f"Logging directory exist ({args.log_path}). Please remove.")
            exit(1)
        elif os.path.isdir(args.log_path) and args.override_log:
            logger.warning(f"Logging directory exist ({args.log_path}). Removing...")
            shutil.rmtree(args.log_path)
        logger.add(f"{os.path.join(args.log_path, 'log.log')}")

    if args.save:
        if os.path.isdir(args.save_path) and not args.override_save:
            logger.error(f"Saving directory exist ({args.save_path}). Please remove.")
            exit(1)
        elif os.path.isdir(args.save_path) and args.override_save:
            logger.warning(f"Saving directory exist ({args.save_path}). Removing...")
            shutil.rmtree(args.save_path)

    return args


def _make_name(args: Namespace) -> str:
    return f'{args.env}|alg={args.alg}|n_env={args.n_env}|n_timesteps={args.n_timesteps}|n_steps={args.n_steps}|frame_stack={args.frame_stack}'


def make_auto_args(args: Namespace) -> Namespace:
    if args.name is None:
        args.name = _make_name(args)
    if args.save and args.save_path is None:
        args.save_path = os.path.expanduser(os.path.join(os.getcwd(), 'models', args.name))
    if args.log and args.log_path is None:
        args.log_path = os.path.expanduser(os.path.join(os.getcwd(), 'logs', args.name))

    args.env_type = get_env_type(args.env)

    if args.cuda and not torch.cuda.is_available():
        logger.warning(f'args.cuda={args.cuda}, but torch.cuda.is_available()={torch.cuda.is_available()}. Set args.cuda=False')
        args.cuda = False

    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    args.n_updates = args.n_timesteps // args.n_steps

    return args


def general_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    general = parser.add_argument_group("general args")
    general.add_argument('--seed', '-s', default=42, type=int)

    general.add_argument('--save', default=False, action='store_true')
    general.add_argument('--save_path', type=str)
    general.add_argument('--override_save', default=False, action='store_true')
    general.add_argument('--save_interval', type=int, default=10000)

    general.add_argument('--load', type=str)

    general.add_argument('--log', default=False, action='store_true')
    general.add_argument('--log_path', type=str)
    general.add_argument('--override_log', default=False, action='store_true')
    general.add_argument('--log_interval', type=int, default=10000)

    general.add_argument('--play', '-p', default=False, action='store_true')
    general.add_argument('--interactive_play', '-interactive_p', default=False, action='store_true')

    general.add_argument('--name', type=str)
    general.add_argument('--cuda', default=False, action='store_true')
    general.add_argument('--timer', default=False, action='store_true')

    return parser


def env_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    envs = list_gym_envs()
    env = parser.add_argument_group("env args")
    env.add_argument('--env', '-e',    type=str, default='BreakoutNoFrameskip-v4', choices=envs)
    env.add_argument('--n_env',        type=int, default=mp.cpu_count())

    env.add_argument('--frame_width',   default=84, type=int)
    env.add_argument('--frame_height',  default=84, type=int)
    env.add_argument('--frame_channel', default=1,  type=int)
    env.add_argument('--frame_stack',   default=4,  type=int)

    return parser


def alg_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    alg = parser.add_argument_group("alg args")
    alg.add_argument('--alg', '-a',     type=str,   default='a2c')
    alg.add_argument('--gamma',         type=float, default=0.99)
    alg.add_argument('--value_loss',    type=str,   default='MSELoss')
    return parser


def model_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    model = parser.add_argument_group("model args")
    return parser


def train_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    train = parser.add_argument_group("train args")
    train.add_argument('--n_timesteps', type=int, default=1000000)
    train.add_argument('--n_steps',     type=int, default=5)
    train.add_argument('--train', '-t',           default=False, action='store_true')

    train.add_argument('--optimizer',     type=str,   default='RMSprop')
    train.add_argument('--lr',            type=float, default=7e-4)
    train.add_argument('--rmsprop_eps',   type=float, default=1e-5)
    train.add_argument('--rmsprop_alpha', type=float, default=0.99)  # Taken from the A3C paper: 'Asynchronous Methods for Deep Reinforcment Learning'

    train.add_argument('--evaluate', default=False, action='store_true')
    train.add_argument('--evaluate_interval', type=int, default=10000)

    return parser


def common_arg_parser() -> Namespace:
    """
    Parse command-line arguments
    :return: argparse.Namespace
    """
    parser = ArgumentParser()

    parser = general_arg_parser(parser)
    parser = env_arg_parser(parser)
    parser = alg_arg_parser(parser)
    parser = model_arg_parser(parser)
    parser = train_arg_parser(parser)

    args = parser.parse_args()

    return args