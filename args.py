import os
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from pprint import pprint


def common_args() -> Namespace:
    """
    Parse command-line arguments
    :return: argparse.Namespace
    """
    parser = ArgumentParser()

    parser.add_argument('--alg', '-a',          type=str, required=True,
                        choices=['a2c', 'ddpg', 'dqn', 'her', 'ppo', 'sac', 'td3'])
    parser.add_argument('--policy', '-p',       type=str, required=True,
                        choices=['MlpPolicy', 'CnnPolicy'])

    parser.add_argument('--env', '-e',          type=str, required=True)
    parser.add_argument('--n_envs', '-n_e',     type=int, default=mp.cpu_count(),
                        help='Number of environments in parallel')
    parser.add_argument('--n_stack',            type=int, default=4,
                        help='Number of frames to stack')
    parser.add_argument('--n_steps',            type=int, default=5)

    parser.add_argument('--timesteps', '-t',        type=int, required=True)
    parser.add_argument('--seed', '-s',             type=int, default=42)
    parser.add_argument('--eval_interval', '-e_i',  type=int, default=1000)
    parser.add_argument('--ckpt_interval', '-c_i',  type=int, default=1000)
    parser.add_argument('--log_path',               type=str)

    parser.add_argument('--verbose', '-v',      type=int, default=1,
                        choices=[0, 1, 2], help='Verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--name', '-n',         type=str)
    parser.add_argument('--load_path',          type=str)
    parser.add_argument('--train',              default=False, action='store_true')
    parser.add_argument('--play',               default=False, action='store_true')

    args = parser.parse_args()

    return args


def make_args(args: Namespace) -> Namespace:
    if args.name is None:
        name = '{}'.format(args.alg) + \
               '|{}'.format(args.policy) + \
               '|{}'.format(args.env) + \
               '|n_envs={}'.format(args.n_envs) + \
               '|n_steps={}'.format(args.n_steps) + \
               '|n_stack={}'.format(args.n_stack)
        args.name = name
    if args.log_path is None:
        log_path = os.path.join('logs', args.name)
        args.log_path = log_path
    if args.alg == 'a2c':
        args.model_kwargs = {
            'policy':   args.policy,
            'env':      args.env,
            'n_steps':  args.n_steps,
            'verbose':  args.verbose
        }

    return args


def check_args(args: Namespace) -> Namespace:
    os.makedirs(args.log_dir, exist_ok=True)
    return args


if __name__ == '__main__':
    args = common_args()
    args = make_args(args)
    args = check_args(args)
    pprint(args)
