import gym
from pprint import pprint
import stable_baselines3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor

from args import common_args, make_args

args = common_args()
args = make_args(args)

pprint(args.__dict__)

env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed)
env = VecFrameStack(env, n_stack=args.n_stack)

if args.load_path is not None:
    model = stable_baselines3.__dict__[args.alg.upper()].load(args.load_path)
else:
    model = stable_baselines3.__dict__[args.alg.upper()](args.policy, env, verbose=args.verbose)

if args.train:
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.ckpt_interval,
        save_path=args.log_path,
        verbose=args.verbose
    )
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback])
    model.save(args.log_path)

if args.play:
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, env, render=True)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
