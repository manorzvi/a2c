import gym
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import stable_baselines3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor

from args import common_args, make_args


def measure_state_difference(model, env, render=True, plot=True):

    from stable_baselines3.common.vec_env import VecEnv

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    l2_diffs = []
    pixel_diffs = []

    h, w, c = env.observation_space.shape

    obs = env.reset()
    prev_obs = obs.copy()
    if render:
        env.render()
    if plot:
        fig0, axes0 = plt.subplots(int(np.ceil(np.sqrt(c))), int(np.ceil(np.sqrt(c))))
        fig0.suptitle('Prev Obs')
        fig1, axes1 = plt.subplots(int(np.ceil(np.sqrt(c))), int(np.ceil(np.sqrt(c))))
        fig1.suptitle('Obs')
        for i, ax in enumerate(axes1.flat):
            ax.imshow(obs[0, :, :, i])

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if render:
            env.render()
        if plot:
            for i, ax in enumerate(axes0.flat):
                ax.imshow(prev_obs[0, :, :, i])
            for i, ax in enumerate(axes1.flat):
                ax.imshow(obs[0, :, :, i])

        l2_diff = obs - prev_obs
        pixel_diff = np.count_nonzero(l2_diff)
        pixel_diffs.append(pixel_diff)
        l2_diff = np.power(l2_diff, 2)
        l2_diff = np.sum(l2_diff)
        l2_diffs.append(l2_diff)

        prev_obs = obs.copy()

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(l2_diffs[c-1:-1])
    axes[0].set_title('L2 Diff')
    axes[1].plot(pixel_diffs[c-1:-1])
    axes[1].set_title('Pixel Diff')
    plt.show()


if __name__ == '__main__':
    args = common_args()
    args = make_args(args)

    pprint(args.__dict__)

    env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed)
    env = VecFrameStack(env, n_stack=args.n_stack)

    if args.load_path is not None:
        model = stable_baselines3.__dict__[args.alg.upper()].load(args.load_path)
    else:
        model = stable_baselines3.__dict__[args.alg.upper()](policy=args.policy,
                                                             env=env,
                                                             n_steps=args.n_steps,
                                                             verbose=args.verbose)

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

        # mean_reward, std_reward = evaluate_policy(model, env, render=True, n_eval_episodes=1)
        # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        measure_state_difference(model, env, render=False, plot=False)