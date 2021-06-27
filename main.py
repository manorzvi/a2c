import gym
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import stable_baselines3
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
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

    if plot:
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(l2_diffs[c-1:-1])
        axes[0].set_title('L2 Diff')
        axes[1].plot(pixel_diffs[c-1:-1])
        axes[1].set_title('Pixel Diff')
        # plt.show()


if __name__ == '__main__':
    args = common_args()
    args = make_args(args)

    pprint(args.__dict__)

    train_env = make_atari_env(args.env, n_envs=args.n_envs, seed=args.seed)
    train_env = VecFrameStack(train_env, n_stack=args.n_stack)

    eval_env = make_atari_env(args.env, n_envs=1, seed=args.seed)
    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)

    model = eval(args.alg.upper())(**args.model_kwargs)
    if args.load_path is not None:
        model = model.load(args.load_path)

    if args.train:
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

        checkpoint_callback = CheckpointCallback(
            save_freq=args.ckpt_interval,
            save_path=args.log_path,
            verbose=args.verbose
        )
        # TODO: Need to write my own EvalCallback due to this bug:
        #  ValueError: Error: Unexpected observation shape (1, 84, 84, 4) for Box environment, please use (3, 210, 160) or (n_env, 3, 210, 160) for the observation shape.
        # eval_callback = EvalCallback(
        #     eval_env=eval_env,
        #     best_model_save_path=args.log_path,
        #     log_path=args.log_path,
        #     eval_freq=args.eval_interval,
        #     verbose=args.verbose
        # )
        # callback = CallbackList([eval_callback, checkpoint_callback])
        callback = CallbackList([checkpoint_callback])

        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(args.log_path)

    if args.play:
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(model, eval_env, render=True, n_eval_episodes=3)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # measure_state_difference(model, env, render=False, plot=True)
