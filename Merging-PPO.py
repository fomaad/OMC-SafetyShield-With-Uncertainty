import os

import gymnasium as gym
import maude
from stable_baselines3 import PPO

import highway_env
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane
import utils

model_dir = "RL-model/merging/ppo"
model_path = os.path.join(model_dir, "model-lighter-pen-lc")
log_path = os.path.join(model_dir, "log")

env_config = {
    "lane_change_reward": -0.1,
    "right_lane_reward": 0.11,
}
def train():
    n_cpu = 8
    batch_size = 128

    # Create and configure the environment
    env = gym.make("merge-v0", render_mode='human', config=env_config)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.95,
        verbose=2,
        tensorboard_log=log_path,
    )

    # Train the agent
    model.learn(total_timesteps=int(400_000), progress_bar=True)
    model.save(model_path)

RECORD_TRAJECTORIES = False
SAFETY_SHIELD_ENABLE = False
trajectory_file_name = "trajectories.yaml"

if SAFETY_SHIELD_ENABLE:
    maude.init()
    maude.load("vehicle.maude")
    maude.load("fmodel.maude")

def test():
    # Load the trained model
    model = PPO.load(model_path)
    env = gym.make("merge-v0", render_mode="human",
                   config={
                       "observation": {
                           "type": "Kinematics",
                           "features": ["presence", "x", "y", "vx", "vy", "heading"]
                       }
                   })

    env.training = False  # Disable normalization updates
    env.norm_reward = False

    filename = utils.get_filename_arg(trajectory_file_name)
    dx_range = AbstractEnv.PERCEPTION_DISTANCE
    dy_range = AbstractLane.DEFAULT_WIDTH * 3
    ego_vehicle = env.unwrapped.vehicle
    speed_bound = ego_vehicle.MAX_SPEED - ego_vehicle.MIN_SPEED
    policy_frequency = env.unwrapped.config["policy_frequency"]

    test_runs = 200
    crashed_runs, total_reward, trajectories = utils.do_test(env, model, test_runs, True, SAFETY_SHIELD_ENABLE,
                                                             policy_frequency, dx_range, dy_range,speed_bound)

    print("\rCrashes:", len(crashed_runs), "/", test_runs, "runs",
          f"({len(crashed_runs) / test_runs * 100:0.1f} %)")

    if RECORD_TRAJECTORIES:
        utils.write_trajectories(trajectories, crashed_runs, total_reward, filename)

    env.close()


if __name__ == "__main__":
    to_train = True
    if to_train:
        train()

    test()