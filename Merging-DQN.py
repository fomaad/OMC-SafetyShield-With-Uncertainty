import os

import gymnasium as gym
import maude
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane

import highway_env

model_dir = "RL-model/merging/dqn"
model_path = os.path.join(model_dir, "model")
log_path = os.path.join(model_dir, "log")

def train():
    # Create and configure the environment
    env = gym.make("merge-v0", render_mode='human')
    # env.unwrapped.configure(env_config)
    env.reset()

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.9,  # Discount factor
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log=log_path)

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_path,
        name_prefix="rl_model"
    )

    # Train the agent
    model.learn(int(10_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
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
    model = DQN.load(model_path)
    env = gym.make("merge-v0", render_mode="human",
                   config={"observation": {
                               "type": "Kinematics",
                                "features": ["presence", "x", "y", "vx", "vy", "heading"]
                   }})

    # env_config["observation"] = {
    #     "type": "Kinematics",
    #     "features": ["presence", "x", "y", "vx", "vy", "heading"],
    # }
    # env_config["simulation_frequency"] = 15
    # env.unwrapped.configure(env_config)
    env.reset()

    env.training = False  # Disable normalization updates
    env.norm_reward = False

    filename = utils.get_filename_arg(trajectory_file_name)
    dx_range = AbstractEnv.PERCEPTION_DISTANCE
    dy_range = AbstractLane.DEFAULT_WIDTH * 3
    ego_vehicle = env.unwrapped.vehicle
    speed_bound = ego_vehicle.MAX_SPEED - ego_vehicle.MIN_SPEED
    policy_frequency = env.unwrapped.config["policy_frequency"]

    test_runs = 10
    crashed_runs, total_reward, trajectories = utils.do_test(env,model, test_runs, True, SAFETY_SHIELD_ENABLE,
                                                             policy_frequency,dx_range,dy_range,speed_bound)

    print("\rCrashes:", len(crashed_runs), "/", test_runs, "runs",
          f"({len(crashed_runs) / test_runs * 100:0.1f} %)")

    if RECORD_TRAJECTORIES:
        utils.write_trajectories(trajectories, crashed_runs, total_reward, filename)

    env.close()

if __name__ == "__main__":
    to_train = False
    if to_train:
        train()

    test()