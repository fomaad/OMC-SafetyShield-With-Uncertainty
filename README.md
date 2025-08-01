## Robust Decision-Making Safety for Autonomous Driving under Observation Uncertainty via Online Model Checking

A safety shield for AI-driven Autonomous Driving (AD) with Observation (Perception) Uncertainty is developed based on Online Model checking.

In this repository, you can find:

- [Supplementary Material document](supplementary-material.pdf) to our paper.
- Eight Reinforcement Learning (RL) agents: `DQN-Single`, `DQN-Single-Adversary`, `DQN`, `DQN-Adversary`, `PPO-Single`, `PPO-Single-Adversary`, `PPO`, `PPO-Adversary` and the trained models in folders `highway_{dqn/ppo}` and `single-lane/highway_{dqn/ppo}`.

- A formal model (`vehicle.maude` and `fmodel.maude`), specifying the environment's state  (e.g., the status of the NPC and ego vehicles).

- A safety shield implementation (`SafetyShield.py`)

- Trajectories in YAML format.


### Execute
Use the following command to execute DQN-based agent in the three-lane highway environment:

```
python DQN.py trajectories.yaml
```

Once finishing, the trajectories will be written to file `trajectories.yaml`.
By default, the shield is enabled. You can disable it by setting flag `SAFETY_SHIELD_ENABLE` to false.
