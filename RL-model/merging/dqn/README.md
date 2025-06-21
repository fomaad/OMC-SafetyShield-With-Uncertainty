1. `model`: default config
```
"collision_reward": -1,
"right_lane_reward": 0.1,
"high_speed_reward": 0.2,
"reward_speed_range": [20, 30],
"merging_speed_reward": -0.5,
"lane_change_reward": -0.05,
```

2. `model-higher-penalty-lc`: higher penalty for lane change
```
"lane_change_reward": -0.2,
```

3. `model-higher-rightlane-rw`: higher reward for keeping right lane
```
"lane_change_reward": -0.2,
"right_lane_reward": 0.2,
```

4. `model-higher-rightlane-rw2`:
```
"lane_change_reward": -0.2,
"right_lane_reward": 0.11,
```