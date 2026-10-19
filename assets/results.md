| method | env | steps to solve | final return (20 ep) | best eval | steps run | wall time |
|---|---|---|---|---|---|---|
| `dqn` | discrete | **50,000** | **209.3 +- 69.5** | 217.8 | 50,000 | 2.3 min |
| `double-dqn` | discrete | **90,000** | **227.6 +- 29.3** | 200.9 | 90,000 | 4.4 min |
| `dueling-dqn` | discrete | **50,000** | 167.1 +- 99.7 | 212.0 | 50,000 | 1.5 min |
| `reinforce` | discrete | not reached | 45.7 +- 102.3 | 111.2 | 1,000,063 | 8.8 min |
| `a2c` | discrete | **420,000** | -144.6 +- 32.4 | 228.9 | 1,000,064 | 4.9 min |
| `ppo` | discrete | **330,000** | **276.4 +- 18.2** | 269.6 | 1,015,808 | 6.2 min |
| `sac-discrete` | discrete | not reached | -53.8 +- 39.1 | -11.7 | 100,000 | 6.1 min |
| `ddpg` | continuous | not reached | -60.5 +- 121.3 | -6.0 | 120,000 | 5.1 min |
| `td3` | continuous | not reached | 3.0 +- 38.4 | -4.0 | 120,000 | 4.6 min |
| `sac` | continuous | not reached | 8.9 +- 21.3 | 13.1 | 120,000 | 9.8 min |
| `ppo-continuous` | continuous | **300,000** | **262.5 +- 35.9** | 241.0 | 311,296 | 1.5 min |
