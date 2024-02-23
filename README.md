# Machine Learing Reinforcement Learning: Frozen Lake Project

Ameer & Namil Collabortation

- Proposed Domain: An agent traverses through the 4x4 lake layout without falling into the holes. Winning can be attained by reaching the goal state. There is no guarantee of the agent moving in the intended direction via the slippery ice panels. 
- Hypothesis: RL can discover a method to successfully traverse any randomly generated solvable Frozen Lake maze. 
- Significance: Solving lower scale projects prepares us to solve real-world, applicable solutions. 
- Existing software: Frozen Lake simulator via gymnasium: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- Proposed Contributions: Ameer and Namil will each implement the different RL algorithms i.e. Q-learning and Dynamic Programming.

## Description
- The game starts with the player at location [0,0] of the frozen lake grid world with the goal located at far extent of the world e.g. [3,3] for the 4x4 environment (for an 8x8 it would be [63,63]).
- Holes in the ice are distributed in set locations when using a pre-determined map or in random locations when a random map is generated (maps can be found in gymnaisum).
- The player makes moves until they reach the goal or fall in a hole.
- The lake is slippery (unless disabled) so the player may move perpendicular to the intended direction sometimes (see is_slippery, can turn off and on).
- Randomly generated worlds will always have a path to the goal.

## Action Space
- The action shape is (1,) in the range {0, 3}, indicating which direction to the player.
- In other words, the player has 4 possible actions (left, down, right, up).
- {0 = left, 1 = down, 2 = right, 3 = up}.

## Observation Space
- The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).=
- For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15. The number of possible observations is dependent on the size of the map.
- For another example, the goal position in the 8x8 map can be calculated as follows: 7 * 8 + 7 = 63. The number of possible observaations is depdent on the size of the map.
- The observation is returned as an int().

## Starting State
- The episode starts with the player in state [0] (location [0, 0]).
- Thus, the terminal state is 0.

## Rewards
- Reach goal: +1
- Reach hole: 0
- Reach frozen: 0

Idea for novelty: negative rewards for the player for each step taken; for example, each frozen will be a negative reward, ultiamley guiding the agent to the correct path. Check more examples in the slides / textbook for maze related games.

## Episode End
The episode ends if the following happens:

### Termination:
- The player moves into a hole.
- The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).

### Truncation (when using the time_limit wrapper):
- The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

## Information
step() and reset() return a dict with the following keys:
p - transition probability for the state.

## Arguments
```
import gymnasium as gym
gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
```

desc=None: Used to specify maps non-preloaded maps.

Specify a custom map.
```desc=["SFFF", "FHFH", "FFFH", "HFFG"].```

A random generated map can be specified by calling the function generate_random_map.

```
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
```

map_name="4x4": ID to use any of the preloaded maps.

    "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]

    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]


If desc=None then map_name will be used. If both desc and map_name are None a random 8x8 map with 80% of locations frozen will be generated.

is_slippery=True: If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions.

For example, if action is left and is_slippery is True, then:
- P(move left)=1/3
- P(move up)=1/3
- P(move down)=1/3

See is_slippery for transition probability information.
