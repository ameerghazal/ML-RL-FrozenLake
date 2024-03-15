
# Import the algorithms from separate files
from Double_Q_Learning.doubleQLearning import doubleQ
from QLearning.QLearning_Soft import normalQ
from SARSA.SarsaDecay import SARSA
import matplotlib.pyplot as plt
import numpy as np

# Call each algorithm for multiple runs, average them, and plot the outcomes.
sarsaAvgTotalReward = []
normalQavgTotalReward = []
doubleQavgTotalReward = []

for i in range(50):
  sarsaAvgTotalReward.append(SARSA(10000, decayParam=(1, 0.0005, "linear")))
  normalQavgTotalReward.append(normalQ(10000, decayParam=(1, 0.0005, "linear"), policy="greedy-policy"))
  doubleQavgTotalReward.append(doubleQ(10000, decayParam=(1, 0.0005, "linear")))

# Average the episodes and plot the results.
plt.plot(np.mean(sarsaAvgTotalReward, axis=0), label='SARSA')
plt.plot(np.mean(normalQavgTotalReward, axis=0), label='Q-Learning')
plt.plot(np.mean(doubleQavgTotalReward,axis=0), label='Double Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Average Total Reward')
plt.title('Total reward over 15000 episodes')
plt.legend()
plt.savefig("RL Methods/Graphing/Comparsions_Multiple_Runs.png")
plt.show()

