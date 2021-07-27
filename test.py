from matplotlib import pyplot as plt
import numpy as np
import pickle

record = pickle.load(open('performance/record_vs_random/record_11000.pkl', 'rb'))

won = [record[rec]['won']/10 for rec in record.keys()]
tie = [record[rec]['tie']/10 for rec in record.keys()]
won_tie = list(np.array(won)+np.array(tie))

x_idx = [i*200/1000 for i in range(len(won))]
plt.plot(x_idx, won, label = "WRRG")
plt.plot(x_idx, won_tie, label = "WDRRG")

plt.xlabel("Number of trained episodes (thousands)")
# plt.ylabel("Percentage (%)")

plt.xticks(np.arange(0,12,1))
plt.yticks(np.arange(50,101,5))
plt.grid(axis='y')
plt.grid(axis='x')

# plt.title("WRRG and WDRRG values over 11000 training games")
plt.legend()

plt.savefig('docs/images/wrrd_0.png')
plt.close()
