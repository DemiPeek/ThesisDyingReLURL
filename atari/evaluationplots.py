import pandas as pd
import matplotlib.pyplot as plt

evaluations = pd.read_csv("episodic_returns_seed2.csv")
means = evaluations.mean()
errors = evaluations.std()
seedax = ['50M', '60M', '70M', '80M', '90M', '100M', '110M', '120M', '130M', '140M', '150M', '160M', '170M', '180M', '190M', '200M']
seed2ax = ['0', '10M', '20M', '30M', '40M', '50M', '60M', '70M', '80M', '90M']

fig, ax = plt.subplots()
means.plot.bar(yerr=errors, ax=ax, capsize=4, rot=0)

ax.set_xticks(range(len(seed2ax)))
ax.set_xticklabels(seed2ax, rotation=45, ha="right")

plt.savefig("noise*0.25_seed2.png")
plt.show()

