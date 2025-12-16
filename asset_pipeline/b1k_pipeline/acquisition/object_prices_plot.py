import numpy as np
import matplotlib.pyplot as plt
import json

with open(r"C:\Users\Cem\Downloads\objprices.json", "r") as f:
    prices = json.load(f)

x = np.array(list(prices.values()))
q25, q75 = np.percentile(x, [25, 75])
bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
bins = round((x.max() - x.min()) / bin_width)
plt.hist(x, bins=bins)
plt.show()

print(
    "\n".join(str(x) for x in sorted(prices.items(), key=lambda x: x[1], reverse=True))
)
