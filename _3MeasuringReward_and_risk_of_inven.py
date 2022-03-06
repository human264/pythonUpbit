import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")

hm = pd.read_csv("hm.csv", index_col="Date", parse_dates = ["Date"])
print(hm)
hm.Price.plot(figsize = (15, 8), fontsize = 13)
plt.legend(fontsize = 13)
plt.show()

print(hm.describe())

mu = hm.Returns.mean()

sigma = hm.Returns.std() #Risk / volatility
print(sigma)
print(np.sqrt(hm.Returns.var()))