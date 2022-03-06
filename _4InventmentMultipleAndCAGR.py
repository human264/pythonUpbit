import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")


hm = pd.read_csv("hm.csv", index_col="Date", parse_dates=["Date"])

multiple = (hm.Price[-1]/hm.Price[0])
print(multiple)

#Price increases
print(f"Price0 : {hm.Price[0]}, price1 : {hm.Price[-1]}, Price increase : {(multiple - 1) * 100}")

print(hm.Price / hm.Price[0])
start = hm.index[0]
print(start)
end = hm.index[-1]
print(end)
td = end - start
print(td)
td_years = td.days / 365.25

cagr = multiple**(1/td_years) - 1

print(f"cagr : {cagr}")
print(f"cagr : {cagr2}")





