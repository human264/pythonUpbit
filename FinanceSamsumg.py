import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")

start = "2014-10-01"
end = "2022-03-04"

## 329180.KS : 현대중공업
symbol = ["005930.KS", "005380.KS", "066570.KS", "GC=F", "BTC-USD"]
df= yf.download(symbol, start, end)
df = df.rename(columns={"005930.KS":"SAMSUNG","005380.KS":"HYUNDAI_MOBILE","066570.KS":"LG"})

print(df)

df.to_csv("koreaAsset.csv")

df = pd.read_csv("koreaAsset.csv", header=[0, 1], index_col=0, parse_dates=[0])
close = df.Close.copy()
print(df)

# normal
norm = close.div(close.iloc[1]).mul(100)
print(norm)

norm.dropna().plot(figsize=(15,8), fontsize = 13, logy=True)
plt.legend(fontsize = 13)
plt.show()

close.to_csv("close.csv")