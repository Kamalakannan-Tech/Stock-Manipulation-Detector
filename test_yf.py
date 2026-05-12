import yfinance as yf
import time

tickers = ['GME', 'AMC', 'TSLA', 'AAPL', 'MSFT', 'NVDA']

t0 = time.time()
for t in tickers:
    tick = yf.Ticker(t)
    try:
        price = tick.fast_info['lastPrice']
        print(f"{t}: {price}")
    except Exception as e:
        print(e)
print(f"Time taken: {time.time() - t0:.2f} seconds")
