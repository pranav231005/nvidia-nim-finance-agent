import yfinance as yf

symbols = [
    "IRFC.NS", "GOODLUCK.NS", "BSE.NS", "LICI.NS", "ASTERDM.NS", "NTPC.NS", 
    "VEDL.NS", "TRUALT.NS", "ETERNAL.NS", "TMPV.NS", "EXIDEIND.NS", "NHPC.NS", 
    "BAJAJHFL.NS", "TRENT.NS", "TATAMOTORS.NS", "HDFCBANK.NS", "RELIANCE.NS", 
    "M&M.NS", "SBIN.NS", "TCS.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "HAL.NS", "NVDA"
]

for s in symbols:
    info = yf.Ticker(s).info
    if 'shortName' in info:
        print(f"{s}: {info['shortName']}")
    else:
        print(f"{s}: NOT FOUND")
