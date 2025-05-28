import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_ema(data, period, day):
    alfa = 2 / (period + 1)
    denominator = 0.0
    numerator = 0.0
    
    for i in range(day - period, day + 1):
        denominator += (1 - alfa)**i
        numerator += data[i] * (1 - alfa)**i

    return numerator / denominator
    
def load_data():
    # data = pd.read_csv("nvda-w-2004-01-22-t-2024-04-08.csv")
    # data = pd.read_csv("sp500-w-2001-05-06-to-2021-04-08.csv")
    data = pd.read_csv("wig20-w-2001-04-16-to-2021-04-9.csv")
    data = data["Close"]
    data = data.to_numpy()
    return data

def simulate(data, macd, signal, assets):
    wallet = assets * data[0]
    bought_indexes = []
    sold_indexes = []
    cash = 0
    looking_to = 'sell'
    prevent_selling_at_start = True
    
    for i in range(len(macd)):
        if prevent_selling_at_start:
            if macd[i] > signal[i]:
                looking_to = 'sell'
                prevent_selling_at_start = False
            else:
                looking_to = 'buy'
                wallet = np.append(wallet, cash + (assets * data[i]))
                continue
        else:
            if macd[i] > signal[i] and looking_to == 'buy':
                assets = cash // data[i]
                cash -= assets * data[i]
                bought_indexes.append(i)
                looking_to = 'sell'
            elif macd[i] < signal[i] and looking_to == 'sell':
                cash = assets * data[i]
                assets = 0
                sold_indexes.append(i)
                looking_to = 'buy'  
                
        wallet = np.append(wallet, cash + (assets * data[i]))
        
    return wallet, int(assets), int(cash), bought_indexes, sold_indexes

if __name__ == "__main__":
    period = 1000
    start_day = 30
    
    data = load_data()
    ema12 = np.zeros(period)
    ema26 = np.zeros(period)
    signal = np.zeros(period)
    macd = np.zeros(period)

    for i in range(period):
        ema12[i] = calculate_ema(data, 12, i + start_day)
        ema26[i] = calculate_ema(data, 26, i + start_day)
        macd[i] = ema12[i] - ema26[i]
        signal[i] = calculate_ema(macd, 9, i)
    
    print("You started with 1000 assets valued at " + str(int(1000 * data[start_day])))
    wallet, assets, cash, bought_indexes, sold_indexes = simulate(data[start_day:period + start_day], macd, signal, 1000)
    print("You ended up with " + str(assets) + " assets valued at " + str(int(assets * data[period + start_day])))
    print("Cash: " + str(cash))
    print("Total value: " + str(int(cash + (assets * data[period + start_day]))) + " cash")
    print("Change (total value): " + str(round(wallet[-1] / (1000 * data[start_day]) * 100, 2)) + "%")
    print("Index change: " + str(round(data[period + start_day] / data[start_day] * 100, 2)) + "%")
    
    plt.plot(macd, label="MACD")
    plt.plot(signal, label="SIGNAL")
    plt.scatter(bought_indexes, macd[bought_indexes], marker="o", c="green", label="BOUGHT")
    plt.scatter(sold_indexes, macd[sold_indexes], marker="o", c="red", label="SOLD")
    plt.title("MACD and SIGNAL")
    plt.legend()
    plt.show()
    
    plt.plot(data[start_day:period + start_day], label="CLOSING VALUE")
    plt.title("CLOSING VALUE")
    plt.legend()
    plt.show()
    
    plt.plot(wallet, label="WALLET VALUE")
    plt.title("WALLET VALUE")
    plt.legend()
    plt.show()
    