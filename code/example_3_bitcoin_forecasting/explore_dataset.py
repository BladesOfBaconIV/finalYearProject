import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "../../datasets/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"


def load_csv():
    return pd.read_csv(DATA_FILE)


def visualise():
    df = load_csv()
    print(f'Features: {list(df.keys())}')
    print(f'Presentations: {len(df)}')
    nan_entries = df['Open'].isnull().sum()
    print(f'NaN presentations: {nan_entries} ({(100*nan_entries)/len(df):.2f}% of entries)')
    time = pd.to_datetime(df['Timestamp'], unit='s')
    plt.figure(0)
    plt.plot(time, df['Volume_(Currency)'])
    plt.title('vol cur')

    plt.figure(1)
    plt.plot(time, df['Volume_(BTC)'])
    plt.title('vol btc')

    plt.figure(2)
    plt.plot(time, df['Weighted_Price'])
    plt.title('weighted')

    plt.show()


if __name__=="__main__":
    visualise()