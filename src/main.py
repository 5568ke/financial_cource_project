from data_processing.data_loader import DataLoader
from clustering.kmeans import KMeansCluster
from trading_strategy.pair_trading import PairTradingStrategy

def main():
    loader = DataLoader('stock_data.csv')
    data = loader.load_data()
    processed_data = loader.preprocess_data(data)

    cluster = KMeansCluster(n_clusters=5)
    cluster_labels = cluster.fit_predict(processed_data)

    strategy = PairTradingStrategy()
    signals = strategy.generate_signals(processed_data)
    performance = strategy.backtest(signals)

    print("backtest result:", performance)

if __name__ == "__main__":
    main()

