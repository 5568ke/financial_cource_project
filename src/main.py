from data_processing.data_loader import DataLoader
from clustering.kmeans import KMeansCluster
# from trading_strategy.pair_trading import PairTradingStrategy

def main():

    print("step : start loading data")
    loader = DataLoader('datashare.csv')
    data = loader.load_data("../datafile/datashare/datashare.csv")
    print("finish : data loading")
    print("test : print permoo 10006 data", data[10006].monthly_data)

    # processed_data = loader.preprocess_data(data)
    #
    # cluster = KMeansCluster(n_clusters=5)
    # cluster_labels = cluster.fit_predict(processed_data)
    #
    # strategy = PairTradingStrategy()
    # signals = strategy.generate_signals(processed_data)
    # performance = strategy.backtest(signals)
    #
    # print("backtest result:", performance)

if __name__ == "__main__":
    main()

