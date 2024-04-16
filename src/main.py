from data_processing.DataProcessor import DataProcessor  
from clustering.kmeans import KMeansCluster
# from trading_strategy.pair_trading import PairTradingStrategy

def main():
    processor = DataProcessor(
        filepath='../datafile/datashare/datashare.csv', 
        filepath_parquet='../datafile/datashare/datashare.parquet', 
        min_year=1980, 
        max_year=2021
    )

    data = processor.load_data()
    
    processed_data = processor.preprocess_data()  

    sanitized_data = processor.sanitize_and_feature_engineer()

    print(sanitized_data.tail(5))

    # cluster = KMeansCluster(n_clusters=5)
    # cluster_labels = cluster.fit_predict(sanitized_data)
    #
    # strategy = PairTradingStrategy()
    # signals = strategy.generate_signals(cluster_labels)
    # performance = strategy.backtest(signals)
    #
    # print("Backtest result:", performance)

if __name__ == "__main__":
    main()

