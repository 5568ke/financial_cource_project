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

    processor.load_data()
    processor.preprocess_data()  
    processor.sanitize_and_feature_engineer()
    pca_data = processor.reduce_dimensionality_with_pca()
    print("pca_data : ",pca_data.tail(100))

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

