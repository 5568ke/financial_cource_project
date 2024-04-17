from data_processing.DataProcessor import DataProcessor  
from clustering.kmeans import KMeansCluster
from trading_strategy.trading import PairTradingService
from tqdm import tqdm as tdqm
import os
import pandas as pd

def main():
    MODELS_FILEPATH = "../datafile/datashare/models_df.pkl"
    CLEAN_DATA_FILEPATH = "../datafile/datashare/clean_data.pkl"
    OUTLIERS_FILEPATH = "../datafile/datashare/outliers.pkl"

    if os.path.exists(MODELS_FILEPATH) and os.path.exists(CLEAN_DATA_FILEPATH) and os.path.exists(OUTLIERS_FILEPATH):
        models_df = pd.read_pickle(MODELS_FILEPATH)
        clean_data = pd.read_pickle(CLEAN_DATA_FILEPATH)
        outliers = pd.read_pickle(OUTLIERS_FILEPATH)
    else:
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

        pca_data.columns = pca_data.columns.astype(str)
        print("pca_data : ", pca_data.tail(100))

        kmeans = KMeansCluster(n_clusters=50, random_state=42)
        models_df, clean_data, outliers = kmeans.fit(pca_data)

        models_df.to_pickle(MODELS_FILEPATH)
        clean_data.to_pickle(CLEAN_DATA_FILEPATH)  
        outliers.to_pickle(OUTLIERS_FILEPATH)      

    print("Cluster information:")
    print(models_df)

    print("Clean data information:")
    print("Number of samples in clean data:", len(clean_data))
    print("Number of samples in each cluster:")
    print(clean_data['km_cluster'].value_counts())

    clean_data['DATE'] = pd.to_datetime(clean_data['DATE'])
    clean_data.set_index('DATE', inplace=True)

    pair_trading_service = PairTradingService(std_dev_factor=1.5)

    # Group the data by month, assuming the DataFrame is already sorted by date index
    grouped_by_month = clean_data.groupby(pd.Grouper(freq='ME'))

    month_data_dict = {}  # Store each month's data for later use
    portfolios_dict = {}  # Store the portfolio for each month

    for month, month_data in tdqm(grouped_by_month):
        month_data_dict[month] = month_data
        # Find trading pairs and form a new portfolio for the current month
        grouped_by_cluster = month_data.groupby('km_cluster')
        current_month_portfolio = {'long': [], 'short': []}
        for cluster, cluster_data in grouped_by_cluster:
            pairs = pair_trading_service.find_pairs(cluster_data)
            portfolio = pair_trading_service.form_portfolio(pairs)
            current_month_portfolio['long'].extend(portfolio['long'])
            current_month_portfolio['short'].extend(portfolio['short'])
        
        portfolios_dict[month] = current_month_portfolio

    print("finish trading")

    # Calculate the returns for each monthly portfolio
    month_return = []
    for month in portfolios_dict:
        if month + pd.DateOffset(months=1) in month_data_dict:
            next_month_data = month_data_dict[month + pd.DateOffset(months=1)]
            portfolio = portfolios_dict[month]
            returns = pair_trading_service.calculate_monthly_returns(portfolio, next_month_data)
            print(f"Returns for {month.strftime('%Y-%m')}: {returns}%")
            month_return.append([month,returns])
        
    total_return = 1
    for data in month_return:
        print("month : " + str(data[0]) + ", return : "+ str(data[1]))
        total_return *= (1+data[1])
        print("accumulate return percentage : ", total_return)
    print("total_return : ",total_return)
        
    
if __name__ == "__main__":
    main()
