import os
import config
import pandas as pd
import numpy as np
from data_processing.DataProcessor import DataProcessor
from trading_strategy.trading import PairTradingService
import matplotlib.pyplot as plt
model_path=""
data_path=""
CLUSTERING_ALGORITHM=""

def load_and_prepare_data():
    processor = DataProcessor(
        filepath='../datafile/datashare/datashare.csv',
        filepath_parquet='../datafile/datashare/datashare.parquet',
        min_year=1980,
        max_year=2021
    )
    processor.load_data()
    processor.preprocess_data()
    processor.sanitize_and_feature_engineer()
    feature_data = processor.reduce_dimensionality_with_pca()
    feature_data.columns = feature_data.columns.astype(str)
    return feature_data


def get_user_decision():
    global model_path, data_path
    print("Available clustering algorithms:")
    for key in config.PATHS.keys():
        print(f"- {key}")

    while True:
        chosen_algorithm = input("Please choose a clustering algorithm from the above list: ").strip().lower()
        if chosen_algorithm in config.PATHS:
            config.CLUSTERING_ALGORITHM = chosen_algorithm  # Update the global config with the chosen algorithm
            model_path = config.PATHS[chosen_algorithm]['models']
            data_path = config.PATHS[chosen_algorithm]['data']
            break
        else:
            print("Invalid algorithm choice. Please choose from the list above.")

    if os.path.exists(model_path) and os.path.exists(data_path):
        print(f"Selected model: {chosen_algorithm.upper()}")
        user_prompt = (
            f"Previously trained model data of '{chosen_algorithm.upper()}' is available at '{data_path}'.\n"
            "Do you want to use the existing data cache? If you choose 'yes', the model will use this data "
            "to display trading results. If you choose 'no', the model will be retrained. (yes/no): "
        )
        while True:
            decision = input(user_prompt).strip().lower()
            if decision == 'yes':
                print(
                    f"You chose to use the existing data cache for {chosen_algorithm.upper()} to view trading results.")
                return 'use cache data'
            elif decision == 'no':
                print(f"You chose to retrain the {chosen_algorithm.upper()} model.")
                return 'retrain model'
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
    else:
        print(f"No existing models and data found for {chosen_algorithm.upper()}. Starting training process.")
        return 'retrain model'


def load_model_data():
    models_df = pd.read_pickle(model_path)
    clustered_data = pd.read_pickle(data_path)
    print("model_df : ", models_df)
    print("clustered_data : ", clustered_data)
    print("Loaded models and data from existing files.")
    return models_df, clustered_data


def train_new_model(cluster_factory):
    feature_data = load_and_prepare_data()
    model = cluster_factory.get_cluster_model_instance()
    models_df, clustered_data = model.fit(feature_data)
    models_df.to_pickle(model_path)
    clustered_data.to_pickle(data_path)
    print("model_df : ", models_df)
    print("clustered_data : ", clustered_data)
    print(f"Results saved to {model_path} and {data_path}")
    return models_df, clustered_data

def manage_trades(clustered_data):
    # Prepare data for trading
    clustered_data['DATE'] = pd.to_datetime(clustered_data['DATE'])
    clustered_data.set_index('DATE', inplace=True)

    pair_trading_service = PairTradingService(std_dev_factor=1.5)
    grouped_by_month = clustered_data.groupby(pd.Grouper(freq='ME'))
    total_return, month_return = pair_trading_service.manage_monthly_trades(grouped_by_month)
    return total_return, month_return

def plot_compare_results():
    # Create a dictionary to store total returns for each clustering algorithm
    all_returns = {}

    # Iterate over each clustering algorithm
    for algo, paths in config.PATHS.items():
        returns_list_path = paths['returns_list']
        if os.path.exists(returns_list_path):
            # Read total returns from returns_list file
            total_returns = pd.read_pickle(returns_list_path)
            # Store total returns in the dictionary
            all_returns[algo] = total_returns

    # Plotting
    plt.figure(figsize=(10, 6))
    for algo, returns in all_returns.items():
        years = np.arange(1981, 1981 + len(returns))  # Generate years starting from 1980
        plt.plot(years, returns, label=algo)

    # Calculate and annotate average annualized return
    for algo, returns in all_returns.items():
        average_return = returns[-1] ** (1/len(returns)) - 1
        plt.text(len(returns) + 1981, returns[-1], f'{algo} IRR : {average_return:.2%}', fontsize=8.5, verticalalignment='bottom')

    plt.title('Comparison of Asset Changes Across Clustering Algorithms')
    plt.xlabel('Years')
    plt.ylabel('Investment Return Multiplier')
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified path
    compare_result_path = config.COMPARE_RESULT_PATH
    plt.savefig(compare_result_path)
    plt.show()