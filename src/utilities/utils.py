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
    feature_data = processor.sanitize_and_feature_engineer()
    # feature_data = processor.reduce_dimensionality_with_pca()
    feature_data.columns = feature_data.columns.astype(str)
    print(feature_data.tail(100))
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
    pair_trading_service.run_stress_tests(clustered_data)
    return total_return, month_return

'''
def plot_compare_results():
    # Create a dictionary to store total returns for each clustering algorithm
    all_returns = {}

    # sp500 price list
    sp500_prices = [
        133.00, 117.30, 144.30, 166.40, 171.60, 208.20, 264.50, 250.50, 285.40, 339.97,
        325.49, 416.08, 435.23, 472.99, 465.25, 614.42, 766.22, 963.36, 1248.77, 1425.59,
        1335.63, 1140.21, 895.84, 1132.52, 1181.41, 1278.73, 1424.16, 1378.76, 865.58,
        1123.58, 1282.62, 1300.58, 1480.40, 1822.36, 2028.18, 1918.60, 2275.12, 2789.80,
        2607.39, 3278.20, 3793.75
    ]
    # Adjust SP500 prices relative to the initial price
    sp500_returns = [price / sp500_prices[0] for price in sp500_prices]

    # Iterate over each clustering algorithm
    for algo, paths in config.PATHS.items():
        returns_list_path = paths['returns_list']
        if os.path.exists(returns_list_path):
            # Read total returns from returns_list file
            total_returns = pd.read_pickle(returns_list_path)
            # Store total returns in the dictionary
            all_returns[algo] = total_returns
    all_returns['SP500'] = sp500_returns
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
'''


def get_sp500_monthly_returns(start_year, num_months):
    print("start_year : ", start_year)
    print("num_months : ", num_months)
    # sp500 price list
    sp500_prices = [
        133.00, 117.30, 144.30, 166.40, 171.60, 208.20, 264.50, 250.50, 285.40, 339.97,
        325.49, 416.08, 435.23, 472.99, 465.25, 614.42, 766.22, 963.36, 1248.77, 1425.59,
        1335.63, 1140.21, 895.84, 1132.52, 1181.41, 1278.73, 1424.16, 1378.76, 865.58,
        1123.58, 1282.62, 1300.58, 1480.40, 1822.36, 2028.18, 1918.60, 2275.12, 2789.80,
        2607.39, 3278.20, 3793.75, 4573.82
    ]

    # Calculate the monthly returns for SP500
    start_index = start_year - 1981
    sp500_monthly_returns = []
    for i in range(num_months):
        year_index = start_index + i // 12
        month_index = i % 12
        if month_index == 0 and i > 0:
            start_price = sp500_prices[year_index - 1]
            end_price = sp500_prices[year_index]
        else:
            start_price = sp500_prices[year_index]
            end_price = sp500_prices[year_index + 1]
        monthly_return = (end_price / start_price) ** (1 / 12) - 1
        sp500_monthly_returns.append(monthly_return)
    return sp500_monthly_returns


def plot_stress_test_comparison(stress_test_returns, algo, suffix, start_year):
    # Plotting stress test results
    plt.figure(figsize=(10, 6))
    years = np.arange(start_year, start_year + len(stress_test_returns) / 12,
                      1 / 12)  # Generate years based on length of returns
    plt.plot(years, stress_test_returns, label=f"{algo} ({suffix})")

    average_return = stress_test_returns[-1] ** (12 / len(stress_test_returns)) - 1
    plt.text(len(stress_test_returns) / 12 + start_year, stress_test_returns[-1], f'IRR: {average_return:.2%}',
             fontsize=8.5, verticalalignment='bottom')

    # Add SP500 comparison
    sp500_monthly_returns = get_sp500_monthly_returns(start_year, len(stress_test_returns))
    sp500_cumulative_returns = np.cumprod([1 + ret for ret in sp500_monthly_returns])
    plt.plot(years, sp500_cumulative_returns, label='SP500', linestyle='--')

    plt.title(f'Asset Changes Over {suffix} for {algo}')
    plt.xlabel('Years')
    plt.ylabel('Investment Return Multiplier')
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified path
    compare_result_path = config.COMPARE_RESULT_PATH.replace(".png", f"_{algo}_{suffix}.png")
    plt.savefig(compare_result_path)
    plt.show()


def plot_compare_results():
    # Create a dictionary to store total returns for each clustering algorithm
    all_returns = {}

    # sp500 price list
    sp500_prices = [
        133.00, 117.30, 144.30, 166.40, 171.60, 208.20, 264.50, 250.50, 285.40, 339.97,
        325.49, 416.08, 435.23, 472.99, 465.25, 614.42, 766.22, 963.36, 1248.77, 1425.59,
        1335.63, 1140.21, 895.84, 1132.52, 1181.41, 1278.73, 1424.16, 1378.76, 865.58,
        1123.58, 1282.62, 1300.58, 1480.40, 1822.36, 2028.18, 1918.60, 2275.12, 2789.80,
        2607.39, 3278.20, 3793.75, 4573.82
    ]
    # Adjust SP500 prices relative to the initial price
    sp500_returns = [price / sp500_prices[0] for price in sp500_prices]

    # Iterate over each clustering algorithm
    for algo, paths in config.PATHS.items():
        returns_list_path = paths['returns_list']
        base_path = os.path.dirname(returns_list_path)
        print("base_path : ", base_path)

        # Read main total returns file
        if os.path.exists(returns_list_path):
            total_returns = pd.read_pickle(returns_list_path)
            all_returns[algo] = total_returns

        # Read stress test total returns files
        stress_test_files = [f for f in os.listdir(base_path) if
                             f.startswith("total_returns_") and f.endswith(".pkl") and not f.endswith(
                                 "full_period.pkl")]
        for file in stress_test_files:
            print("in")
            stress_test_suffix = file.split('_')[2] + '_to_' + file.split('_')[4].replace('.pkl', '')
            start_year = int(file.split('_')[2].split('-')[0])
            stress_test_returns = pd.read_pickle(os.path.join(base_path, file))
            plot_stress_test_comparison(stress_test_returns, algo, stress_test_suffix, start_year)
            all_returns[f"{algo}_stress_{stress_test_suffix}"] = stress_test_returns

    all_returns['SP500'] = sp500_returns

    # Plotting full period returns
    plt.figure(figsize=(10, 6))
    for algo, returns in all_returns.items():
        if 'stress' not in algo:  # Exclude stress test returns from this plot
            years = np.arange(1981, 1981 + len(returns))  # Generate years starting from 1980
            plt.plot(years, returns, label=algo)

    # Calculate and annotate average annualized return for full period
    for algo, returns in all_returns.items():
        if 'stress' not in algo:  # Exclude stress test returns from this plot
            average_return = returns[-1] ** (1 / len(returns)) - 1
            plt.text(len(returns) + 1981, returns[-1], f'{algo} IRR: {average_return:.2%}', fontsize=8.5,
                     verticalalignment='bottom')

    plt.title('Comparison of Asset Changes Across Clustering Algorithms (Full Period)')
    plt.xlabel('Years')
    plt.ylabel('Investment Return Multiplier')
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified path
    compare_result_path = config.COMPARE_RESULT_PATH.replace(".png", "_full_period.png")
    plt.savefig(compare_result_path)
    plt.show()
