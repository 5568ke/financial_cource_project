import config
from utilities.utils import get_user_decision, load_model_data, train_new_model, manage_trades, plot_compare_results
from clustering.cluster_factory import ClusterFactory

def main():
    # Prompt the user to decide whether to use cached data or retrain the model
    user_decision = get_user_decision()
    cluster_factory = ClusterFactory(config)

    if user_decision == 'use cache data':
        # Use existing cached data for training results
        models_df, clustered_data = load_model_data()
    elif user_decision == 'retrain model':
        # Retrain the model
        models_df, clustered_data = train_new_model(cluster_factory)

    # Manage trades using clustered data
    total_return, month_return = manage_trades(clustered_data)
    print("Finish trading")
    print("Total accumulated return percentage: ", total_return)

    # Call the plot_compare_results function
    plot_compare_results()

    # Print a message indicating that the comparison plot has been generated
    print("Comparison plot of asset changes across clustering algorithms has been generated.")

if __name__ == "__main__":
    main()