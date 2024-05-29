# Define the clustering algorithm to use
CLUSTERING_ALGORITHM = 'kmeans'  # Options: 'kmeans', 'dbscan', 'agglomerative', 'faiss'


# File path configurations
PATHS = {
    'kmeans': {
        'models': "../datafile/kmeans/models_df.pkl",
        'data': "../datafile/kmeans/clean_data.pkl",
        'plot': "../plots/kmeans_asset_changes.png",
        'returns_list': "../datafile/kmeans/total_returns_full_period.pkl",  # Add returns_list path for kmeans
        'monthly_return_plot':"../plots/kmeans_monthly_return_plot.png"
    },
    'dbscan': {
        'models': "../datafile/dbscan/models_df.pkl",
        'data': "../datafile/dbscan/clean_data.pkl",
        'plot': "../plots/dbscan_asset_changes.png",
        'returns_list': "../datafile/dbscan/total_returns_full_period.pkl",  # Add returns_list path for dbscan
        'monthly_return_plot':"../plots/dbscan_monthly_return_plot.png"
    },
    'agglo': {
        'models': "../datafile/agglomerative/models_df.pkl",
        'data': "../datafile/agglomerative/clean_data.pkl",
        'plot': "../plots/agglomerative_asset_changes.png",
        'returns_list': "../datafile/agglomerative/total_returns_full_period.pkl",  # Add returns_list path for agglomerative
        'monthly_return_plot':"../plots/agglo_monthly_return_plot.png"
    }
}

# Algorithm parameters configuration
KMEANS_PARAMS = {
    'n_clusters': 32,
    'random_state': 1211
}
DBSCAN_PARAMS = {
    'alpha': 0.9
}
AGGLOMERATIVE_PARAMS = {
    'alpha': 0.99,
    'linkage': 'complete'
}

# Path for saving comparison result plot
COMPARE_RESULT_PATH = "../plots/clustering_comparison.png"

# Stress test periods
STRESS_TEST_PERIODS = [
    {"start": "2007-01-01", "end": "2009-12-31"},  # 2008 financial crisis
    {"start": "1999-01-01", "end": "2001-12-31"},  # Dot-com bubble
    {"start": "2019-01-01", "end": "2021-12-31"}   # COVID-19 pandemic
]