# Define the clustering algorithm to use
CLUSTERING_ALGORITHM = 'kmeans'  # Options: 'kmeans', 'dbscan', 'agglomerative', 'faiss'


# File path configurations
PATHS = {
    'kmeans': {
        'models': "../datafile/kmeans/models_df.pkl",
        'data': "../datafile/kmeans/clean_data.pkl",
        'plot': "../plots/kmeans_asset_changes.png",
        'returns_list': "../datafile/kmeans/total_returns.pkl"  # Add returns_list path for kmeans
    },
    'dbscan': {
        'models': "../datafile/dbscan/models_df.pkl",
        'data': "../datafile/dbscan/clean_data.pkl",
        'plot': "../plots/dbscan_asset_changes.png",
        'returns_list': "../datafile/dbscan/total_returns.pkl"  # Add returns_list path for dbscan
    },
    'agglo': {
        'models': "../datafile/agglomerative/models_df.pkl",
        'data': "../datafile/agglomerative/clean_data.pkl",
        'plot': "../plots/agglomerative_asset_changes.png",
        'returns_list': "../datafile/agglomerative/total_returns.pkl"  # Add returns_list path for agglomerative
    },
    'faiss': {
        'models': "../datafile/faiss/models_df.pkl",
        'data': "../datafile/faiss/clean_data.pkl",
        'plot': "../plots/faiss_asset_changes.png",
        'returns_list': "../datafile/faiss/total_returns.pkl"  # Add returns_list path for faiss
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
FAISS_PARAMS = {
    'n_clusters': 12,
    'n_init': 10,
    'max_iter': 300
}

# Path for saving comparison result plot
COMPARE_RESULT_PATH = "../plots/clustering_comparison.png"