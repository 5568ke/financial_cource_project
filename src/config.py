# Define the clustering algorithm to use
CLUSTERING_ALGORITHM = 'kmeans'  # Options: 'kmeans', 'dbscan', 'agglomerative', 'faiss'

# File path configurations
PATHS = {
    'kmeans': {
        'models': "../datafile/kmeans/models_df.pkl",
        'data': "../datafile/kmeans/clean_data.pkl"
    },
    'dbscan': {
        'models': "../datafile/dbscan/models_df.pkl",
        'data': "../datafile/dbscan/clean_data.pkl"
    },
    'agglomerative': {
        'models': "../datafile/agglomerative/models_df.pkl",
        'data': "../datafile/agglomerative/clean_data.pkl"
    },
    'faiss': {
        'models': "../datafile/faiss/models_df.pkl",
        'data': "../datafile/faiss/clean_data.pkl"
    }
}

# Algorithm parameters configuration
KMEANS_PARAMS = {
    'n_clusters': 32,
    'random_state': 1211
}
DBSCAN_PARAMS = {
    'alpha': 0.1
}
AGGLOMERATIVE_PARAMS = {
    'n_clusters': 12,
    'affinity': 'euclidean',
    'linkage': 'ward'
}
FAISS_PARAMS = {
    'n_clusters': 12,
    'n_init': 10,
    'max_iter': 300
}
