from data_processing.DataProcessor import DataProcessor  
from clustering.kmeans import KMeansCluster
import os
import pandas as pd

def main():
    MODELS_FILEPATH = "../datafile/datashare/models_df.pkl"
    CLEAN_DATA_FILEPATH = "../datafile/datashare/clean_data.pkl"
    OUTLIERS_FILEPATH = "../datafile/datashare/outliers.pkl"

    if os.path.exists(MODELS_FILEPATH) and os.path.exists(CLEAN_DATA_FILEPATH) and os.path.exists(OUTLIERS_FILEPATH):
        km_models_df = pd.read_pickle(MODELS_FILEPATH)
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

        kmeans = KMeansCluster(n_clusters=4, random_state=42)
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

    print("Outliers information:")
    print("Number of outliers:", len(outliers))
    print("Outliers data:")
    print(outliers)

if __name__ == "__main__":
    main()
