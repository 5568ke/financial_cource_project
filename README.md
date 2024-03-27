# Financial course final Project

## Introduction
This project is the final assignment for the course "Topics in Financial Mathematics I", aiming to replicate the research presented in the paper "Pair Trading via Unsupervised Learning". The goal is to apply various unsupervised learning techniques to develop a pair trading strategy that can be utilized in the financial markets.

## Project Structure

Below is the directory structure of the project and explanations for each module:

```
pair_trading_project/
│
├── data/ # Data storage directory
│ ├── raw/ # Raw data
│ └── processed/ # Preprocessed data
│
├── docs/ # Documentation and project explanation
│
├── src/ # Source code directory
│ ├── init.py
│ ├── main.py # Program entry point and workflow control
│ ├── config.py # Configuration file processing
│ ├── data_processing/ # Data processing module
│ │ ├── init.py
│ │ └── data_loader.py # Data loading and preprocessing
│ │
│ ├── clustering/ # Clustering analysis module
│ │ ├── init.py
│ │ ├── cluster_base.py # Base class for clustering algorithms
│ │ ├── kmeans.py # K-means implementation
│ │ ├── dbscan.py # DBSCAN implementation
│ │ └── agglomerative.py# Agglomerative clustering implementation
│ │
│ ├── trading_strategy/ # Trading strategy module
│ │ ├── init.py
│ │ ├── strategy_base.py# Base class for strategies
│ │ └── pair_trading.py # Pair trading strategy implementation
│ │
│ └── utilities/ # Utilities module
│ ├── init.py
│ ├── logger.py # Logging management
│ └── metrics.py # Performance evaluation metrics
│
├── tests/ # Unit testing directory
│ ├── init.py
│ └── test_data_loader.py
│
└── requirements.txt # Project dependencies
```


## Modules Description

- `data/`: Contains all the data used in the project, including raw financial data and processed data ready for analysis.
- `docs/`: Stores documentation related to the project, including setup instructions and methodology explanations.
- `src/`: The core of the project, including all source code for data processing, clustering algorithms, and trading strategies.
  - `data_processing/`: Handles data loading from various sources and preprocessing tasks such as cleaning and normalization.
  - `clustering/`: Implements various unsupervised learning algorithms to identify pairs in the financial data.
  - `trading_strategy/`: Contains the logic for executing pair trading based on the output from the clustering algorithms.
  - `utilities/`: Provides additional functionalities like logging and performance metrics to support the main modules.
- `tests/`: Contains unit tests for various components of the project to ensure code reliability and correctness.
- `requirements.txt`: Lists all the Python packages required to run the project.

## Design Patterns and Technologies Used

- **Strategy Pattern**: Implemented in the `trading_strategy/` module to allow easy interchangeability between different trading strategies.
- **Factory Pattern**: Utilized for creating instances of different clustering algorithms in the `clustering/` module, promoting flexibility and extensibility.
- **Singleton Pattern**: Applied in the `config.py` for managing global configurations throughout the application.
- **Observer Pattern**: Could be used in conjunction with the `logger.py` to notify subscribers of log events throughout the application's execution.

## Getting Started

To run this project, you'll need Python 3.6+ installed on your system. First, install the required dependencies:

```bash
pip install -r requirements.txt
```
Next, navigate to the src/ directory and run the main.py script to start the analysis:
```
cd src
python main.py
```

