import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd

from function_cluster import (GMMClustering,
                              KMeansClustering,
                              AgglomerativeClusteringOptimal)


class ParameterClusterer:
    """
    A class for clustering and finding optimal parameters based on backtest metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the ParameterClusterer with configuration settings.
        
        Args:
            config (dict): Configuration dictionary containing:
                - backtest_metrics (list): Metrics to use for clustering
                - weights_metrics (list): Weights to apply to each metric
                - pct_sample_cluster (int): Minimum percentage of samples a cluster must have
                - method_cluster (str): Clustering method ('kmeans', 'gmm', or 'agglomerative')
                - min_samples (int): Minimum number of samples needed for clustering
                
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate required parameters
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
            
        # Check for required parameters
        required_params = ['backtest_metrics', 'weights_metrics']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
            
        # Validate backtest_metrics and weights_metrics
        if not isinstance(config['backtest_metrics'], list):
            raise ValueError("backtest_metrics must be a list")
        if not isinstance(config['weights_metrics'], list):
            raise ValueError("weights_metrics must be a list")
        if len(config['backtest_metrics']) != len(config['weights_metrics']):
            raise ValueError("backtest_metrics and weights_metrics must have the same length")
            
        # Validate method_cluster if provided
        valid_methods = ['kmeans', 'gmm', 'agglomerative']
        method_cluster = config.get('method_cluster', 'kmeans')
        if method_cluster not in valid_methods:
            raise ValueError(f"Invalid clustering method. Must be one of: {', '.join(valid_methods)}")
            
        # Assign values with validation
        self.backtest_metrics = config['backtest_metrics']
        self.weights_metrics = config['weights_metrics']
        self.pct_sample_cluster = config.get('pct_sample_cluster', 15)
        self.method_cluster = method_cluster
        self.min_samples = config.get('min_samples', 30)
        
        # Validate numeric parameters
        if not isinstance(self.pct_sample_cluster, (int, float)) or self.pct_sample_cluster <= 0:
            raise ValueError("pct_sample_cluster must be a positive number")
        if not isinstance(self.min_samples, int) or self.min_samples < 30:
            raise ValueError("min_samples must be 30")
        
    @staticmethod
    def sigmoid_normalize(x):
        """
        Apply sigmoid normalization to center and scale data.
        
        Args:
            x (pd.Series): Data to normalize
            
        Returns:
            pd.Series: Normalized data
        """
        # Centrar los datos alrededor de la media antes de aplicar sigmoid
        x_centered = (x - x.mean()) / x.std()
        return 1 / (1 + np.exp(-x_centered))
    
    def _preprocess_data(self, stats_params_backtest):
        """
        Preprocess the data by filtering zero rows and normalizing.
        
        Args:
            stats_params_backtest (pd.DataFrame): Raw parameter statistics
            
        Returns:
            tuple: (filtered DataFrame, normalized DataFrame)
        """
        # Filter out rows where all metrics are zero
        metrics_data = stats_params_backtest[self.backtest_metrics]
        non_zero_mask = ~(metrics_data == 0).all(axis=1)
        # Create a copy to avoid SettingWithCopyWarning
        filtered_data = stats_params_backtest.loc[non_zero_mask].copy()
        
        # Validate that there's enough data for clustering after filtering
        if len(filtered_data) < self.min_samples:
            return None, None
            
        # Apply sigmoid normalization by column
        normalized_data = filtered_data.loc[:, self.backtest_metrics].apply(self.sigmoid_normalize)
        
        # Apply weights to normalized data
        normalized_data = normalized_data.mul(self.weights_metrics, axis=1)
        
        # Clean NaN values (although with sigmoid they rarely occur)
        normalized_data.fillna(0, inplace=True)
        
        return filtered_data, normalized_data
    
    def _perform_clustering(self, normalized_data):
        """
        Perform clustering on normalized data using the specified method.
        
        Args:
            normalized_data (pd.DataFrame): Normalized data for clustering
            
        Returns:
            np.array: Cluster labels
        """
        if self.method_cluster == 'kmeans':
            clusterer = KMeansClustering(max_clusters=15)
        elif self.method_cluster == 'gmm':
            clusterer = GMMClustering(max_components=15, criterion='bic')
        else:  # agglomerative
            clusterer = AgglomerativeClusteringOptimal(max_clusters=15)
            
        return clusterer.fit_predict(normalized_data)
    
    def _evaluate_clusters(self, stats_params_backtest, cluster_labels):
        """
        Evaluate clusters and find the best one.
        
        Args:
            stats_params_backtest (pd.DataFrame): Parameter statistics
            cluster_labels (np.array): Cluster assignments
            
        Returns:
            int: Label of the best cluster
        """
        stats_params_backtest["cluster_label"] = cluster_labels
        
        # Count samples per cluster
        stats_count = stats_params_backtest.groupby("cluster_label").count()[stats_params_backtest.columns[0]]
        stats_count.name = 'num_sample_cluster'
        
        # Calculate median metrics by cluster
        stats_backtest_mean = stats_params_backtest.loc[:, self.backtest_metrics + ['cluster_label']]
        stats_backtest_mean = stats_backtest_mean.groupby('cluster_label').median()
        stats_backtest_mean = pd.concat([stats_backtest_mean, stats_count], axis=1)
        
        # Filter clusters by size
        if self.pct_sample_cluster is not None:
            min_samples = int(len(stats_params_backtest) * self.pct_sample_cluster / 100)
            stats_backtest_mean = stats_backtest_mean[
                stats_backtest_mean['num_sample_cluster'] > min_samples
            ]
        
        # Normalize metrics
        stats_backtest_mean_norm = stats_backtest_mean.loc[:, self.backtest_metrics + ['num_sample_cluster']]
        stats_backtest_mean_norm = stats_backtest_mean_norm / stats_backtest_mean_norm.max()
        
        # Evaluate clusters
        cluster_evaluation = []
        
        for cluster_label in stats_backtest_mean_norm.index:
            cluster_data = stats_params_backtest[
                stats_params_backtest["cluster_label"] == cluster_label
            ].loc[:, self.backtest_metrics]
            
            # Intra-cluster variability
            intra_cluster_variability = cluster_data.var().median()
            
            # Average distance between points
            pairwise_distances = pdist(cluster_data.values, metric='euclidean')
            avg_pairwise_distance = np.median(pairwise_distances) if len(pairwise_distances) > 0 else 0
            
            # Combined metric
            combined_metric = 1 / (1 + intra_cluster_variability + avg_pairwise_distance)
            cluster_evaluation.append(combined_metric)
        
        stats_backtest_mean_norm['cluster_evaluation'] = cluster_evaluation
        return stats_backtest_mean_norm.sum(axis=1).idxmax()
    
    def find_best_parameters(self, stats_params_backtest):
        """
        Find the best parameters by clustering and evaluating backtest results.
        
        Args:
            stats_params_backtest (pd.DataFrame): DataFrame with parameter statistics
            
        Returns:
            pd.Series: Best parameters (median of best cluster)
        """
        # Preprocess data
        filtered_data, normalized_data = self._preprocess_data(stats_params_backtest)
        if filtered_data is None:
            return None
            
        # Perform clustering
        cluster_labels = self._perform_clustering(normalized_data)
        
        # Evaluate clusters and find the best one
        best_cluster_label = self._evaluate_clusters(filtered_data, cluster_labels)
        
        # Get median parameters from best cluster
        best_params = filtered_data[filtered_data["cluster_label"].eq(best_cluster_label)]
        return best_params.median(axis=0)
