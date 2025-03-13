import pandas as pd

from search_parameters import ParameterClusterer
from plots import graph_2d_opt

if __name__ == "__main__":

    # Configure the clusterer
    config = {}

    backtest_metrics = [
        'sqn',
        'kratio', 'net_profit',
        'winners', 'recovery_factor'
        ]

    weights_metrics = [
        0.3,
        0.2, 0.2,
        0.1, 0.2,
        ]

    config['backtest_metrics'] = backtest_metrics
    config['weights_metrics'] = weights_metrics
    config['method_cluster'] = 'kmeans'
    config['pct_sample_cluster'] = 5

    # Load the backtest results
    stats_params_backtest = pd.read_csv('stats_backtest.csv')

    # Plot the 2D optimization
    graph_2d_opt(stats_params_backtest, 'length', 'n_bars', 'net_profit', 'XAUUSD', True)

    # Initialize the clusterer
    clusterParam = ParameterClusterer(config)

    # Find best parameters
    best_parameters = clusterParam.find_best_parameters(stats_params_backtest)
