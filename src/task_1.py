from functions import get_parameter_b, optimize_portfolio, plot_shares, plot_map_with_one_portfolio

def task_1(selected_stocks, beta, df_for_graph_selected, mean_vec, cov_matrix):
    b = get_parameter_b(beta)
    X = optimize_portfolio(mean_vec, cov_matrix, b, bounds=((0, 1),))
    plot_shares(X, selected_stocks)
    plot_map_with_one_portfolio(X, mean_vec, cov_matrix, df_for_graph_selected, b)
    return X
    