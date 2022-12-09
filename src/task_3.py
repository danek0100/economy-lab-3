from functions import *
def task_3(generated_data, beta, X, stocks, X_est, df_for_graph_selected, mean_vec, cov_matrix, mean_vec_est, cov_matrix_est):
    b = get_parameter_b(beta)

    ########################################################################################################################
    #3.1                                                        
    ########################################################################################################################

    result = cvar_optimize_portfolio(generated_data, beta)
    T = 30
    N = 20
    UXalpha = result
    cvar_X_est = UXalpha[T:N+T]
    alpha_est  = UXalpha[-1]

    plot_shares(cvar_X_est, stocks)
    plot_map_with_one_portfolio(cvar_X_est, mean_vec, cov_matrix, df_for_graph_selected, b)
    
    ########################################################################################################################
    #3.2                                                        
    ########################################################################################################################

    plot_map_with_two_portfolio(df_for_graph_selected, cov_matrix, mean_vec, X, cvar_X_est)
    print('L1 норма вектора X-Xest в 2.3 :', np.around(np.linalg.norm(X - X_est, ord=1), 3))
    print('L1 норма вектора для CVAR X-cvar_Xest:', np.around(np.linalg.norm(X - cvar_X_est, ord=1), 3))

    ########################################################################################################################
    #3.3                                                        
    ########################################################################################################################

    S = 40 

    cvar_experiments = []
    for i in tqdm_notebook(range(S)):
        cvar_experiment = {}
        cvar_experiment['generated_data'] = np.random.multivariate_normal(mean_vec, cov_matrix, T)
        cvar_experiment['mean_vec'] = np.mean(cvar_experiment['generated_data'], axis=0)
        cvar_experiment['cov_matrix'] = np.cov(cvar_experiment['generated_data'].T)
        result = cvar_optimize_portfolio(cvar_experiment['generated_data'], beta)
        cvar_experiment['shapes'] =  result[T:N+T]
        cvar_experiment['L1-norm'] = np.linalg.norm(X - cvar_experiment['shapes'], ord=1)
        cvar_experiment['sigma'] = risk_portfolio(cvar_experiment['shapes'], cov_matrix)
        cvar_experiment['E'] = np.dot(cvar_experiment['shapes'], mean_vec)
        cvar_experiments.append(cvar_experiment)
        
    l1_norms = [cvar_experiment['L1-norm'] for cvar_experiment in cvar_experiments]
    l1_norms_mean  = np.mean(l1_norms)
    print('Средняя L1-норма CVAR по %d экспериментам: %.3f' % (S, l1_norms_mean))

    plot_samples_CVAR(df_for_graph_selected, cvar_experiments,T,N, X, cov_matrix, mean_vec, b)
    plt.show()






   
    