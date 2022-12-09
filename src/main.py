import os
from data import *
from stock import Stock
from task_1 import task_1
from task_2 import task_2
from task_3 import task_3
from functions import *

ids_path = "../resource/brazil_ids.csv"
level_VaR = '0.95'

# Провряем существует ли файл с id доступных акций.
if not os.path.isfile(ids_path):
    print("Ids didn't found")
    exit(1)

# Загружаем данные, если они не были загружены ранее.
if not os.path.isdir('../resource/data/'):
    get_data(ids_path)

# Формируем список акций из данных. Формируем логорифмические доходности, а также вычисляем оценки.

stocks_for_remove = []
stocks = get_Stocks(level_VaR)
for stock in stocks:
    if stock.risk > 0.5:
        stocks_for_remove.append(stock)

for stock in stocks_for_remove:
    stocks.remove(stock)

# Выбрали 50 акций, которые входят в состав индекса Bovespa
selected_stocks_names = ['QUAL3', 'RADL3', 'GOAU4', 'BEEF3', 'MRVE3', 'MULT3', 'NTCO3', 'PETR3', 'PETR4', 'PRIO3',
                         'POSI3', 'ABEV3', 'LAME3', 'ARZZ3', 'B3SA3', 'BPAN4', 'BBSE3', 'BRML3', 'BBDC3', 'BBDC4',
                         'BRAP4', 'PCAR4', 'BRKM5', 'CCRO3', 'CIEL3', 'COGN3', 'CPLE6', 'CSAN3', 'CPFE3', 'CVCB3',
                         'CYRE3', 'DXCO3', 'ECOR3', 'ENBR3', 'EMBR3', 'ENEV3', 'EGIE3', 'EQTL3', 'EZTC3', 'FLRY3',
                         'GGBR4', 'GOLL4', 'HYPE3', 'ITSA4', 'ITUB4', 'JBSS3', 'RENT3', 'LREN3', 'VALE3', 'MRFG3']

selected_stocks = []
for name_stock in selected_stocks_names:
    find_stock = next(stock for stock in stocks if stock.key == name_stock)
    selected_stocks.append(find_stock)


sharpe_ratio = pd.DataFrame()
for stock in selected_stocks:
    sharpe_ratio.insert(0 , str(stock.key) , [(stock.E)**2 / stock.risk])

sharpe_ratio = sharpe_ratio.sort_values(by = 0,  axis = 1)

name_stocks = []
for col in sharpe_ratio.columns:
    name_stocks.append(col)

name_stocks = name_stocks[-20:]

selected_stocks_20 = []
for name_stock in name_stocks:
    find_stock = next(stock for stock in stocks if stock.key == name_stock)
    selected_stocks_20.append(find_stock)

Es_selected = []
risks_selected = []

for stock in selected_stocks_20:
    Es_selected.append(stock.E)
    risks_selected.append(stock.risk)

df_for_graph_selected = pd.DataFrame(
    {'σ': risks_selected,
     'E': Es_selected
     })

sns.set_style('dark')
plt.grid()
sns.scatterplot(data=df_for_graph_selected, x='σ', y='E', c='#6C8CD5', label='Stocks', edgecolors = 'white').set_title("Profitability/Risk Map")
plt.legend(["Stocks"])
plt.show()

return_matrix, mean_vec, cov_matrix = get_return_mean_cov(selected_stocks_20) 
print("Детерминант матрицы ковариации: " , np.linalg.det(cov_matrix))
eigenvec = np.linalg.eigvals(cov_matrix)
print('Обусловность матрицы ковариации: ', max(eigenvec) /  min(eigenvec))

beta = 0.95
X = task_1(selected_stocks_20, beta, df_for_graph_selected, mean_vec, cov_matrix)
X_generated, generated_data, mean_vec_generated, cov_matrix_generated = task_2(mean_vec, cov_matrix, selected_stocks_20, beta, df_for_graph_selected, X)
task_3(generated_data, beta, X, selected_stocks_20, X_generated, df_for_graph_selected ,mean_vec, cov_matrix, mean_vec_generated, cov_matrix_generated)
