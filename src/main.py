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

Es_selected = []
risks_selected = []

for stock in selected_stocks:
    Es_selected.append(stock.E)
    risks_selected.append(stock.risk)

df_for_graph_selected = pd.DataFrame(
    {'σ': risks_selected,
     'E': Es_selected
     })

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


return_matrix, mean_vec, cov_matrix = get_return_mean_cov(selected_stocks_20) 
print("Детерминант матрицы ковариации: " , np.linalg.det(cov_matrix))
eigenvec = np.linalg.eigvals(cov_matrix)
print('Обусловность матрицы ковариации: ', max(eigenvec) /  min(eigenvec))

beta = 0.95
X = task_1(selected_stocks_20, beta, df_for_graph_selected, mean_vec, cov_matrix)
X_est,r_matrix_gen, mean_vec_est, cov_matrix_est = task_2(mean_vec, cov_matrix, selected_stocks_20, beta, df_for_graph_selected, X)
task_3(r_matrix_gen, beta, X, selected_stocks_20, X_est, df_for_graph_selected ,mean_vec, cov_matrix, mean_vec_est, cov_matrix_est)















#
# selected_stocks_from_BOVESPA, df_BOVESPA = task_2(painter, selected_stocks, level_VaR, df_for_graph_selected,
#                                                   "10 from BOVESPA. Profitability/Risk Map", 1)
# selected_stocks_from_market, df_market = task_2(painter, stocks, level_VaR, df_for_graph,
#                                                 "10 from Brazilian Market. Profitability/Risk Map", 2)
# task_3(painter, selected_stocks, df_for_graph_selected)
# task_4(painter, selected_stocks_from_BOVESPA, df_BOVESPA)
# task_4(painter, selected_stocks_from_market, df_market)
# task_5(painter, selected_stocks_from_BOVESPA, df_BOVESPA)
# task_5(painter, selected_stocks_from_market, df_market)

# Бонус
# Индекс Бовеспа - более известный как Ibovespa, является базовым индексом примерно 92 акций, торгуемых на B3
# (Brasil Bolsa Balcão), на долю которого приходится большая часть торговли и рыночной капитализации на
# бразильском фондовом рынке. Это взвешенный индекс измерения.
# Он включает «голубые фишки» Бразилии и другие мение крупные компании из 600+ компаний на бирже.
# Особенные участиники: Оборонно-аэрокосмическая корпорация Embrarer и Фондовая биржа Сан-Паулу.

# Преимущества индекса BOVESPA:
# - завязан на реальный сектор;
# - легко накачивается спекулятивными деньгами;
# - отличается умеренной волатильностью;
# - является единственным фондовым индексом Латинской Америки;

# Особенности индекса BOVESPA:
# o Каждый год или раз в несколько лет учредитель индекса снижает его котировку
# (по отношению к исходному показателю) на несколько единиц.
# Это удешевляет индекс и делает его менее привлекательным для спекулянтов.
# o Помимо BOVESPA, существуют подиндексы.
# - IBRX: 50 включает 50 самых ликвидных бразильских акций в рамках сессии;
# - MLCX: компании с наивысшей или средней капитализацией;
# - SMLL: структуры с малой капитализацией;
# - IEE: компании энергетического сектора;
# - INDX: компании промышленного сектора;

# Зависимости:
# чрезвычайно спекулятивен, хотя экономика Бразилии — мощна;
# зависит от монетарной политики США и Азии;
# регулятор вынужден ограничивать котировку индекса;
# характеризуется мощными немотивированными падениями;
# восстановление рынка занимает годы;

# Метод расчета индекса BOVESPA
# IN = sqrt((ni/N) * (vi/V)) - коэфицент торгуемости.
# ni = количество сделок течение сессии
# N = общее количество торгов
# vi = общий объем торгов
# V = общая стоимость биржи.

# Формула расчета индекса
# Ibovespa(t) = sum((i, N)(P(i,t)*Q(i,t))
# N = общее количество акций в обращении,
# P = цена акций на момент измерения,
# Q = теоретическое количество акций в портфеле.

# Индекс представляет собой общую доходность, составленную из теоретического портфеля следующим образом:
# Критерии отбора из акций бирже:
# o Акция входит 15% лучших по ндивидуальному коэффициента торгуемости (IN);
# o Акиця торгуется в 95% торговых сессий;
# o 0,1% от стоимости, торгуемой на наличныхфондовый рынок (круглые лоты);
# o Не penny.
# o Как минимум 80% акций компании должны быть в свободном обращении;
# o До включения бумаг в корзину они должны торговаться на биржах в течение как минимум 1 года;
# o Торговый оборот должен хотя бы на 0,1% превышать общее число акций доступных на бирже.
# o Взвешенная рыночная стоимость free float / cap 20% на компанию / cap 2x IN
# o Все перечисленные требования должны выполняться в течение года, предшествующего включению бумаг в корзину.
#  ---------------------------------------------------------------------------------------------------------------------
#  Если хотя бы 2 пункта не выполняются, бумагу удаляют из состава индекса,
#  пересмотр состава Бовеспа происходит ежеквартально.

# Bovespa показывает теоретический рост портфеля условного инвестора, сформированный в 1968 г.
# исходя из условий полного реинвестирования дивидендов.


# Индекс пересматривается на основе 4-месячного портфельного цикла в январе, мае и сентябре.
# В среднем на стоимость компонент Ibovespa приходится 70% суммарной стоимости акций, которыми торгуют.

# Индексный номер представляет текущую стоимость портфеля, начатую 2 января 1968 года, с начальной стоимостью 100 и
# с учетом роста цен на акции плюс реинвестирование всех полученных доходов (например от дивидендов).

# Функция взвешивания
# Взвешивание основано на рыночной стоимости, приходящейся на долю акций, находящихся в свободном обращении,
# при этом предел ликвидности (коэффициент торгуемости) установлен в два раза больше гипотетического веса акций.

# Коэффициент торгуемости (Indice de Negociabilidade, или IN) рассчитывается с учетом 1/3 доли компонента в
# общем количестве сделок и 2/3 доли компонента в общей стоимости, торгуемой на рынке наличных акций;

# Пороговое значение коэффициента торгуемости, позволяющее акции претендовать на включение в индекс 85%.
# Требование ‘активной торговли’, измеряемое на основе количества торговых сессий 95%.
# Существует ограничение с точки зрения максимального относительного веса общего вклада компании в индекс.

# index = get_market_index('^BVSP', level_VaR)
# painter.plot_line_graph_from_df(index.profitability, 'Adj Close', 'BOVESPA profitability')
# painter.plot()
# painter.plot_line_graph_from_df(pd.DataFrame({'Close': index.close_price}), 'Close', 'BOVESPA close prices')
# painter.plot()
# painter.plot_gist_from_df(index.profitability, 50)
# painter.plot()
# pca_algorithm(painter, stocks, df_for_graph, 94, index)
# plt.show()
