import pandas as pd
import matplotlib.pyplot as plt

from resource.constant_strings import *
from src.data import get_market_index
from src.functions import *


def call_plot_effective_results(painter, min_risk, min_risk_return, sigmas, returns, colour_point, colour_line, end,
                                points=-1, title='', figure_number=0):
    painter.plot_effective_point(min_risk, min_risk_return, colour_point, title + pwmr + ' ' + end, figure_number)
    painter.plot_effective_front(sigmas, returns, colour_line, title + ef + ' ' + end, points, figure_number)


def task_1(painter, stocks, level_VaR, df_for_graph, set_name, colour_base):
    return None