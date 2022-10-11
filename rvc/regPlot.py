import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import statsmodels.api as sm
import statsmodels.formula.api as smf
import datetime

def regplot(res):
    """
    it will plot follwoing by each column
    1.scatter plot(raw data)
    2.regression plot(y~x)
    3.regression plot(optimized formula)
    """
    num_graph = len(res.col_ex)
    _, axes = plt.subplots(num_graph, 1, figsize=(8, 8*num_graph))

    dir_store = f"_rvc/regPlot"
    if os.path.isdir("_rvc") == False:
        os.mkdir("_rvc")
        os.mkdir(dir_store)
    elif os.path.isdir(dir_store) == False:
        os.mkdir(dir_store)

    for i in range(num_graph):
        colx = res.col_ex[i]
        dfx = res.df.sort_values(colx)
        x = dfx[colx].values
        y = dfx[res.col_target].values

        formula = res.res_formula[i]
        res_opt = smf.ols(formula, data=res.df).fit()
        res_ols = sm.OLS(y, sm.add_constant(x)).fit()
        df_predict = pd.DataFrame(index=range(len(res.df.index)))
        df_predict["y"] = res_opt.predict()
        df_predict["x"] = res.df[colx].values
        df_predict.sort_values("x", inplace=True)

        ax = axes if num_graph == 1 else axes[i]
        ax.scatter(x, y, s=2)
        ax.plot(x, res_ols.predict(), color="orange", alpha=0.5, label="y~x")
        ax.plot(df_predict["x"].values, df_predict["y"].values, color="r", label="opt")
        ax.set_title(f"{colx} {res.res_ind[i]}")
        ax.legend()

    now = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S')
    plt.savefig(dir_store+f"/{now}_{res.col_target}.png")
    plt.close()  # disable to plot
    plt.show()
