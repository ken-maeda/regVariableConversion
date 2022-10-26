import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf


class regVariableConversion(object):
    def __init__(self, **kwargs):
        # optional 
        self.on_cpPlot = kwargs.get("on_cpPlot", False)
        self.on_log = kwargs.get("on_log", True)
        self.on_recCp = kwargs.get("on_recCp", False)
        self.on_recAll = kwargs.get("on_recAll", True)

        # key setting
        self.comb = kwargs.get("comb", 3)  # number of factor
        self.KEY = [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "a",
            "b", "c", "d"
        ]

        self.comb_nCr = [0]  #n: key total, r:combination 
        sum = 0
        self.label_list = []  # all combination label
        for i_comb in range(1, 1+self.comb):
            x = len(list(itertools.combinations(range(len(self.KEY)), i_comb)))
            sum += x
            self.comb_nCr.append(sum)
            for combo in itertools.combinations(self.KEY, i_comb):
                label = ""
                for i in range(len(combo)):
                    label += combo[i]
                self.label_list.append(label)

        # decision
        self.sep_homos = 10  # number of separating for homoscedasticity
        self.th_homoAndMse = kwargs["th_homoAndMse"] if "th_homoAndMse" in kwargs else 1.05

        # optional
        self.now = datetime.now().strftime('%Y_%m%d_%H%M%S')
        self.dir_regPlot = "../_rvc/regPlot"
        self.dir_temp = "../_temp"

    def _print(self, log):
        if self.on_log == True:
            print(log)

    def clear(self):
        self.res_formula = []  # "Y ~ + X + np.log(X)"
        self.res_key = []  # "123"
        self.res_ind = []  # ['X3', 'X3', 'logX/X', 'logX']

        # predict
        idx_costs = ["mse", "r", "homo","aic", "1d"]
        self.df_bestCosts_score = pd.DataFrame(index=idx_costs)
        self.df_bestCosts_key = pd.DataFrame(index=idx_costs)
        self.df_bestCosts_mse = pd.DataFrame(index=idx_costs)

        self.df_output = pd.DataFrame()

        # slidingFilter
        self.GAP = []

        # record change point calculation
        self.cp_col  = []  # id
        self.cp_ls = []  # calculation result
        self.cp_cost = []  # cp cost

    def slidingFilter(self):
        min_ = list(self.df.iloc[:, :-1].min())
        for i_col in range(len(min_)):
            gap = 0
            if min_[i_col] < 1:
                if min_[i_col] >= 0:
                    gap = min_[i_col] + 1
                else:
                    gap = -(min_[i_col])*1.1 + 1

            if gap != 0:
                self.df.iloc[:, i_col] = self.df.iloc[:, i_col] + gap

            self.GAP.append(gap)

    def df_init(self, df, col_target):
        """
        change columns name of input dataframe
        ["AT", "V", "EP"] => ["X0", "X1", "Y"] 
        """
        cols = list(df.columns)
        
        col_ex = cols.copy()
        col_ex.remove(col_target)

        self.col_origin = col_ex.copy()
        self.col_origin.append(col_target)
        df = df.reindex(self.col_origin, axis="columns")

        col_replace = []
        for i in range(len(col_ex)): col_replace.append(f"X{i}")
        col_replace.append("Y")
        df.columns = col_replace
    
        self.col_target = col_replace[-1]
        self.col_ex = col_replace[:-1]
        self.df = df

        # sliding Filter
        self.slidingFilter()


    def key2smf(self, keys):
        """ key list is converted to smf format

        Args:
            keys (tuple, list, str): [ex] ("1", "2", "3")

        Returns:
            form (str) : [ex] "Y ~ + X + np.log(X)"
        """
        form = f"{self.col_target} ~ "
        if "1" in keys: form += f"+ {self.col_ex0} "
        if "2" in keys: form += f"+ np.square({self.col_ex0}) "
        if "3" in keys: form += f"+ np.power({(self.col_ex0)}, 3) "
        if "4" in keys: form += f"+ np.log({self.col_ex0}) "
        if "5" in keys: form += f"+ np.reciprocal({self.col_ex0}) "
        if "6" in keys: form += f"+ np.sqrt({self.col_ex0}) "
        if "7" in keys: form += f"+ {self.col_ex0}:np.log({self.col_ex0}) "
        if "8" in keys: form += f"+ np.square({self.col_ex0}):np.log({self.col_ex0}) "
        if "9" in keys: form += f"+ np.log({self.col_ex0}):np.reciprocal({self.col_ex0}) "
        if "a" in keys: form += f"+ np.reciprocal(np.square({self.col_ex0})) "
        if "b" in keys: form += f"+ {self.col_ex0}:np.sqrt({self.col_ex0}) "
        if "c" in keys: form += f"+ np.sqrt(np.log({self.col_ex0})) "
        if "d" in keys: form += f"+ np.sqrt({self.col_ex0}*np.log({self.col_ex0})) "

        return form

    def changepoint(self, Y):
        score = []
        X = range(len(Y))
        for i in range(len(Y)-1):
            if i == 0:
                res1 = sm.OLS(Y, sm.add_constant(X)).fit()
                score.append(np.mean(np.abs(res1.resid)))
            else:
                X1, X2, Y1, Y2 = X[:i+1], X[i:], Y[:i+1], Y[i:]
                res1 = sm.OLS(Y1, sm.add_constant(X1)).fit()
                res2 = sm.OLS(Y2, sm.add_constant(X2)).fit()
                score1 = np.abs(res1.resid)
                score.append(np.sum(score1 + np.abs(res2.resid))/(len(X)+1))

        idx_cp = np.array(score).argmin()

        if self.on_recCp:
            self.cp_col.append(f"{self.col_ex0}")
            self.cp_ls.append(idx_cp)  # calculation result
            self.cp_cost.append(np.array(score))  # cp cost

        return idx_cp

    def publish_cpTable(self):
        df_cp = pd.DataFrame(self.cp_cost, index=self.cp_col).T
        idx_cp = self.cp_ls
        return df_cp, idx_cp

    def cost(self):
        """
        In given a variable, it calculates cost for all formula combination.
        """
        x_ = self.df[self.col_ex0].values
        idx_xsort = np.argsort(x_)
        unit = round(len(x_)/self.sep_homos)

        # formula combination loop
        score = []
        for i_comb in tqdm(range(1, 1+self.comb)):
            for combo in itertools.combinations(self.KEY, i_comb):
                formula= self.key2smf(combo) 
                results = smf.ols(formula, data=self.df).fit()
                resid_sort = results.resid.values[idx_xsort]

                residMean = []
                for i_homos in range(self.sep_homos):
                    residMean.append(np.mean(resid_sort[unit*i_homos:unit*(i_homos+1)]))

                score.append([results.mse_resid, results.rsquared, np.std(residMean), results.aic])

        self.costs = np.array(score).T  #[cost type]x[factor combination]

    def evaluateOnLayer(self):
        evals = [] #  mse, r, homo, aic
        evals_id = [] #  mse, r, homo, aic
        nCr = self.comb_nCr

        # best evals in each factor numbers[layer](1, 2, 3....)
        for i in range(len(nCr)-1):
            c_ = self.costs[:, nCr[i]:nCr[i+1]]
            evals.append(np.min(c_, axis=1))
            evals_id.append(np.argmin(c_, axis=1) + nCr[i])

        evals = np.asarray(evals).T  # [cost type]x[factor layer]
        evals_id = np.asarray(evals_id).T  # [cost type]x[factor layer]

        # Which layer(factor) is appropriate in each cost type?
        # Basically if layer increased, performace also increase. it needs to find converged point.
        res0, res1 = [], []
        # MSE
        idx_mse = self.changepoint(evals[0, :])
        res0.append(evals[0, idx_mse])
        res1.append(int(evals_id[0, idx_mse]))
        # R
        idx_r = self.changepoint(evals[1, :])
        res0.append(evals[1, idx_r])
        res1.append(int(evals_id[1, idx_r]))
        # Homo
        idx_homo = int((evals[2, :]).argmin())
        res0.append(evals[2, idx_homo])
        res1.append(int(evals_id[2, idx_homo]))
        # AIC
        idx_aic = int((evals[3, :]).argmin())
        res0.append(evals[3, idx_aic])
        res1.append(int(evals_id[3, idx_aic]))
        # 1Factor limitation
        res0.append(evals[0, 0])
        res1.append(int(evals_id[0, 0]))

        self.df_bestCosts_score[self.col_ex0] = res0
        self.df_bestCosts_key[self.col_ex0] = res1
        self.df_bestCosts_mse[self.col_ex0] = [
            res0[0],
            self.costs[0, res1[1]],
            self.costs[0, res1[2]],
            self.costs[0, res1[3]],
            self.costs[0, res1[4]]
        ]

    def decision(self):
        costs = self.df_bestCosts_score.loc[:, self.col_ex0].values
        keys = self.df_bestCosts_key.loc[:, self.col_ex0].values
        labels = self.label_list
        r = costs[1]
        numComb_mse, numComb_homo = len(labels[keys[0]]), len(labels[keys[2]])

        if r < 0.6:  # well fitted? => No
            key_ = labels[keys[4]]
            self.res_formula.append(self.key2smf(key_))
            self._print(f"[Result/{self.col_ex0}]:r is {r} >> 1d")
        else:
            mse_homo = self.df_bestCosts_mse.loc["homo", self.col_ex0]
            mse_mse = self.df_bestCosts_mse.loc["mse", self.col_ex0]
            if mse_homo/mse_mse > self.th_homoAndMse:
                key_ = labels[keys[0]]
                self.res_formula.append(self.key2smf(key_))
                self._print(f"[Result/{self.col_ex0}]:homo/mse is {mse_homo/mse_mse} >> mse")
            else:
                if numComb_mse >= numComb_homo:
                    key_ = labels[keys[2]]
                    self.res_formula.append(self.key2smf(key_))
                    self._print(f"[Result/{self.col_ex0}]:mse,homo:{numComb_mse},{numComb_homo} >> mse")
                else:
                    key_ = labels[keys[0]]
                    self.res_formula.append(self.key2smf(key_))
                    self._print(f"[Result/{self.col_ex0}]:mse,homo:{numComb_mse},{numComb_homo} >> homo")

    def publish(self):
        """ 
        Mods: df_output
        """
        index_ = self.col_ex.copy()
        index_.append("Y")
        formula_ = self.res_formula.copy()
        formula_.append(None)
        GAP_ = self.GAP.copy()
        GAP_.append(np.nan)

        self.df_output.index = index_
        self.df_output["formula"] = formula_
        self.df_output["col_origin"] = self.col_origin
        self.df_output["gap"] = GAP_

    def fit(self, df, col_target):
        self.clear()
        self.df_init(df, col_target)

        for i_col in range(len(self.col_ex)):
            self.col_ex0 = self.col_ex[i_col]
            self.cost()
            self.evaluateOnLayer()
            self.decision()
            
            if self.on_recAll: 
                pd.to_pickle(self.costs, f"{self.dir_temp}/{self.now}_{self.col_ex0}.pkl")

        self.publish()

        return self.df_output

    def plot_result(self, df, save=True):
        Path(self.dir_regPlot).mkdir(parents=True, exist_ok=True)

        num_graph = len(self.col_ex)
        _, axes = plt.subplots(num_graph, 1, figsize=(8, 8*num_graph))
        for i_col in range(len(self.col_ex)):
            df_out_ = self.df_output.iloc[[i_col, -1], :]
            col_origin = df_out_.iloc[:, 1].values  # AH, EP
            col_replace = df_out_.index  # X0, Y

            x_ = df.loc[:, col_origin[0]]
            x_dummy = np.linspace(np.min(x_), np.max(x_), 100)

            df_sel = df.loc[:, df_out_.iloc[:, 1].values]
            df_sel.columns = col_replace
            df_sel.iloc[:, 0] = df_sel.iloc[:, 0] + df_out_.iloc[0, -1]  # add gap
            formula = df_out_.iloc[0, 0]

            # optimized regression
            results_opt = smf.ols(formula, data=df_sel).fit()
            y_pred_opt = results_opt.predict(pd.DataFrame(x_dummy, columns=[col_replace[0]])).values

            # linear regression
            x_linear = sm.add_constant(df_sel.values[:, 0])
            x_linear_dummy = sm.add_constant(x_dummy)
            result_linear = sm.OLS(df_sel.values[:, 1], x_linear).fit()
            y_pred_linear = result_linear.predict(x_linear_dummy)

            # plot
            sns.scatterplot(x=col_origin[0], y=col_origin[1], data=df, ax=axes[i_col])
            axes[i_col].plot(x_dummy, y_pred_linear, color="red", linewidth=2, label="y~x")
            axes[i_col].plot(x_dummy, y_pred_opt, color="orange", linewidth=3, label="y~opt(x)")
            axes[i_col].set_title(f"[{col_origin[0]}] {formula}")
            axes[i_col].legend()

        if save == True:
            plt.savefig(self.dir_regPlot+f"/{self.now}_{col_origin[1]}.png")
            plt.close()  # disable to plot

        plt.show()