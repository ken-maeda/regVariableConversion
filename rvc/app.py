import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import itertools
from tqdm import tqdm
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf

class regVariableConversion(object):
    """
    # Tour for adding new KEY
    1.__init__ (self.key)
    2._key2smf ()
    3._key2ind ()
    4._publish ()
    """

    def __init__(self, **kwargs):
        self.CPonPlot = kwargs["CPonPlot"] if "CPonPLot" in kwargs else False
        self.onCombine = kwargs["onCombine"] if "onCombine" in kwargs else False
        self.onLog = kwargs["onLog"] if "onLog" in kwargs else False

        self.COMB = 4
        self.KEY = [
            "1", "2", "3", "4", "5",
            "6", "7", "8", "9", "a",
            "b", "c", "d"
        ]
        self.comb_nCr, sum = [0], 0
        for i in range(1, 1+self.COMB):
            x = len(list(itertools.combinations(range(len(self.KEY)), i)))
            sum += x
            self.comb_nCr.append(sum)

        self.HOMOS = 10  # number of separating for homoscedasticity
        self.th_homoAndMse = kwargs["th_homoAndMse"] if "th_homoAndMse" in kwargs else 1.05

        self._clear()

    def _clear(self):
        self.res_formula = []  # "Y ~ + X + np.log(X)"
        self.res_key = []  # "123"
        self.res_ind = []  # ['X3', 'X3', 'logX/X', 'logX']

        # predict
        self.col_target, self.col_ex = None, None
        self.df, self.df_cost = None, None
        self.df_mse_bestCosts, self.df_bestCosts = None, None
        self.ls_df_cost = []

        # _slidingFilter
        self.GAP = {}
        self.df_output = pd.DataFrame()

    def _print(self, var):
        if self.onLog == True:
            print(var)

    #####################################################################################
    ##############################################################   tool functions   ###
    #####################################################################################

    def _key2smf(self, label_list, returnLabel=False):
        """ key list is converted to smf format

        Args:
            label_list (tuple, list, str): [ex] ("1", "2", "3")
            returnLabel (bool): if label is returned or not

        Returns:
            form (str) : [ex] "Y ~ + X + np.log(X)"
            [label (str)] : [ex] "123"
        """
        label = ""
        for i in range(len(label_list)):
            label += label_list[i]

        form = f"{self.col_target} ~ "
        if "1" in label_list: form += f"+ {self.col_ex0} "
        if "2" in label_list: form += f"+ np.square({self.col_ex0}) "
        if "3" in label_list: form += f"+ np.power({(self.col_ex0)}, 3) "
        if "4" in label_list: form += f"+ np.log({self.col_ex0}) "
        if "5" in label_list: form += f"+ np.reciprocal({self.col_ex0}) "
        if "6" in label_list: form += f"+ np.sqrt({self.col_ex0}) "
        if "7" in label_list: form += f"+ {self.col_ex0}:np.log({self.col_ex0}) "
        if "8" in label_list: form += f"+ np.square({self.col_ex0}):np.log({self.col_ex0}) "
        if "9" in label_list: form += f"+ np.log({self.col_ex0}):np.reciprocal({self.col_ex0}) "
        if "a" in label_list: form += f"+ np.reciprocal(np.square({self.col_ex0})) "
        if "b" in label_list: form += f"+ {self.col_ex0}:np.sqrt({self.col_ex0}) "
        if "c" in label_list: form += f"+ np.sqrt(np.log({self.col_ex0})) "
        if "d" in label_list: form += f"+ np.sqrt({self.col_ex0}*np.log({self.col_ex0})) "
        if returnLabel:
            return form, label
        else:
            return form

    def _key2ind(self, key):
        """ key is converted the name human can recognize easily.

        Args:
            key (str): [ex] "3394"

        Returns:
            list: [ex] ['X3', 'X3', 'logX/X', 'logX']
        """
        result = [s.replace('1', 'X') for s in key]
        result = [s.replace('2', 'X2') for s in result]
        result = [s.replace('3', 'X3') for s in result]
        result = [s.replace('4', 'logX') for s in result]
        result = [s.replace('5', '1/X') for s in result]
        result = [s.replace('6', '√X') for s in result]
        result = [s.replace('7', 'X_logX') for s in result]
        result = [s.replace('8', 'X2_logX') for s in result]
        result = [s.replace('9', 'logX/X') for s in result]
        result = [s.replace('a', '1/X2') for s in result]
        result = [s.replace('b', 'X√X') for s in result]
        result = [s.replace('c', '√logx') for s in result]
        result = [s.replace('d', '√XlogX') for s in result]
        return result

    def _slidingFilter(self, df):
        dfx = df.drop(self.col_target, axis=1)
        min_ = list(dfx.min())
        for i_col in range(len(min_)):
            gap = 0
            if min_[i_col] < 1:
                if min_[i_col] >= 0:
                    gap = min_[i_col] + 1
                else:
                    gap = -(min_[i_col])*1.1 + 1

            if gap != 0:
                dfx.iloc[:, i_col] = dfx.iloc[:, i_col] + gap

            self.GAP[dfx.columns[i_col]] = gap

        dfx[self.col_target] = self.df[self.col_target]

        return dfx.copy()

    #####################################################################################
    ##############################################################   main functions   ###
    #####################################################################################

    def _cost(self):
        """ caluculate cost in all combination of element
        Cost are MSE, R, Homoscedasticity, AIC

        Returns: None
        """
        df_cost = pd.DataFrame(index=["mse", "r", "homo", "aic"])

        def homoscedasticity():
            df_homo = pd.DataFrame(columns=["resid", "X"])
            df_homo["resid"] = results.resid.values
            df_homo["X"] = self.df[self.col_ex0].values
            df_homo.sort_values(by="X", inplace=True)
            df_homo.reset_index(drop=True, inplace=True)
            residMean = []
            unit = round(df_homo.shape[0]/self.HOMOS)
            for i_homos in range(self.HOMOS):
                residMean.append(df_homo.iloc[unit*i_homos:unit*(i_homos+1), 0].mean())

            return np.std(residMean)

        for i_comb in tqdm(range(1, 1+self.COMB)):
            for combo in itertools.combinations(self.KEY, i_comb):
                formula, label = self._key2smf(combo, returnLabel=True)  # ex will be updated
                results = smf.ols(formula, data=self.df).fit()

                homo = homoscedasticity()

                df_cost[label] = [results.mse_resid, results.rsquared, homo, results.aic]

        return df_cost

    def _evaluateOnLayer(self):
        """evaluate cost on each layer of element number,and decide best element
         combination in a cost. df_mse_allCost is for comapring these by same
         character.

        Returns: None

        Mods:df_bestEachCost, df_mse_allCost
        """
        def changepoint(Y):
            """ find the converge point
            Args: Y (numpy,ndarray):
            Returns: int: changepoint relative index
            """
            res = []
            X = range(len(Y))
            for i in range(len(Y)-1):
                if i == 0:
                    model1 = sm.OLS(Y, sm.add_constant(X))
                    result1 = model1.fit()
                    res.append(np.abs(result1.resid).mean())
                else:
                    X1, X2, Y1, Y2 = X[0:i+1], X[i:], Y[0:i+1], Y[i:]
                    model1 = sm.OLS(Y1, sm.add_constant(X1))
                    result1 = model1.fit()
                    result_x = np.abs(result1.resid).mean()
                    model2 = sm.OLS(Y2, sm.add_constant(X2))
                    result2 = model2.fit()
                    res.append((result_x + np.abs(result2.resid).mean())/2)

            out = np.array(res).argmin()
            if self.CPonPlot:
                plt.figure(figsize=(12, 3))
                plt.plot(Y)
                plt.axvline(out, color="r")
                plt.title(f"{self.col_ex0}")
                plt.show()

            return out

        mse, r, homo, aic = [], [], [], []
        mse_label, r_label, homo_label, aic_label = [], [], [], []
        nCr = self.comb_nCr
        df = self.df_cost.T

        # find best combination in each element number
        for i in range(len(nCr)-1):
            dfx = df.iloc[nCr[i]:nCr[i+1]]
            mse.append(dfx.loc[:, "mse"].min())
            mse_label.append(dfx.loc[:, "mse"].idxmin())
            r.append(dfx.loc[:, "r"].max())
            r_label.append(dfx.loc[:, "r"].idxmax())
            homo.append(dfx.loc[:, "homo"].min())
            homo_label.append(dfx.loc[:, "aic"].idxmin())
            aic.append(dfx.loc[:, "aic"].min())
            aic_label.append(dfx.loc[:, "aic"].idxmin())

        # which element number is appropriate in each cost?
        # basically number increased, performace also increase. Need converged point
        result = []
        # MSE
        mse_best_formatID = changepoint(np.array(mse))
        result.append(mse_label[mse_best_formatID])
        result.append(mse[mse_best_formatID])
        # R
        r_best_formatID = changepoint(np.array(r))
        result.append(r_label[r_best_formatID])
        result.append(r[r_best_formatID])
        # Homo
        homo_best_formatID = int(np.array(homo[0:3]).argmin())
        result.append(homo_label[homo_best_formatID])
        result.append(homo[homo_best_formatID])
        # AIC
        aic_best_formatID = int(np.array(aic).argmin())
        result.append(aic_label[aic_best_formatID])
        result.append(aic[aic_best_formatID])
        # 1 element limitation
        result.append(mse_label[0])
        result.append(mse[0])

        self.df_bestCosts[self.col_ex0] = result

        result_mse = [
            mse[mse_best_formatID],
            df.loc[r_label[r_best_formatID], "mse"],
            df.loc[homo_label[homo_best_formatID], "mse"],
            df.loc[aic_label[aic_best_formatID], "mse"],
            df.loc[mse_label[0], "mse"]
        ]
        self.df_mse_bestCosts[self.col_ex0] = result_mse

    def _decision(self):
        """ finally decide which combination is best in this dataset
        Returns: None

        >> res_key, res_ind, res_formula
        """
        dfx = self.df_bestCosts.loc[:, self.col_ex0].T
        r = dfx["r_c"]
        numElement_mse, numElement_homo = len(dfx["mse"]), len(dfx["homo"])
        if r < 0.6:
            self.res_key.append(dfx["1d"])
            self.res_ind.append(self._key2ind(dfx["1d"]))
            self.res_formula.append(self._key2smf(dfx["1d"]))
            self._print(f"[Result/{self.col_ex0}]:r is {r} >> 1d")
        else:
            mse_homo = self.df_mse_bestCosts.loc["homo", self.col_ex0]
            mse_mse = self.df_mse_bestCosts.loc["mse", self.col_ex0]
            if mse_homo/mse_mse > self.th_homoAndMse:
                self.res_key.append(dfx["mse"])
                self.res_ind.append(self._key2ind(dfx["mse"]))
                self.res_formula.append(self._key2smf(dfx["mse"]))
                self._print(f"[Result/{self.col_ex0}]:homo/mse is {mse_homo/mse_mse} >> mse")
            else:
                if numElement_mse >= numElement_homo:
                    self.res_key.append(dfx["homo"])
                    self.res_ind.append(self._key2ind(dfx["homo"]))
                    self.res_formula.append(self._key2smf(dfx["homo"]))
                    self._print(f"[Result/{self.col_ex0}]:mse,homo:{numElement_mse},{numElement_homo} >> mse")
                else:
                    self.res_key.append(dfx["mse"])
                    self.res_ind.append(self._key2ind(dfx["mse"]))
                    self.res_formula.append(self._key2smf(dfx["mse"]))
                    self._print(f"[Result/{self.col_ex0}]:mse,homo:{numElement_mse},{numElement_homo} >> homo")

    def _publish(self):
        """ publish each transformed variables in a dataframe including target columns
        Returns: None

        Mods:df_output
        """
        out = self.df_output  # reference passing
        out[self.col_target] = self.df[self.col_target].values
        for i in range(len(self.res_key)):
            colx = self.col_ex[i]
            x = self.df[colx].values
            if "1" in self.res_key[i]: out[f"{colx}"] = x
            if "2" in self.res_key[i]: out[f"{colx}2"] = np.square(x)
            if "3" in self.res_key[i]: out[f"{colx}3"] = np.power(x, 3)
            if "4" in self.res_key[i]: out[f"log{colx}"] = np.log(x)
            if "5" in self.res_key[i]: out[f"1/{colx}"] = np.reciprocal(x)
            if "6" in self.res_key[i]: out[f"√{colx}"] = np.sqrt(x)
            if "7" in self.res_key[i]: out[f"{colx}_log{colx}"] = x*np.log(x)
            if "8" in self.res_key[i]: out[f"{colx}2_log{colx}"] = np.square(x)*np.log(x)
            if "9" in self.res_key[i]: out[f"log{colx}/{colx}"] = np.log(x)/x
            if "a" in self.res_key[i]: out[f"1/{colx}2"] = np.reciprocal(np.square(x))
            if "b" in self.res_key[i]: out[f"{colx}√{colx}"] = x*np.sqrt(x)
            if "c" in self.res_key[i]: out[f"√log{colx}"] = np.sqrt(np.log(x))
            if "d" in self.res_key[i]: out[f"√{colx}log{colx}"] = np.sqrt(x*np.log(x))

    def _combine(self):  # optional funcitons
        """ df _publish generate is all element is isolated. This combine each variables.
        Returns:None

        Mods:df_output
        """
        self._print("[Info]Combine is triggered")

        df_combine = pd.DataFrame()
        df_combine[self.col_target] = self.df[self.col_target].values
        col_search = 1  # target columns is stored, it have to be skipped
        for i_col in range(len(self.res_key)):
            colx = self.col_ex[i_col]
            key = self._key2smf(self.res_key[i_col])
            res = smf.ols(key, data=self.df).fit()
            y = res.params[0]

            for i_para in range(len(res.params)-1):
                y = y + res.params[i_para]*self.df_output.iloc[:, col_search]
                col_search += 1
            df_combine[colx] = y

        self.df_output = df_combine.copy()

    def fit(self, df, col_target):
        """
        Parameters
        ----------
        df : dataframe
        col_target : dependent variable
        """
        self._clear()
        self.col_target = col_target
        self.col_ex = list(df.columns)
        self.col_ex.remove(col_target)
        self.df = df

        # value around 0 can't be into exponetial and log, so shift themf
        self.df = self._slidingFilter(self.df)

        ################################################
        # calculate each ex value
        self.df_mse_bestCosts = pd.DataFrame(
            index=["mse", "r", "homo", "aic", "1d"], columns=self.col_ex
        )

        self.df_bestCosts = pd.DataFrame(index=[
            "mse", "mse_c", "r", "r_c",
            "homo", "homo_c", "aic", "aic_c",
            "1d", "1d_c"
        ])

        for i_col in range(len(self.col_ex)):
            self.col_ex0 = self.col_ex[i_col]

            # calculate cost in all variables combination
            self.df_cost = self._cost()

            # mse of best combination in each cost
            # >> self.df_bestCosts, self.df_mse_bestCosts
            self._evaluateOnLayer()

            # decide best combination
            self._decision()

            self.ls_df_cost.append(self.df_cost.copy())

        # create df_output
        self._publish()

    def predict(self):
        if self.onCombine: self._combine()
        return self.df_output
