import rvc
import pandas as pd

#%%
df0 = pd.read_csv("data/df_g1.csv", index_col=0)
df0 = df0.iloc[:, :5]
df0.head()
#%%
RVC = rvc.RVC()
RVC.fit(df0, "EP")
#%%
