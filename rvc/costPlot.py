import numpy as np
import pandas as pd
import os
from datetime import datetime
import math

from bokeh.layouts import column, gridplot
from bokeh.models import HoverTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import figure, show, output_file, save, ColumnDataSource
from bokeh.palettes import all_palettes

def costPlot(res, **kwargs):
    # plot_width = kwargs["plot_width"] if "plot_width" in kwargs else 1500
    # plot_height = kwargs["plot_height"] if "plot_height" in kwargs else 670
    onShow = kwargs["onShow"] if "onShow" in kwargs else True
    onStore = kwargs["onStore"] if "onStore" in kwargs else True
    dir_store = kwargs["path"] if "path" in kwargs else "."
    output_name = kwargs["output_name"] if "output_name" in kwargs else "."

    if dir_store == ".":
        dir_store = f"_rvc/costPlot"
        if os.path.isdir("_rvc") == False:
            os.mkdir("_rvc")
            os.mkdir(dir_store)
        elif os.path.isdir(dir_store) == False:
            os.mkdir(dir_store)

    colors = all_palettes['Viridis'][len(res.comb_nCr)-1]
    TOOLS = "pan, wheel_zoom, box_zoom, reset"

    def createP():
        # p = figure(plot_width=plot_width, plot_height=plot_height, tools=TOOLS)
        p = figure(tools=TOOLS)
        p.sizing_mode = 'stretch_width'
        p.title.text_font_size = "20pt"
        p.xaxis.axis_label = "MSE"
        p.yaxis.axis_label = "homoscedasticity"
        p.ygrid.grid_line_alpha = 1
        p.background_fill_color = "#dddddd"
        p.background_fill_alpha = 0.5
        tips = [("(x,y)", "($x, $y)"), ("label", "@label")]
        p.add_tools(HoverTool(tooltips=tips, mode="mouse", point_policy="follow_mouse"))
        p.legend.click_policy = "mute"
        return p

    list_tab, list_ols, list_opt = [], [], []
    for i_col in range(len(res.ls_df_cost)):
        p = createP()

        dfx = res.ls_df_cost[i_col].T
        x = dfx.iloc[:, 0].values  # mse
        y = dfx.iloc[:, 2].values  # homo
        idx = dfx.index

        selected = dfx.loc[res.res_key[i_col], :].values
        p.circle(x[0], y[0], line_width=2.5, size=8, alpha=0.8, color="orange")  # if it was y ~ x
        p.circle(selected[0], selected[2], line_width=2.5, size=8, alpha=0.8, color="red")  # selected

        for i_ncr in range(len(res.comb_nCr)-1):
            x0 = x[res.comb_nCr[i_ncr]:res.comb_nCr[i_ncr+1]]
            y0 = y[res.comb_nCr[i_ncr]:res.comb_nCr[i_ncr+1]]
            idx0 = idx[res.comb_nCr[i_ncr]:res.comb_nCr[i_ncr+1]]
            source = ColumnDataSource(data=dict(x=x0, y=y0, label=idx0))
            p.circle('x', 'y', line_width=2.5, alpha=0.8, color=colors[i_ncr], source=source)

        list_ols.append([x[0], y[0]])
        list_opt.append([selected[0], selected[2]])

        tab = Panel(child=p, title=res.col_ex[i_col])
        list_tab.append(tab)

    p = createP()
    ols, opt = np.array(list_ols), np.array(list_opt)
    source_ols = ColumnDataSource(data=dict(x=ols[:, 0], y=ols[:, 1], label=res.col_ex))
    source_opt = ColumnDataSource(data=dict(x=opt[:, 0], y=opt[:, 1], label=res.col_ex))
    for i in range(len(ols)):
        xy = np.vstack([ols[i, :], opt[i, :]])
        p.line(xy[:, 0], xy[:, 1], width=2, color="black", line_dash='dotted')
    p.circle('x', 'y', line_width=2.5, alpha=0.8, color="orange", source=source_ols)
    p.circle('x', 'y', line_width=2.5, alpha=0.8, color="red", source=source_opt)

    tab = Panel(child=p, title="Compare")
    list_tab.append(tab)

    tabs = Tabs(tabs=list_tab)

    ##################################################################
    now = datetime.now().strftime('%Y_%m%d_%H%M%S')
    if output_name == ".":
        output_name = f"/{now}_{res.col_target}"

    if onShow:
        show(tabs)

    if onStore:
        output_file(f"{dir_store}/{output_name}.html", title="cost_plot")
        save(tabs)
