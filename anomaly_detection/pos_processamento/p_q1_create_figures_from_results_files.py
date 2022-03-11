# FIX2022forpubrep: ready for publication
from anomaly_detection.general.utils_da import get_specific_params, prettyfy_dataframe
import os
import pandas as pd

from anomaly_detection.general.utils_da import filter_dataframes,create_dir_if_doesnt_exist
from anomaly_detection.general.global_variables import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Set general settings
def g_avaliation_metrics_versus_params_generic(df, output_path, x_param, y_param, facet_col, labels,
                                       param_to_be_color=None, iplot=False,
                                       print_figure=True, type_fig="box", points=None, title=None,
                                       facet_with_independent_xaxes=False,
                                       facet_with_repeated_labels_axes=True,
                                       facet_col_wrap=None,range_y=[0, 1.01],bar_mode="group"):
    # Figure type: Boxplot and points
    # avaliation_metrics: column name of df which will be used as Y axis
    # facet_cols : column name of df which will be used as X axis
    # param_to_be_color : column name of df which will be used as colors
    from anomaly_detection.general.utils_da import get_abreviation_of, get_pretty_names_for_params
    import plotly.express as px
    import plotly.offline as py
    import numpy as np
    import math
    py.init_notebook_mode(connected=True)
    height_auto=None
    height=None

    dissertation=True
    if(dissertation==True):
        height1 = 400

    if (facet_col_wrap is not None):
        nr_sub_figures = len(df[facet_col].unique())
        if(facet_col_wrap>4):
            height_auto = math.ceil(nr_sub_figures / facet_col_wrap) * 330

    if(facet_col is not None):
        nr_sub_figures = len(df[facet_col].unique())
        facet_col_wrap1 = 2
        facet_col_wrap2 = 1
        height1 = math.ceil(nr_sub_figures / 2) * 625
        height2 = nr_sub_figures * 625

    else:
        facet_col_wrap2 = None
        facet_col_wrap1 = None
        height2 = None

    prettyn_y_param=get_pretty_names_for_params(y_param, language='PT')
    prettyn_x_param = get_pretty_names_for_params(x_param, language='PT')
    prettyn_facet_col = get_pretty_names_for_params(facet_col, language='PT')
    prettyn_param_to_be_color = get_pretty_names_for_params(param_to_be_color, language='PT')

    labels = {x_param: prettyn_x_param if x_param is not None else None,
              y_param: prettyn_y_param if y_param is not None else None,
              facet_col: prettyn_facet_col if facet_col is not None else None,
              param_to_be_color: prettyn_param_to_be_color if param_to_be_color is not None else None}

    general_params = dict(data_frame=df,
                          y=y_param,
                          labels=labels,
                          range_y=range_y,
                          facet_col=facet_col,
                          color=param_to_be_color,
                          )
    if x_param is not None:
        general_params["x"] = x_param

    if (type_fig == "box"): #this is the most general.
        prettyn_param_to_be_color=get_pretty_names_for_params(param_to_be_color)
        specific_params=dict(
                        points=points,
                        title=(prettyn_y_param + " versus parameterizations"),
                        category_orders={x_param: list(np.sort(df[x_param].unique())) if x_param is not None else None,
                                      facet_col: list(np.sort(df[facet_col].unique())) if facet_col is not None else None},
                        facet_col_wrap=facet_col_wrap2 if facet_col_wrap is None else facet_col_wrap,
                        height=height2 if height_auto is None else height_auto,
                        boxmode="group"  #several bloxplots for each x axis
                        )

        # combine both dicts and update values if there are repeated keys
        dict_params = {**general_params , **specific_params}
        fig = px.box(**dict_params)

    if (type_fig == "scatter"):
        specific_params = dict(
                         title=(prettyn_y_param + " versus parameterizations") if title is None else title,
                         #category_orders={x_param: list(np.sort(df[x_param].unique())),
                         #                 facet_col: list(np.sort(df[facet_col].unique()))},
                         facet_col_wrap=facet_col_wrap1,
                         height=height1,
                         #boxmode="overlay"
                         )

        # combine both dicts and update values if there are repeated keys
        dict_params = {**general_params , **specific_params}
        fig = px.scatter(**dict_params)

    if (type_fig == "line"):
        specific_params = dict(
                      title=(prettyn_y_param + " versus logs, iterationcv andscaling factors") if title is None else title,
                      facet_col_wrap=facet_col_wrap,
                      facet_row_spacing = 0.2
                      )

        # combine both dicts and update values if there are repeated keys
        dict_params = {**general_params , **specific_params}
        fig = px.line(**dict_params)

    if (type_fig == "bar"):
        specific_params = dict(
                      title=(prettyn_y_param + " versus logs, iterationcv andscaling factors") if title is None else title,
                        barmode=bar_mode
                      )

        # combine both dicts and update values if there are repeated keys
        dict_params = {**general_params , **specific_params}
        fig = px.bar(**dict_params)

    fig.update_xaxes(type='category')

    if facet_with_independent_xaxes:
        fig.update_xaxes(matches=None)

    if facet_with_repeated_labels_axes:
        for i in range(len(fig["data"])+1):
            xaxis_name = 'xaxis' if i == 0 else f'xaxis{i + 1}'
            try:
                fig.layout[xaxis_name].showticklabels = True
                fig.layout[xaxis_name].title.text = prettyn_x_param
            except:
                continue

    if(type_fig=='line'):
        fig.update_traces(mode='lines+markers')

    # Exibindo figura/gráfico
    if (iplot == True):
        py.iplot(fig)
    else:
        py.plot(fig)
    #py.plot(fig)
    if (print_figure == True):
        fig.write_image(os.path.join(output_path / ("fig11_X%s_Y%s_L%s_.jpeg" % (get_abreviation_of(facet_col), y_param, param_to_be_color))))

def g_avaliation_metrics_versus_params(df, output_path, x_param, y_param, facet_col, labels,
                                       param_to_be_color=None, iplot=False,
                                       print_figure=True, type_fig="box", points=None, title=None,
                                       facet_with_independent_xaxes=False,
                                       facet_with_repeated_labels_axes=True,
                                       facet_col_wrap=None,range_y=[0, 1.01],bar_mode="group",facet_col_wrap1=None,repeated_axes=["x","y"]):


    # Figure type: Boxplot and points
    # avaliation_metrics: column name of df which will be used as Y axis
    # facet_cols : column name of df which will be used as X axis
    # param_to_be_color : column name of df which will be used as colors
    from anomaly_detection.general.utils_da import get_abreviation_of, get_pretty_names_for_params
    import plotly.express as px
    import plotly.offline as py
    import numpy as np
    import math
    py.init_notebook_mode(connected=True)
    height_auto=None
    height1 = None

    dissertation=True
    if(dissertation==True):
        facet_col_wrap1 = 5
        repeated_axes=["x"]
        height_auto=400

    if (facet_col_wrap is not None):
        nr_sub_figures = len(df[facet_col].unique())

        if(facet_col_wrap>4):
            height_auto = math.ceil(nr_sub_figures / facet_col_wrap) * 330

    if(facet_col is not None):
        nr_sub_figures = len(df[facet_col].unique())
        facet_col_wrap1 = 2 if facet_col_wrap1 is None else facet_col_wrap1
        facet_col_wrap2 = 1

        if facet_col_wrap1 <=2 :
            height1 = math.ceil(nr_sub_figures / facet_col_wrap1) * 625
            height2 = nr_sub_figures * 625
        else:
            height1 =math.ceil(nr_sub_figures / facet_col_wrap1) * 500
            height2 = nr_sub_figures * 625

    else:
        facet_col_wrap2 = None
        facet_col_wrap1 = None
        #height1 = None
        height2 = None

    prettyn_y_param=get_pretty_names_for_params(y_param, language='PT')
    prettyn_x_param = get_pretty_names_for_params(x_param, language='PT')
    prettyn_facet_col = get_pretty_names_for_params(facet_col, language='PT')
    prettyn_param_to_be_color = get_pretty_names_for_params(param_to_be_color, language='PT')

    labels = {x_param: prettyn_x_param if x_param is not None else None,
              y_param: prettyn_y_param if y_param is not None else None,
              facet_col: prettyn_facet_col if facet_col is not None else None,
              param_to_be_color: prettyn_param_to_be_color if param_to_be_color is not None else None}

    general_params = dict(data_frame=df,
                          y=y_param,
                          labels=labels,
                          range_y=range_y,
                          facet_col=facet_col,
                          color=param_to_be_color,
                          )
    if x_param is not None:
        general_params["x"] = x_param

    if param_to_be_color is None:
        if(type_fig=="box") :
            fig = px.box(data_frame=df,
                         x=x_param,
                         y=y_param,
                         points="all",
                         labels={x_param:prettyn_x_param,
                                 y_param:prettyn_y_param,
                                 facet_col:prettyn_facet_col},
                         title=(prettyn_y_param + " versus parameterizations"),
                         range_y=[0, 1.01],
                         category_orders={x_param: list(np.sort(df[x_param].unique())) if x_param is not None else None,
                                          facet_col: list(np.sort(df[facet_col].unique())) if facet_col is not None else None},
                         facet_col_wrap=facet_col_wrap1,
                         facet_col=facet_col,
                         height=height1,
                         boxmode="overlay"
                         )
        if (type_fig == "scatter"):
            fig = px.scatter(data_frame=df,
                             x=x_param,
                             y=y_param,
                             labels={x_param: prettyn_x_param,
                                     y_param: prettyn_y_param,
                                     facet_col: prettyn_facet_col},
                             title=(prettyn_y_param + " versus parameterizations") if title is None else title,
                             range_y=[0, 1.01],
                             #category_orders={x_param: list(np.sort(df[x_param].unique())),
                             #                 facet_col: list(np.sort(df[facet_col].unique()))},
                             facet_col=facet_col,
                             facet_col_spacing=0.09,
                             facet_row_spacing=0.09 if dissertation==True else None,
                            facet_col_wrap=facet_col_wrap1,
                             height=height1,
                             #boxmode="overlay"
                             )


    else:  # graphic detailing logs name

        if (type_fig == "box"): #this is the most general. Try print here
            prettyn_param_to_be_color=get_pretty_names_for_params(param_to_be_color)
            specific_params=dict(
                            points=points,
                            title=(prettyn_y_param + " versus parameterizations"),
                            category_orders={x_param: list(np.sort(df[x_param].unique())) if x_param is not None else None,
                                          facet_col: list(np.sort(df[facet_col].unique())) if facet_col is not None else None},
                            facet_col_wrap=facet_col_wrap2 if facet_col_wrap is None else facet_col_wrap,
                            height=height2 if height_auto is None else height_auto,
                            boxmode="group"  #several bloxplots for each x axis
                            )

            # combine both dicts and update values if there are repeated keys
            dict_params = {**general_params , **specific_params}
            fig = px.box(**dict_params)

        if (type_fig == "scatter"):
            specific_params = dict(
                             title=(prettyn_y_param + " versus parameterizations") if title is None else title,
                             #category_orders={x_param: list(np.sort(df[x_param].unique())),
                             #                 facet_col: list(np.sort(df[facet_col].unique()))},
                             facet_col_wrap=facet_col_wrap1,
                             height=height1,
                             #boxmode="overlay"
                             )

            # combine both dicts and update values if there are repeated keys
            dict_params = {**general_params , **specific_params}
            fig = px.scatter(**dict_params)

        if (type_fig == "line"):
            specific_params = dict(
                          title=(prettyn_y_param + " versus logs, iterationcv andscaling factors") if title is None else title,
                          facet_col_wrap=facet_col_wrap,
                          facet_row_spacing = 0.2
                          )

            # combine both dicts and update values if there are repeated keys
            dict_params = {**general_params , **specific_params}
            fig = px.line(**dict_params)

    if (type_fig == "bar"):
        specific_params = dict(
                      title=(prettyn_y_param + " versus logs, iterationcv andscaling factors") if title is None else title,
                      #facet_col_wrap=facet_col_wrap,
                      #facet_row_spacing = 0.2
                        barmode=bar_mode
                      )

        # combine both dicts and update values if there are repeated keys
        dict_params = {**general_params , **specific_params}
        fig = px.bar(**dict_params)

    fig.update_xaxes(type='category')

    if facet_with_independent_xaxes:
        fig.update_xaxes(matches=None)

    if facet_with_repeated_labels_axes:
        for axis,prettyn_param in zip(repeated_axes,[prettyn_x_param,prettyn_y_param]):
            for i in range(len(fig["data"])+1):
                axis_name = axis+'axis' if i == 0 else f'{axis}axis{i + 1}'
                try:
                    fig.layout[axis_name].showticklabels = True
                    fig.layout[axis_name].title.text = prettyn_param
                except:
                    continue

    if(type_fig=='line'):
        fig.update_traces(mode='lines+markers')

    # Showing figure
    if (iplot == True):
        py.iplot(fig)
    else:
        py.plot(fig)
    #py.plot(fig)
    if (print_figure == True):
        fig.write_image(os.path.join(output_path / ("fig11_X%s_Y%s_L%s_.jpeg" % (get_abreviation_of(facet_col), y_param, param_to_be_color))))

def g_avaliation_metrics_versus_logs(df, output_path, x_param, y_param, labels,
                                     param_to_be_color=None, iplot=False,
                                     print_figure=True):
    # Figure type: Boxplot and points.
    # Description: similar than g_avaliation_metrics_versus_params but here we have boxplots by each log
    # avaliation_metrics: column name of df which will be used as Y axis
    # facet_cols : column name of df which will be used as X axis
    # param_to_be_color : column name of df which will be used as colors
    from anomaly_detection.general.utils_da import get_abreviation_of, get_pretty_names_for_params
    import plotly.express as px
    import plotly.offline as py
    import numpy as np
    py.init_notebook_mode(connected=True)

    prettyn_y_param=get_pretty_names_for_params(y_param, language='PT')
    prettyn_x_param = get_pretty_names_for_params(x_param, language='PT')
    prettyn_param_to_be_color = get_pretty_names_for_params(param_to_be_color, language='PT')

    fig = px.box(data_frame=df,
                 x=x_param,
                 y=y_param,
                 points="all",
                 labels={x_param:prettyn_x_param,
                         y_param:prettyn_y_param,
                         param_to_be_color:prettyn_param_to_be_color if param_to_be_color is not None else None},
                 title=(prettyn_y_param + " versus logs"),
                 range_y=[0, 1.01],
                 category_orders= {x_param: list(np.sort(df[x_param].unique()))} if x_param !=None else {},
                 boxmode="overlay",
                 color = param_to_be_color,
                 )

    # Exibindo figura/gráfico
    if (iplot == True):
        py.iplot(fig)
    else:
        py.plot(fig)
    #py.plot(fig)
    if (print_figure == True):
        fig.write_image(os.path.join(output_path, ("fig11_X%s_Y%s.jpg") % (get_abreviation_of(x_param), y_param)))


def create_figure(figure_number, df_pretty, dfg_pretty, rd, full_name_approach,
                  params_to_focus_on_df, valor_to_focus_on_df, x_param, facet_col, params_to_be_color, params, labels,
                  experiments_group_path, y_params, iplot=False, print_figure=True):

    if figure_number == 11:

        # Iterate by attbs
        print("Fig11")
        for param_to_focus_on_df in params_to_focus_on_df:
            if param_to_focus_on_df is not None:
                print("====== Figures with df filtered using %s=%s ======"%(param_to_focus_on_df,valor_to_focus_on_df))
                df_pretty = df_pretty[df_pretty[param_to_focus_on_df] == valor_to_focus_on_df]

            for param_to_be_color in params_to_be_color:
                for y_param in y_params:
                    if (full_name_approach== 'detect_infrequents') | (full_name_approach== 'random'):
                        g_avaliation_metrics_versus_logs(df_pretty, experiments_group_path / "figures", param_to_be_color, y_param, labels,
                                                         None, iplot,
                                                         print_figure)
                    else:
                        g_avaliation_metrics_versus_params(df_pretty, experiments_group_path / "figures", x_param, y_param, facet_col, labels,
                                                           param_to_be_color, iplot,
                                                           print_figure)


    if figure_number == 13:

        # Ap versus logs
        y_param=y_params[0]

        # Todos os pontos em um boxplot
        print("Fig13.A")
        g_avaliation_metrics_versus_params(df_pretty,
                                           experiments_group_path / "figures",
                                           x_param,
                                           y_param,
                                           facet_col, labels,
                                           None, iplot,
                                           print_figure)

        x_param="log_name_abbrev"
        param_to_be_color="log_anomaly_intuition"

        print("Fig13.B")
        g_avaliation_metrics_versus_logs(df_pretty,
                                         experiments_group_path / "figures",
                                         x_param,
                                         y_param, labels,
                                         param_to_be_color, iplot,
                                         print_figure)

    if figure_number == 18:

        # Name: AP by anmaly intuition and approach
        # x: log_anomaly_intuition
        # y: AP
        # param_to_be_color: p_modelit_type ( approach)

        # All approaches
        x_params = ["log_anomaly_intuition", "log_name_abbrev"]
        x_param = x_params[1]
        facet_col = None
        #param_to_be_color = "p_modelit_type"
        param_to_be_color = None
        labels = {}
        y_param = y_params[0]
        #iplot=False

        print("Fig18A")
        g_avaliation_metrics_versus_params(df_pretty, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure, points="all")

        print("Fig18B")
        param_to_be_color = "p_modelit_type"
        g_avaliation_metrics_versus_params(df_pretty, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure, points=None)

        # Save stats related to figure
        fig_stats=pd.DataFrame()
        fig_stats["min"] = df_pretty.groupby(x_params)[y_param].min()
        fig_stats["max"] = df_pretty.groupby(x_params)[y_param].max()
        fig_stats["mean"] = df_pretty.groupby(x_params)[y_param].mean()
        fig_stats["std"] = df_pretty.groupby(x_params)[y_param].std()
        fig_stats["median"] = df_pretty.groupby(x_params)[y_param].median()
        fig_stats["count"] = df_pretty.groupby(x_params)[y_param].count()
        fig_stats=fig_stats.reset_index()
        fig_stats.to_csv(experiments_group_path / STATS_ANALYSIS_DIR_NAME/ ("stats_fig%s.csv"%figure_number), index=False,encoding="utf-8")

        print("Fig18C")
        x_params=["log_anomaly_intuition", "log_name_abbrev","p_modelit_type"]
        df_pretty2=(df_pretty.groupby(x_params)[y_param].mean().reset_index())
        x_param="log_name_abbrev"
        param_to_be_color="p_modelit_type"
        g_avaliation_metrics_versus_params_generic(df_pretty2, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure,  type_fig="scatter")

    # 3.
    if figure_number == 14:
        # Name: AP by anmaly intuition and approach
        # x: log_anomaly_intuition
        # y: AP
        # param_to_be_color: p_modelit_type ( approach)

        # All approaches
        x_param=["log_anomaly_intuition","log_name_abbrev"]
        x_param=x_param[0]
        facet_col=None
        param_to_be_color="p_modelit_type"
        labels={}
        y_param = y_params[0]

        print("Fig14")
        g_avaliation_metrics_versus_params(df_pretty, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure,points="all")
    if figure_number == 16:

        # All approaches
        x_param=["log_anomaly_intuition","log_name_abbrev"]
        x_param=x_param[1]
        facet_col="log_anomaly_intuition"
        param_to_be_color="p_modelit_type"
        labels={}
        y_param = y_params[0]

        print("Fig16.modeA")
        g_avaliation_metrics_versus_params(df_pretty, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure,points="all",facet_with_independent_xaxes=True)

        print("Fig16.modeB")
        x_param="p_modelit_type"
        x_param=None
        facet_col="log_name_abbrev"
        param_to_be_color="p_modelit_type"
        g_avaliation_metrics_versus_params(df_pretty, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure,facet_col_wrap=10,facet_with_repeated_labels_axes=False)

    # 3. print 12 figure
    if figure_number == 12:
        print("Fig12")
        y_param="exp_auc_pr"
        param_to_be_color=None
        #param_to_be_color="log_name_abbrev"
        if full_name_approach=='aalst_approach':
            param_to_be_color="p_modelit_vapproach"
        #iplot=False
        if len(params)>0:
            g_avaliation_metrics_versus_params(dfg_pretty, experiments_group_path / "figures", x_param, y_param,
                                               facet_col, labels,
                                               param_to_be_color, iplot,
                                               print_figure,type_fig="scatter")


    if figure_number == 15:
        print("Fig15")
        x_param="p_modelit_type"
        y_param = "exp_auc_pr"
        param_to_be_color = None
        facet_col="log_anomaly_intuition"
        g_avaliation_metrics_versus_params(dfg_pretty, experiments_group_path / "figures", x_param, y_param,
                                           facet_col, labels,
                                           param_to_be_color, iplot,
                                           print_figure, type_fig="scatter",title="AP by anomaly types")

        #fig.show()
        #py.plot(fig)
    if figure_number == 17:
        print("Fig17")
        x_param="scaling_factor"
        y_param="avaliation_metric_value"
        facet_col="log_name_abbrev"
        labels={}
        param_to_be_color='avaliation_metric'
        for log_anomaly_intuition in rd["log_anomaly_intuition"].unique():
            for iteracao_cv in rd["iteracao_cv"].unique():
                rd1=rd[(rd["log_anomaly_intuition"]==log_anomaly_intuition) & (rd["iteracao_cv"]==iteracao_cv)]
                title = "iteracao=%s, intuicao=%s" % (iteracao_cv,log_anomaly_intuition)
                g_avaliation_metrics_versus_params(rd1, experiments_group_path / "figures", x_param, y_param,
                                                   facet_col, labels,
                                                   param_to_be_color, iplot,
                                                   print_figure, type_fig="line",facet_col_wrap=5,title=title)

def graphics(results_data, experiments_group_path, dict_limiar_tests, statistic_analysis=False, iplot=False, print_figure=True, figure_numbers=[11],
             ):

    params_to_focus_on_df = [None]  # param to focus
    valor_to_focus_on_df = ''  # same lenth than params_to_focus_on_df
    params_to_be_color=[]
    facet_col=None
    x_param=None
    labels = {}
    rd=None
    params = []
    y_params = []
    columns_to_use_rd=[]
    df=None
    dfg=None
    df_pretty=None
    dfg_pretty=None

    # 1. Select columns
    if ( (11 in figure_numbers) | (12 in figure_numbers) | (13 in figure_numbers)):
        y_params = ["exp_auc_pr", "exp_auc_roc"]
        full_name_approach = results_data.iloc[0]['p_modelit_type']
        list_to_groupby=['p_modelit_id', 'iteracao_cv']
        columns_to_use=["exp_group_id",
                           "log_name_abbrev",
                           "exp_auc_pr",
                           "exp_auc_roc",
                           "log_anomaly_intuition"
                           ]

    if ((14 in figure_numbers) | (15 in figure_numbers) | (16 in figure_numbers)  | (18 in figure_numbers)):
        y_params = ["exp_auc_pr", "exp_auc_roc"]
        full_name_approach=None
        list_to_groupby = ['p_modelit_type','p_modelit_id', 'iteracao_cv']
        columns_to_use=["exp_group_id",
                        "log_name_abbrev",
                        "exp_auc_pr",
                        "exp_auc_roc",
                        "log_anomaly_intuition",
                        "iteracao_cv",
                        "p_modelit_type"
                           ]

    if (17 in figure_numbers):
        y_params_rd = ["precision_p", "recall_p","PP","PN"]
        columns_to_use_rd= ["exp_group_id",
                        "log_anomaly_intuition",
                        "log_name_abbrev",
                        "p_modelit_type",
                        "iteracao_cv",
                        "scaling_factor",
                           ] + y_params_rd

        full_name_approach = results_data.iloc[0]['p_modelit_type']

    #2. Prepare dataframe dfg e df
    if ('exp_auc_pr' in y_params):
        results_data_dropped = results_data.drop_duplicates(subset=list_to_groupby, keep='first')

        if (11 in figure_numbers) | (12 in figure_numbers) | (13 in figure_numbers):
            params = get_specific_params(full_name_approach)

            df = results_data_dropped[columns_to_use + params].sort_values(by=['log_name_abbrev']+params)
            df_for_stats = df.copy()
            dfg = df_for_stats.groupby(get_specific_params(full_name_approach) + ['log_name_abbrev']).mean()
            dfg[["exp_auc_pr_std","exp_auc_roc_std"]]=df_for_stats.groupby(get_specific_params(full_name_approach) + ['log_name_abbrev']).std()
            dfg=dfg.reset_index()
            dfg.rename(columns={"exp_auc_pr":"exp_auc_pr_mean","exp_auc_roc":"exp_auc_roc_mean"}).to_csv(experiments_group_path / "stats_analysis_dfg1.csv", index=False,encoding="utf-8")
        else: # phase 02?

            df = results_data_dropped[columns_to_use + params].sort_values(by=['p_modelit_type','log_name_abbrev','iteracao_cv'] + params)
            df_for_stats = df.copy()
            dfg = df_for_stats.groupby(['p_modelit_type','log_name_abbrev','log_anomaly_intuition'])[["exp_auc_pr","exp_auc_roc"]].mean()
            dfg[["exp_aux_pr_std", "exp_auc_roc_std"]] = df_for_stats.groupby(['p_modelit_type','log_name_abbrev','log_anomaly_intuition'])[["exp_auc_pr","exp_auc_roc"]].std()
            dfg = dfg.reset_index()
            full_path = experiments_group_path / STATS_ANALYSIS_DIR_NAME
            create_dir_if_doesnt_exist(full_path)
            dfg.rename(columns={"exp_auc_pr":"exp_auc_pr_mean","exp_auc_roc":"exp_auc_roc_mean"}).to_csv(full_path/"stats_analysis_dfg2.csv", index=False)

        if full_name_approach == 'autoencoder_nolle':
            # Create a new column
            df["actv_functions"] = df["funcao_f"] + "-" + df["funcao_g"]

            if 12 in figure_numbers:
                dfg["actv_functions"] = dfg["funcao_f"] + "-" + dfg["funcao_g"]

    if len(columns_to_use_rd)>0:
        rd = results_data[columns_to_use_rd]
        rd["PP_percent"] = (results_data["PP"] / (results_data["PP"] + results_data["PN"]))

        rd = rd.melt(
            id_vars=['exp_group_id', 'log_anomaly_intuition', 'log_name_abbrev', 'p_modelit_type', 'iteracao_cv',
                     'scaling_factor', 'PP', 'PN']).sort_values(by=['log_name_abbrev', 'variable', 'scaling_factor'],
                                                                ascending=True)
        rd = rd.rename(columns={'variable': 'avaliation_metric', 'value': 'avaliation_metric_value'})

    # 2. Set params for figures
    if ( 11 in figure_numbers) | (12 in figure_numbers) | (13 in figure_numbers):
        params_to_be_color = [None, "log_name_abbrev"]
        facet_col = params[0] if params!=[] else None
        x_param = params[1]  if params!=[] else None

        if full_name_approach == 'tstideplus':
            # Set params for figures
            labels = {}

        if full_name_approach == 'autoencoder_nolle':
            # Set params for figures
            x_param = "actv_functions"

        if "p_modelit_vapproach" in params:
            params_to_be_color.append("p_modelit_vapproach")
            params_to_focus_on_df.append('p_modelit_vapproach')
            valor_to_focus_on_df='original_mod'

    if df is not None:
        df_pretty = prettyfy_dataframe(df.copy()) #using AP ( It shows an AP for each sublog)
    if dfg is not None:
        dfg_pretty = prettyfy_dataframe(dfg.copy()) # Using MAP (It showa an MAP for each log)

    # 3. Print Figures
    for figure_number in figure_numbers:
        create_figure(figure_number, df_pretty, dfg_pretty, rd, full_name_approach,
                      params_to_focus_on_df, valor_to_focus_on_df, x_param, facet_col, params_to_be_color, params,
                      labels, experiments_group_path, y_params, iplot, print_figure)

    # 4. Generate statistics
    if (statistic_analysis == True) & ((15 in figure_numbers)| (12 in figure_numbers)):
        print(" ========================== Test for approaches  ========================== ")


    if (statistic_analysis==True) & (15 in figure_numbers):
        params_of_approach=[]
        for anomaly_type in dfg["log_anomaly_intuition"].unique():
            print("\n Test Analysis: for %s" % anomaly_type)
            dfg_=dfg[dfg["log_anomaly_intuition"]==anomaly_type]
            execute_statistic_analysis(dfg_,experiments_group_path,params_of_approach,full_name_approach,dict_limiar_tests,2,iplot,anomaly_type)


    if( (statistic_analysis==True) & (12 in figure_numbers)):
        params_of_approach=get_specific_params(full_name_approach)
        if(len(params_of_approach)>0):
            print("\n Test Analysis: for %s" % full_name_approach)
            execute_statistic_analysis(dfg,experiments_group_path,params_of_approach,full_name_approach,dict_limiar_tests,1,iplot)
        else:
            print("\n Test Analysis: Approach %s has not parameters to be included in test" % full_name_approach)



def execute_statistic_analysis(dfg,experiments_group_path,params_of_approach,full_name_approach,dict_limiar_tests,phase_number,iplot,anomaly_type=None):
    # 2.0 hypothesis test
    # H0:  the samples have the same distribution
    # H1: the samples have different distribution


    alpha = 0.05
    quantile=0
    limiar_exp_auc_pr=None
    if dict_limiar_tests is not None:
        if full_name_approach in dict_limiar_tests.keys():
            limiar_exp_auc_pr=dict_limiar_tests[full_name_approach] # autoencoder

    # 1. Filtering the best
    if phase_number==1:
        # dfg_best will save a dataframe containing all best parameters values
        dfg_best = dfg.groupby(params_of_approach)['exp_auc_pr'].quantile(q=quantile).reset_index()
        if (len(dfg_best) < 2):
            print("only exists one parameterization. That is the best: ")
            print(dfg_best)
            return

        dfg_best = dfg_best.loc[dfg_best['exp_auc_pr'] > limiar_exp_auc_pr, :]
        del dfg_best['exp_auc_pr']

        # We will save in dfg only best results
        cond = None

        for index, serie in dfg_best.iterrows():
            conditions = serie.to_dict()
            if (cond is None):  # If there are not any
                cond = filter_dataframes(dfg, conditions)[0]
            else:
                cond = cond | filter_dataframes(dfg, conditions)[0]

        dfg = dfg[cond]

        # Create a column parameterizations
        for param in params_of_approach: # Only of phase01
            if param == params_of_approach[0]:
                if (dfg[param].dtype == object) | (dfg[param].dtype == str):
                    dfg_params = dfg[param]
                else:
                    dfg_params = dfg[param].astype(str)
            else:
                if (dfg[param].dtype == object) | (dfg[param].dtype == str) :
                    dfg_params = dfg_params + "-" + dfg[param]
                else:
                    dfg_params = dfg_params + "-" + dfg[param].astype(str)


        dfg['params']=dfg_params

    # 2. Create a table where columns are the treatments (parameterizations) and rows are the blocks( logs)
    if phase_number==1:
        # treatments (parameterizations) and  blocks( logs)
        dfg = dfg.pivot(index="log_name_abbrev", columns="params", values="exp_auc_pr")
    if phase_number ==2:
        # treatments (approaches) and  blocks(logs)
        dfg = dfg.pivot(index="log_name_abbrev", columns="p_modelit_type", values="exp_auc_pr")

    # 3 Normality Analysis
    if phase_number==1:
        facet_col="params"
    if phase_number==2:
        facet_col="p_modelit_type"

    full_path = experiments_group_path / STATS_ANALYSIS_DIR_NAME
    create_dir_if_doesnt_exist(full_path)
    from anomaly_detection.general.utils_da import execute_all_histogram_tests_in_df, execute_all_median_tests_in_df
    df_rslt_normality = execute_all_histogram_tests_in_df(dfg, facet_col, iplot, alpha)

    # 4 Test of medians
    df_rslt_tests=execute_all_median_tests_in_df(dfg, limiar_exp_auc_pr)

    # Save

    if phase_number == 1:
        df_rslt_normality.to_csv(full_path / ("stats_analysis_ph%s_tnormality.csv" % phase_number), index=False,
                                 encoding="utf-8")
        df_rslt_tests.to_csv(full_path / ("stats_analysis_ph%s_tmedian.csv"%phase_number), index=False)
    elif phase_number==2:
        df_rslt_normality.to_csv(full_path / ("stats_analysis_ph%s_at%s_tnormality.csv" % (phase_number,anomaly_type)), index=False,
                                 encoding="utf-8")
        df_rslt_tests.to_csv(full_path / ("stats_analysis_ph%s_at%s_tmedian.csv" % (phase_number,anomaly_type)), index=False)
    print(df_rslt_tests.head())


def create_analysis_q1(group_experiments_path, results_data_filename, dict_limiar_test,statistic_analysis,iplot=False, print_figure=True,figure_numbers=[11]):
    import os
    # Open File
    results_data = pd.read_csv(os.path.join(group_experiments_path, results_data_filename),dtype={"p_modelit_thd":object},encoding="utf-8",sep=',')
    # results_data.info()

    # Create output folder
    if print_figure == True:
        output_path = os.path.join(group_experiments_path, "figures") + "\\"
        if not (os.path.isdir(output_path)):  # If folder doesnt exist
            os.makedirs(output_path)  # Creat folder
        print('folder_output: ')
        print(output_path)
        print('\n')

    # Figues for AP
    graphics(results_data, group_experiments_path, dict_limiar_test,statistic_analysis, iplot, print_figure, figure_numbers)