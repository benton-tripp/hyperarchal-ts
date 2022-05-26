# Import libraries
import pandas as pd
from dateutil.relativedelta import relativedelta
import itertools
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def get_hierarchy_labels(data, grp1, grp2, sep='_', agg_type='hierarchy'):
    """
    Get the labels of each category and subcategory (including the total).
    Called in the first line of `get_hierarchal()`
    Parameters
    ----------
    data : pandas dataframe with a date column, two hierarchies, and a value
    grp1 : the first (topmost) hierarchal group
    gp2 : the second (bottom) hierarchal group
    sep : a character that is NOT included in any column names (will be used as a 
          separater for the new column)
    agg_type : either 'hierarchy' or 'grouped'. See Hyndman's  "Forecasting: Principles and Practice":
        hierarchy - https://otexts.com/fpp3/hts.html#fig:HierTree
        grouped - https://otexts.com/fpp3/hts.html#grouped-time-series

    Returns
    ----------
    new_ : new column name
    btm : new bottom layer (list)
    labs : all column labels (list)
    """
    new_ = f'{grp1}_{grp2}'
    data[new_] = data.apply(lambda x: f'{x[grp1]}{sep}{x[grp2]}', axis=1)
    g1s = data[grp1].unique().tolist()
    g2s = data[grp2].unique().tolist()
    ngs = data[new_].unique().tolist()
    btm = {k: [v for v in ngs if k == v.split(sep)[0]] for k in g1s}
    btm = list(itertools.chain.from_iterable(btm.values()))
    if agg_type == 'hierarchy':
        labs = ['total'] + g1s + btm
    elif agg_type == 'grouped':
        labs = ['total'] + g2s + g1s + btm
    else:
        raise ValueError('agg_type must be "hierarchy" or "grouped"')
    return new_, btm, labs



def get_hierarchal(data, grp1, grp2, date_col='date', val='value', sep='_', agg_type='hierarchy'):
    """
    Parameters
    ----------
    data : pandas dataframe with a date column, two hierarchies, and a value
    grp1 : the first (topmost) hierarchal group
    gp2 : the second (bottom) hierarchal group
    date_col : date column
    val : value column
    sep : a character that is NOT included in any column names (will be used as a 
          separater for the new column)
    agg_type : either 'hierarchy' or 'grouped'. See Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/hts.html#fig:HierTree
            grouped - https://otexts.com/fpp3/hts.html#grouped-time-series

    Returns
    ----------
    hd : new dataframe with each distinct hierarchy as an individual column, and the dates as the index
    btm : bottom layer (list)
    labs : all column labels (list)
    """
    new_, btm, labs = get_hierarchy_labels(data, grp1, grp2, sep=sep, agg_type=agg_type)
    hd = data.pivot(index=date_col, columns=new_, values=val)\
        .join(
            data.groupby([date_col, grp2], as_index=False, observed=True)\
                .agg({val : lambda x: data.loc[x.index][val].sum()})\
                    .pivot(index=date_col, columns=grp2, values=val)
            )\
            .join(
                data.groupby([date_col, grp1], as_index=False, observed=True)\
                    .agg({val : lambda x: data.loc[x.index][val].sum()})\
                        .pivot(index=date_col, columns=grp1, values=val)
                )\
                    .join(
                        data.groupby(date_col, observed=True)\
                            .agg({val : lambda x: data.loc[x.index][val].sum()})\
                                .rename(columns={val:'total'})
                        )[labs]
    hd.index = pd.DatetimeIndex(hd.index, freq=hd.index.inferred_freq)
    return hd, btm, labs

def map_hierarchies(col, sep='_', agg_type='hierarchy'):
    """
    Parameters
    ----------
    col : (zero filled) pandas series (name matches column of the dataframe returned by `get_hierarchal()`)
    sep : a character that is NOT included in any column names (will be used as a 
          separater for the new column)
    agg_type : either 'hierarchy' or 'grouped'. See Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/hts.html#fig:HierTree
            grouped - https://otexts.com/fpp3/hts.html#grouped-time-series

    Returns
    ----------
    col : column updated with the value 1 where the category is True, or it falls under a higher category. 
          see Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/reconciliation.html#matrix-notation
            grouped - https://otexts.com/fpp3/hts.html#fig:GroupTree
    """
    if agg_type == 'hierarchy':
        col.loc[col.name.split(sep)[0]] = 1
        return col
    elif agg_type == 'grouped':
        col.loc[col.name.split(sep)] = 1
        return col
    else:
        raise ValueError('agg_type must be "hierarchy" or "grouped"')

def get_S(btm, labs, sep='_', agg_type='hierarchy'):
    """
    Parameters
    ----------
    btm : bottom layer (list)
    labs : all column labels (list)
    sep : a character that is NOT included in any column names (will be used as a 
          separater for the new column)
    agg_type : either 'hierarchy' or 'grouped'. See Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/hts.html#fig:HierTree
            grouped - https://otexts.com/fpp3/hts.html#grouped-time-series

    Returns
    ----------
    sum matrix : see Hyndman's  "Forecasting: Principles and Practice" https://otexts.com/fpp3/hts.html#fig:HierTree
    """
    return np.vstack((
            np.ones(len(btm)), 
            pd.DataFrame(index=labs[1:len(labs)-len(btm)], columns=btm, data=0)\
                .apply(lambda x: map_hierarchies(x, sep=sep, agg_type=agg_type)).values, 
            np.identity(len(btm))
        ))
    

def hier_arima(col, forecast_idx, order=(1,1,0), steps_out=1):
    mod = ARIMA(col.diff(), order=order, enforce_stationarity=False)
    mod = mod.fit(method_kwargs={'warn_convergence': False})
    if steps_out == 1:
        yhat = mod.predict(forecast_idx).values + col.iloc[-1]
    else:
        yhat = mod.predict(start=forecast_idx[0], end=forecast_idx[-1]).values 
        yhat[0] += col.iloc[-1]
        yhat = np.cumsum(yhat)
    out = {col.name:{'yhat':yhat, 'training_df':col, 'model':mod}}
    return out

def get_models(hdf, order=(1,1,0), steps_out=1, period='months'):
    if period.lower() in ['month', 'months']:
        if steps_out == 1:
            forecast_idx = hdf.index[-1] + relativedelta(months=1)
        else:
            forecast_idx = pd.date_range(
                hdf.index[-1] + relativedelta(months=1), 
                hdf.index[-1] + relativedelta(months=steps_out), 
                freq=hdf.index.freq)
        cols_dict = dict()
        mods = {'steps_out':steps_out, 'period':period, 'index':forecast_idx}
        hdf.apply(lambda x: cols_dict.update(hier_arima(col=x, forecast_idx=forecast_idx, order=order, steps_out=steps_out)))
        mods.update({'columns':cols_dict})
        return mods
    else:
        raise ValueError('Invalid period')

def get_forecast_matrix(mods):
    labs = list(mods['columns'].keys())
    for i in range(0, len(labs)):
        if i == 0:
            out = mods['columns'][labs[i]]['yhat'][:, np.newaxis]
        else:
            out = np.concatenate((out, mods['columns'][labs[i]]['yhat'][:, np.newaxis]), axis=1)
    return out

def reconcile(yh, s_matrix, method='ols'):
    if method.lower() == 'ols':
        ols = np.dot(np.dot(s_matrix, np.linalg.inv(np.dot(np.transpose(s_matrix), s_matrix))), np.transpose(s_matrix))
        rec = np.array([np.dot(ols, np.transpose(yh[x, :])) for x in range(yh.shape[0])])
        return rec
    else:
        raise ValueError('Invalid method')