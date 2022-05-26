# import libraries
import pandas as pd
from dateutil.relativedelta import relativedelta
import itertools
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def get_hierarchy_labels(data, grp1, grp2, sep='_', agg_type='hierarchy'):
    """
    Get the labels of each category and subcategory (including the total).
    Called in the first line of `get_hierarchal()`
    ----------
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
    ----------
    Returns
    ----------
    new_ : new column name
    btm : new bottom layer (list)
    labs : all column labels (list)
    """
    # define new column name
    # new column is a concatenation of `grp1` and `grp2` values
    new_ = f'{grp1}_{grp2}'
    data[new_] = data.apply(lambda x: f'{x[grp1]}{sep}{x[grp2]}', axis=1)
    # get lists of group values (including those from `new_`)
    g1s = data[grp1].unique().tolist()
    g2s = data[grp2].unique().tolist()
    ngs = data[new_].unique().tolist()
    # define bottom level
    btm = {k: [v for v in ngs if k == v.split(sep)[0]] for k in g1s}
    btm = list(itertools.chain.from_iterable(btm.values()))
    # define labels
    # if `agg_type` is hierarchy, exclude `g2s` (redundant)
    if agg_type == 'hierarchy':
        labs = ['total'] + g1s + btm
    elif agg_type == 'grouped':
        labs = ['total'] + g2s + g1s + btm
    else:
        raise ValueError('agg_type must be "hierarchy" or "grouped"')
    return new_, btm, labs


def get_hierarchal(data, grp1, grp2, date_col='date', val='value', sep='_', agg_type='hierarchy'):
    """
    ----------
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
    ----------
    Returns
    ----------
    hd : new dataframe with each distinct hierarchy as an individual column, and the dates as the index
    btm : list - bottom layer
    labs : list - all column labels
    """
    # get new column values, bottom-level labels, and full label list 
    new_, btm, labs = get_hierarchy_labels(data, grp1, grp2, sep=sep, agg_type=agg_type)
    # define hierarchal dataframe by joining grouped data
    # starting data - `new_` values as column names, `date_col` as index
    # join with data grouped by date and `grp2` (columns are `grp2` values) -> if grouped `agg_type`
    # then with data grouped by date and `grp1` (columns are `grp1` values)
    # finally, join with data grouped by date (new column is the total)
    # reorder columns to match `labs` order
    if agg_type == 'hierarchy':
        hd = data.pivot(index=date_col, columns=new_, values=val)\
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
    elif agg_type == 'grouped':
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
    else:
        raise ValueError('agg_type must be "hierarchy" or "grouped"')
    

def map_hierarchies(col, sep='_', agg_type='hierarchy'):
    """
    ----------
    Parameters
    ----------
    col : (zero filled) pandas series (name matches column of the dataframe returned by `get_hierarchal()`)
    sep : a character that is NOT included in any column names (will be used as a 
          separater for the new column)
    agg_type : either 'hierarchy' or 'grouped'. See Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/hts.html#fig:HierTree
            grouped - https://otexts.com/fpp3/hts.html#grouped-time-series
    ----------
    Returns
    ----------
    col : column updated with the value 1 where the category is True, or it falls under a higher category. 
          see Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/reconciliation.html#matrix-notation
            grouped - https://otexts.com/fpp3/hts.html#fig:GroupTree
    """
    # determine whether to map the both groups, or just group 1 (based on `agg_type`)
    # set values where the group maps to the column name to 1
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
    Create sum matrix 
    (see Hyndman's  "Forecasting: Principles and Practice" https://otexts.com/fpp3/hts.html#fig:HierTree)
    ----------
    Parameters
    ----------
    btm : list - bottom layer
    labs : list - all column labels
    sep : a character that is NOT included in any column names (will be used as a 
          separater for the new column)
    agg_type : either 'hierarchy' or 'grouped'. See Hyndman's  "Forecasting: Principles and Practice":
            hierarchy - https://otexts.com/fpp3/hts.html#fig:HierTree
            grouped - https://otexts.com/fpp3/hts.html#grouped-time-series
    ----------
    Returns
    ----------
    sum matrix - numpy array 
    """
    # stack array of ones on top (for the total)
    # matrix output from `map_hierarchies()` in the middle
    # identity matrix of bottom values at the bottom
    return np.vstack((
            np.ones(len(btm)), 
            pd.DataFrame(index=labs[1:len(labs)-len(btm)], columns=btm, data=0)\
                .apply(lambda x: map_hierarchies(x, sep=sep, agg_type=agg_type)).values, 
            np.identity(len(btm))
        ))
    

def hier_arima(col, forecast_idx, order=(1,1,0), steps_out=1):
    """
    ----------
    Parameters
    ----------
    col : column from the pandas dataframe output from `get_hierarchal()`
    forecast_idx : datetime index of forecast period
    order : order of ARIMA forecasting model
    steps_out : number of periods in forecast horizon
    ----------
    Returns
    ----------
    out : dictionary with yhat, training data (col), and the fitted model
    """
    # ARIMA forecast; make data stationary by taking the difference
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
    """
    ----------
    Parameters
    ----------
    hdf : pandas dataframe - output from `get_hierarchal()`
    order : order of ARIMA forecasting model
    steps_out : number of periods in forecast horizon
    period : time period (default months)
    ----------
    Returns
    ----------
    mods : dictionary containing fitted models and metadata
    """
    # apply ARIMA to each column n steps out
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
    """
    Get yhat matrix
    See Hyndman's "Forecasting: Principles and Practice": https://otexts.com/fpp3/reconciliation.html#eq:MinT
    ----------
    Parameters
    ----------
    mods : output from `get_models()`
    ----------
    Returns
    ----------
    out : numpy array - yhat matrix used in reconciliation equation (with sum matrix)
    """
    # Get forecasts as numpy array
    labs = list(mods['columns'].keys())
    for i in range(0, len(labs)):
        if i == 0:
            out = mods['columns'][labs[i]]['yhat'][:, np.newaxis]
        else:
            out = np.concatenate((out, mods['columns'][labs[i]]['yhat'][:, np.newaxis]), axis=1)
    return out


def reconcile(yh, s_matrix, method='ols'):
    """
    For a full explanation, see Hyndman's "Forecasting: Principles and Practice": https://otexts.com/fpp3/reconciliation.html
    ----------
    Parameters
    ----------
    yh : numpy array - yhat matrix; This should be of shape (steps out, hdf column count)
    s_matrix : numpy array - sum matrix
    method : reconciliation method (default is OLS)
    ----------
    Returns
    ----------
    rec : numpy array - reconciled forecasts
    """
    # Reconcile forecasts according to specified method
    if method.lower() == 'ols':
        ols = np.dot(np.dot(s_matrix, np.linalg.inv(np.dot(np.transpose(s_matrix), s_matrix))), np.transpose(s_matrix))
        rec = np.array([np.dot(ols, np.transpose(yh[x, :])) for x in range(yh.shape[0])])
        return rec
    else:
        raise ValueError('Invalid method')


def predict_hier(hdf, yh, rec):
    """
    ----------
    Parameters
    ----------
    hdf : pandas dataframe - output from `get_hierarchal()`
    yh : numpy array - yhat matrix
    rec : numpy array - reconciled forecast matrix
    ----------
    Returns
    ----------
    hdf_yhat : hdf with appended (original) forecasts
    hdf_rec : hdf with appended reconciled forecasts
    """
    # 
    hdf_yhat = hdf.copy()
    hdf_rec = hdf.copy()
    return hdf_yhat, hdf_rec