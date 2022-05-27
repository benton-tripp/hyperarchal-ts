# import libraries
import pandas as pd
from dateutil.relativedelta import relativedelta
import itertools
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error


#---------------------------------------------------------------------
# Data Pre-Processing
#---------------------------------------------------------------------   


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


def get_hierarchal(dataset, grp1, grp2, date_col='date', val='value', sep='_', agg_type='hierarchy'):
    """
    ----------
    Parameters
    ----------
    dataset : pandas dataframe with a date column, two hierarchies, and a value
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
    data = dataset.copy()
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


#---------------------------------------------------------------------
# Forecasting Functions
#---------------------------------------------------------------------   


def hier_arima(col, forecast_idx, order=(1,1,0), steps_out=1, make_stationary=True):
    """
    ----------
    Parameters
    ----------
    col : column from the pandas dataframe output from `get_hierarchal()`
    forecast_idx : datetime index of forecast period
    order : order of ARIMA forecasting model
    steps_out : number of periods in forecast horizon
    make_stationary : bool - difference the column to make data stationary
    ----------
    Returns
    ----------
    out : dictionary with yhat, training data (col), and the fitted model
    """
    # ARIMA forecast; make data stationary by taking the difference
    if make_stationary is True:
        mod = ARIMA(col.diff(), order=order, enforce_stationarity=False, enforce_invertibility=False)
        mod = mod.fit(method_kwargs={'warn_convergence': False})
        # predict; undo differencing by taking the cumsum
        if steps_out == 1:
            yhat = mod.predict(forecast_idx[0]).values + col.iloc[-1]
        else:
            yhat = mod.predict(start=forecast_idx[0], end=forecast_idx[-1]).values 
            yhat[0] += col.iloc[-1]
            yhat = np.cumsum(yhat)
    else:
        mod = ARIMA(col, order=order, enforce_stationarity=False, enforce_invertibility=False)
        mod = mod.fit(method_kwargs={'warn_convergence': False})
        # predict
        if steps_out == 1:
            yhat = mod.predict(forecast_idx[0]).values
        else:
            yhat = mod.predict(start=forecast_idx[0], end=forecast_idx[-1]).values 
    out = {col.name:{'yhat':yhat, 'training_df':col, 'model':mod}}
    return out


def hier_auto_arima(col, future=False, horizon=6, verbose=True, dict_out=False, **kwargs):
    """
    ----------
    Parameters
    ----------
    col : column from the pandas dataframe output from `get_hierarchal()`
    future : bool - whether or not to predict future period
    horizon : int - n steps in future period
    verbose : bool - whether to print error
    dict_out : bool - whether the output should be a dictionary format
    **kwargs : additional parameters for model_selection.train_test_split 
               and pm.auto_arima
    ----------
    Returns
    ----------
    out : dictionary with yhat, training data (col), and the fitted model 
          (OR) if only training/testing, the prediction, C.I., and model
    """
    train, test = model_selection.train_test_split(col, **{'train_size': .8, **kwargs})
    # SARIMA(p, d, q)(P, D, Q)m 
    mod = pm.auto_arima(
        train, 
        **{
            'start_p' : 0, # p - Trend Order autoregression
            'start_q' : 0, # q - Trend Order 
            'start_P' : 0, # P - Seasonal Order autoregression
            'start_Q' : 0, # Q - Seasonal Order MA
            'max_p' : 5, 
            'max_q' : 5, 
            'max_P' : 3, 
            'max_Q' : 3, 
            'max_d' : 3, # d - Trend Order difference
            'max_D' : 3, # D - Seasonal Order Difference
            'm' : 12, # m - time steps in seasonal period
            'seasonal' : True, 
            'stepwise' : True, 
            'suppress_warnings' : True, 
            'error_action' : 'ignore',
            **kwargs
            }
        )
    test_preds, test_conf_int = mod.predict(n_periods=test.shape[0], return_conf_int=True)
    if verbose is True:
        print(f"Test RMSE - {col.name}: %.3f" % np.sqrt(mean_squared_error(test, test_preds)))
    if future is False:
        return mod, test_preds, test_conf_int
    else:
        preds, conf_int = mod.predict(n_periods=horizon, return_conf_int=True)
        if dict_out is False:
            return mod, preds, conf_int
        else:
            out = {col.name:{'yhat':preds, 'training_df':col, 'model':mod, 'conf_int':conf_int}}
            return out


def get_models(hdf, method='auto_arima', steps_out=6, period='months', **kwargs):
    """
    ----------
    Parameters
    ----------
    hdf : pandas dataframe - output from `get_hierarchal()`
    order : order of ARIMA forecasting model
    steps_out : number of periods in forecast horizon
    period : time period (default months)
    make_stationary : bool - difference the data to make data stationary
    ----------
    Returns
    ----------
    mods : dictionary containing fitted models and metadata
    """
    if period.lower() in ['month', 'months']:
        if steps_out == 1:
            forecast_idx = np.array([hdf.index[-1] + relativedelta(months=1)])
        else:
            forecast_idx = pd.date_range(
                hdf.index[-1] + relativedelta(months=1), 
                hdf.index[-1] + relativedelta(months=steps_out), 
                freq=hdf.index.freq)
        cols_dict = dict()
        mods = {'index':forecast_idx}
        # apply ARIMA to each column n steps out
        if method.lower() == 'arima':
            hdf.apply(lambda x: cols_dict.update(hier_arima(
                col=x, 
                forecast_idx=forecast_idx, 
                steps_out=steps_out,
                **{
                    'order' : (1,0,0), 
                    'make_stationary' : True,
                    **kwargs
                    }
                )))
            mods.update({'columns':cols_dict})
            return mods
        # apply auto_arima to each column n steps out
        elif method.lower() == 'auto_arima':
            hdf.apply(lambda x: cols_dict.update(hier_auto_arima(
                col=x, 
                future=True, 
                horizon=steps_out, 
                dict_out=True,
                **{
                    'verbose' : True, 
                    **kwargs
                }
            )))
            mods.update({'columns':cols_dict})
            return mods
        else:
            raise ValueError('Method must either be "arima" or "auto_arima"')
    else:
        raise ValueError('Invalid period')


#---------------------------------------------------------------------
# Reconciliation
#---------------------------------------------------------------------   


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


#---------------------------------------------------------------------
# Output functions & Wrapper
#---------------------------------------------------------------------   


def predict_hier(hdf, yh, rec, labs, forecast_idx):
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
    # Create additional column flagging forecasted records
    # Append forecasts to actuals (both original and reconciled)
    hdf['actual'] = True
    yh_df = pd.DataFrame(data=yh, columns=labs, index=forecast_idx)
    yh_df['actual'] = False
    hdf_yhat = pd.concat([hdf, yh_df], axis=0)
    rec_df = pd.DataFrame(data=rec, columns=labs, index=forecast_idx)
    rec_df['actual'] = False
    hdf_rec = pd.concat([hdf, rec_df], axis=0)
    return hdf_yhat, hdf_rec


### TODO: Wrapper function OR refactor as classes?