from dateutil.relativedelta import relativedelta
import random
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.style.use('bmh')

def get_sample_data(agg_type='hierarchy', verbose=False, extend=True, years=5):
    if agg_type == 'hierarchy':
        file_loc = '../data/hierarchal_agg.csv'
    elif agg_type == 'grouped':
        file_loc = '../data/grouped_agg.csv'
    else:
        raise ValueError('agg_type must be either "hierarchy" or "grouped"')
    data = pd.read_csv(file_loc)
    # Update Data Types
    data['date'] = pd.to_datetime(data['date'])
    data['subcategory'] = data['subcategory'].astype('category')
    data['category'] = data['category'].astype('category')
    if extend is True:
        data = data.groupby(['category', 'subcategory'], as_index=False)\
            .apply(lambda x: append_years(x, years=years)).reset_index(drop=True)
    if verbose is True:
        display(data.head())
        data.info()
        for col in data.columns:
            if data[col].dtype == 'category':
                print(f'{col} unique values: {len(data[col].unique())}')
            elif data[col].dtype == 'datetime64[ns]':
                print(f'Min Date: {data[col].min()}')
                print(f'Max Date: {data[col].max()}')
            else:
                print(f'{col}:\n{data[col].describe()}\n')
    return data

def append_years(data, years=5):
    start_date = max(data.date) + relativedelta(months=1)
    new_end_date = start_date + relativedelta(months=years * 12 - 1)
    dates = pd.date_range(start_date, new_end_date, freq='MS')
    new_data = pd.DataFrame(
        {
            'date' : dates,
            'category' : data.category.unique()[0],
            'subcategory' : data.subcategory.unique()[0]
        }
        )
    value = np.round(data.groupby(data.date.dt.month).value.mean().values * random.uniform(.5, 2) + random.uniform(0, 2), 1)
    if years > 1:
        for i in range(years - 1):
            value = np.concatenate((
                value,                
                np.round(data.groupby(data.date.dt.month).value.mean().values * random.uniform(.5, 2) + random.uniform(0, 2), 1)
            ), axis=0)
    new_data['value'] = value
    data = pd.concat([data, new_data], axis=0)
    return data.reset_index(drop=True)


def plot_single(data, rec_data, col='total'):
    data['pred'] = ~data.actual
    data.loc[max(data.loc[(data.pred == False)].index), 'pred'] = True
    rec_data['pred'] = ~rec_data.actual
    rec_data.loc[max(rec_data.loc[(rec_data.pred == False)].index), 'pred'] = True
    fig, ax = plt.subplots(figsize=(16, 5))
    l1, = ax.plot(data.loc[data.actual==True][col], lw=1.5, c='blue')
    l1.set_label(f'Actual {col}')
    l2, = ax.plot(data.loc[data.pred==True][col], lw=1.5, c='darkred')
    l2.set_label(f'Forecast {col}')
    l3, = ax.plot(rec_data.loc[rec_data.pred==True][col], lw=1.5, c='green')
    l3.set_label(f'Reconciled Forecast {col}')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', shadow=True, prop={'size': 11})