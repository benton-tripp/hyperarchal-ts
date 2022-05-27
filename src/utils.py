from dateutil.relativedelta import relativedelta
import random
import pandas as pd
import numpy as np

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