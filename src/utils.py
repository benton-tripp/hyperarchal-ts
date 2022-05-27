from dateutil.relativedelta import relativedelta
import random

def append_years(data, years=3):
    start_date = max(data.index) + relativedelta(months=1)
    new_end_date = start_date + relativedelta(months=years * 12 - 1)
    idx = pd.date_range(start_date, new_end_date, freq='MS')
    new_data = pd.DataFrame(
        columns=data.columns,
        data = np.round(data.groupby(data.index.month).mean().values * random.uniform(1, 1.25), 1)
        )
    if years > 1:
        for i in range(years - 1):
            new_data = pd.concat([
                new_data, 
                pd.DataFrame(
                    columns=data.columns,
                    data = np.round(data.groupby(data.index.month).mean().values * random.uniform(1, 1.25), 1)
            )], axis=0)
    new_data.index = idx
    return new_data