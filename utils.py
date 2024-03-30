import pandas as pd

def set_pandas_display_options(max_rows=500, max_cols=100, max_colwidth=800):
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_rows', max_cols)
    pd.set_option('display.max_colwidth', max_colwidth)