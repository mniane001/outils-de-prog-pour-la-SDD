import numpy as np
import os
import pandas as pd

pd.set_option('display.max_columns', None)
df=pd.read_table("data/datatraining.txt", sep=",")
print(df.isna().sum())