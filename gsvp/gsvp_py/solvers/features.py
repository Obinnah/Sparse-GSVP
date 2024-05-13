import pandas as pd
import numpy as np
from tabulate import tabulate
from termcolor import colored
from IPython.display import HTML, display
import pandas as pd

def indices(a,b):
  merged = []
  seen = set()
  for num in a:
     if num not in seen:
        merged.append(num)
        seen.add(num)
  for num in b:
     if num not in seen:
        merged.append(num)
        seen.add(num)
  return np.array(merged)

def top_features(label, runs_list,**kwargs):
    x = kwargs['x']
    y = kwargs['y']
    feature_list = np.load(f'C:/Users/uugob/GSVP/dataheads/{label}.npy')
    result_df1 = pd.DataFrame({'Feature Index ({})'.format(x): runs_list['topA'], 'Top Features ({})'.format(x): [feature_list[idx] for idx in runs_list['topA']]})
    result_df2 = pd.DataFrame({'Feature Index ({})'.format(y): runs_list['topB'], 'Top Features ({})'.format(y): [feature_list[idx] for idx in runs_list['topB']]})
    combined_df = pd.concat([result_df1.iloc[0:11,:],result_df2.iloc[0:11,:]],axis=1)
    display(HTML(combined_df.style.set_table_styles([{'selector': 'thead', 'props': [('background', 'red')]}, {'selector': 'tbody', 'props': [('background', 'blue')]}]).to_html()))





