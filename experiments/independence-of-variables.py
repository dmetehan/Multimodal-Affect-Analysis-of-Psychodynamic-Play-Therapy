import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

import dataset as db
import util


def independence_test(data: pd.DataFrame, ground_truth: pd.DataFrame):
    print('\nPearson with target variable:')
    if 'Key' in data.columns:
        merged = data.merge(ground_truth, left_on='Key', right_index=True)
        util.print_dataframe(merged.drop(columns='Key').corr('pearson').loc[:'a', 'Affect_Anger':])
        data = data.drop(columns='Key')
    else:
        merged = data.merge(ground_truth.drop(columns='Key'), left_index=True, right_index=True)
        util.print_dataframe(merged.corr('pearson').loc[:'arousal_var', 'Affect_Anger':])
    ground_truth = ground_truth.drop(columns='Key')
    print('\nPearson between features:')
    util.print_dataframe(data.corr('pearson'))
    # print('\nKendall:')
    # util.print_dataframe(data.corr('kendall'))
    # print('\nSpearman:')
    # util.print_dataframe(data.corr('spearman'))
    print('\nMutual Information:')
    for c, cols in enumerate(data):
        print(cols, mutual_info_regression(data, data.iloc[:, c], discrete_features=False))
    # for col_a, col_b in itertools.combinations(data.columns.tolist(), 2):
    #     print(col_a, col_b, pearsonr(data.loc[:, col_a], data.loc[:, col_b]))


ROOT = r'D:\Datasets\Bilgi Universitesi'
info, labels = db.load_labels(os.path.join(ROOT, 'SPSS.csv'))
text_va = db.load_text_va(info, os.path.join(ROOT, 'text_va'))
# fake = pd.DataFrame(np.random.randint(0, 5, size=text_va.shape), columns=text_va.columns)
# fake['Key'] = text_va['Key']
independence_test(text_va, labels)
text_va = util.preprocess_text_va(text_va)
# fake = pd.DataFrame(np.random.randint(0, 5, size=text_va.shape), columns=text_va.columns)
# fake['Key'] = text_va['Key']
independence_test(text_va, labels)
