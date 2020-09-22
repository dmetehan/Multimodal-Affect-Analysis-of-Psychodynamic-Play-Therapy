import os
import numpy as np
import pandas as pd

import util
import dataset
import visualization as vis


def statistics(info, labels):
    count = len(info["ID"].unique())
    print(f'{count} children')
    diagnosis = info.groupby('ID').agg({'Diagnosis': 'mean'})
    print(f'{np.count_nonzero(diagnosis)} children with internalizing, externalizing and comorbid problems')
    print(f'{np.count_nonzero(diagnosis == 1) / count:.2f} internalizing', end='\t')
    print(f'{np.count_nonzero(diagnosis == 2) / count:.2f} externalizing', end='\t')
    print(f'{np.count_nonzero(diagnosis == 3) / count:.2f} comorbid')


ROOT = r'D:\Datasets\Bilgi Universitesi'
info, labels = dataset.load_labels(os.path.join(ROOT, 'SPSS.csv'))
statistics(info, labels)
