import os
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import dataset as db
import visualization as vis
import util

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from sklearn_extensions.extreme_learning_machines import MLPRandomLayer
from sklearn_extensions.extreme_learning_machines.elm import GenELMRegressor


def get_x_y(data, ground_truth, keys):
    x, y = data.loc[data['Key'].isin(keys), :].copy(), ground_truth.loc[keys, :].copy()
    y = y.reset_index(drop=True)
    y.index = x.index
    x = x.drop(columns='Key')
    return x, y


def leave_one_subject_out(raw_data, preprocess_fn, ground_truth, df_info, funcs=()):
    ground_truth = ground_truth.drop(columns='Key')
    regressors = {'Linear Regression': LinearRegression(),
                  'Polynomial Regression': Pipeline([('poly', PolynomialFeatures(degree=2)),
                                                     ('linear', LinearRegression(fit_intercept=False))]),
                  'Support Vector Regressor (SVR) Linear Kernel': MultiOutputRegressor(SVR(kernel='linear', C=1.0, epsilon=0.2, max_iter=10000)),
                  'Support Vector Regressor (SVR) RBF Kernel': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.2, max_iter=10000)),
                  'Extreme Learning Machine Regressor': GenELMRegressor(MLPRandomLayer(n_hidden=25, random_state=0))}
    scalers = {'Linear Regression': StandardScaler(),
               'Polynomial Regression': StandardScaler(),
               'Support Vector Regressor (SVR) Linear Kernel': StandardScaler(),
               'Support Vector Regressor (SVR) RBF Kernel': StandardScaler(),
               'Extreme Learning Machine Regressor': StandardScaler()}
    results = {}
    for name, regressor in regressors.items():
        print(f'{name}', end='\t')
        all_preds = pd.DataFrame(np.ones_like(ground_truth) * np.nan, columns=ground_truth.columns)
        all_preds['Key'] = ground_truth.index
        for train_keys, test_keys in db.train_test_splits(df_info, raw_data):
            reg = clone(regressor)
            data = preprocess_fn(raw_data, scaler=scalers[name], train_keys=train_keys, funcs=funcs)
            train_keys, test_keys = data.loc[data['Key'].isin(train_keys), 'Key'], data.loc[data['Key'].isin(test_keys), 'Key']
            train_x, train_y = get_x_y(data, ground_truth, train_keys)
            test_x, test_y = get_x_y(data, ground_truth, test_keys)
            reg.fit(train_x, train_y)
            preds = reg.predict(test_x)
            # preds = pd.DataFrame(preds.values if type(preds) == pd.DataFrame else preds, index=test_keys).clip(0, 5)
            preds = pd.DataFrame((preds.values * 2) if type(preds) == pd.DataFrame else preds * 2, index=test_keys).round(0).div(2).clip(0, 5)
            preds.columns = ground_truth.columns
            if len(all_preds.loc[all_preds['Key'].isin(test_keys), :ground_truth.columns[-1]]) != len(preds):
                preds.reset_index(inplace=True)
                preds = preds.groupby('Key').agg({col: 'mean' for col in preds.columns})
                preds.drop(columns=['Key'], inplace=True)
            else:
                preds.reset_index(drop=True, inplace=True)
            all_preds.loc[all_preds['Key'].isin(test_keys), :ground_truth.columns[-1]] = preds.values
        all_preds.dropna(inplace=True)
        if len(all_preds) == 2 * len(ground_truth):
            gt_reduced = ground_truth.iloc[np.arange(len(ground_truth)).repeat(2)].reset_index(drop=True).copy()
            gt_reduced = gt_reduced.iloc[all_preds.index, :].copy()
        else:
            gt_reduced = ground_truth.iloc[all_preds.index, :].copy()
        results[name] = (all_preds, gt_reduced)
    print()
    table = util.calc_results(results)
    # util.print_results_table(results, tab_delimited=True)
    # util.save_predictions(results, 'Support Vector Regressor (SVR) RBF Kernel', 'Affect_Anxiety_Fear', df_info)
    return table, results


def run_experiments(raw_data, preprocess_fn, ground_truth, df_info, name):
    all_tables, all_results = None, None
    for col in ground_truth:
        if col == 'Key':
            continue
        funcs, names = util.get_funcs()
        table, results = leave_one_subject_out(raw_data, preprocess_fn, ground_truth[['Key', col]], df_info, funcs=funcs)
        if all_tables is None:
            all_tables, all_results = table, results
        else:
            for key in all_tables:
                all_tables[key] = pd.concat([all_tables[key], table[key]], axis=1)
            for key in results:
                all_results[key] = (pd.concat([all_results[key][0], results[key][0]], axis=1), pd.concat([all_results[key][1], results[key][1]], axis=1))
    print(all_tables)
    util.save_all_predictions(all_results, df_info, name=name, all_tables=all_tables)


ROOT = r'D:\Datasets\Bilgi Universitesi'
info, labels = db.load_labels(os.path.join(ROOT, 'SPSS.csv'))

child_text_va = db.load_text_va(info, os.path.join(ROOT, 'text_va'))
therapist_text_va = db.load_text_va(info, os.path.join(ROOT, 'text_va_therapist'))

face_va = db.load_face_va(info, os.path.join(ROOT, 'face_va'))
child_face_va, therapist_face_va = util.threshold_face_va(face_va)

optic_flow = db.load_optic_flow(info, os.path.join(ROOT, 'optic_flow'))
child_optic_flow, therapist_optic_flow = util.split_child_therapist(optic_flow)

print('\n\n\tText Child\n')
run_experiments(child_text_va, util.preprocess_text_va, labels[['Key', 'Affect_Anger', 'Affect_Anxiety_Fear', 'Affect_Pleasure', 'Affect_Sadness']],
                info, 'text')

print('\n\n\tFace Child\n')
run_experiments(child_face_va, util.preprocess_face_va, labels[['Key', 'Affect_Anger', 'Affect_Anxiety_Fear', 'Affect_Pleasure', 'Affect_Sadness']],
                info, 'face')

