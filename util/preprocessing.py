import itertools
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression


def get_x_y(data, ground_truth, keys):
    x, y = data.loc[data['Key'].isin(keys), :].copy(), ground_truth.loc[keys, :].copy()
    y = y.reset_index(drop=True)
    y.index = x.index
    x = x.drop(columns='Key')
    return x, y


def combine_text_face(text, face, labels):
    prep_text = extract_features(text, columns=['v', 'a'])
    prep_face = extract_features(face, columns=['v', 'a'])
    prep_face.columns = [f'{col}_face' for col in prep_face.columns]
    combined = pd.concat([prep_text, prep_face], axis=1).dropna()

    # labels = labels.loc[labels['Key'].isin(combined.index), 'Affect_Anger':]
    # transformed = SelectKBest(mutual_info_regression, k=8).fit_transform(combined, labels.iloc[:, 1])
    # transformed = pd.DataFrame(transformed, index=combined.index)
    # transformed.reset_index(inplace=True)

    # combined = combined.groupby('Key').agg({'v_mean': 'mean', 'v_min': 'min', 'v_max': 'max', 'v_var': 'sum',
    #                                         'a_mean': 'mean', 'a_min': 'min', 'a_max': 'max', 'a_var': 'sum'})

    combined.reset_index(inplace=True)
    return combined


def get_funcs(names=('mean', 'median', 'std', 'min', 'max', 'diff', 'diff2', 'var')):
    diff = lambda x: x.diff().mean() if len(x) > 1 else 0
    diff.__name__ = 'diff'
    diff2 = lambda x: x.diff().diff().mean() if len(x) > 2 else 0
    diff2.__name__ = 'diff2'
    func_dict = {'mean': 'mean', 'median': 'median', 'std': 'std', 'min': 'min', 'max': 'max', 'diff': diff, 'diff2': diff2, 'var': 'var'}
    return [func_dict[fun] for fun in names], names


def extract_features(data: pd.DataFrame, columns: List[str], funcs=()):
    if len(funcs) == 0:
        funcs, names = get_funcs()
    feats = data.groupby('Key').agg({col: funcs for col in columns})
    feats.columns = [f'{col}_{fun if type(fun) is str else fun.__name__}' for (col, fun) in itertools.product(columns, funcs)]
    return feats.fillna(0)


def scaler_function(data: pd.DataFrame, scaler, train_keys=None) -> pd.DataFrame:
    if train_keys is None:
        prep_data = pd.DataFrame(scaler.fit_transform(data))
    else:
        scaler.fit(data.loc[set(train_keys).intersection(set(data.index))])
        prep_data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    return prep_data


def preprocess_text_va(data: pd.DataFrame, scaler=StandardScaler(), train_keys=None, funcs=()) -> pd.DataFrame:
    prep_data = extract_features(data, columns=['v', 'a'], funcs=funcs)
    prep_data.reset_index(inplace=True)
    final_data = scaler_function(prep_data.iloc[:, 1:], scaler=scaler, train_keys=train_keys)
    final_data['Key'] = prep_data['Key']
    return final_data


def preprocess_face_va(data: pd.DataFrame, scaler=StandardScaler(), train_keys=None, funcs=()) -> pd.DataFrame:
    # TODO: camera-based approaches might be added
    return preprocess_text_va(data, scaler=scaler, train_keys=train_keys, funcs=funcs)


def preprocess_optic_flow(data: pd.DataFrame, scaler=StandardScaler(), train_keys=None, funcs=()) -> pd.DataFrame:
    # pca = PCA(n_components=5)
    # transformed = pca.fit_transform(data.loc[:, 'x_mean':'m_var'])
    # prep_data = scaler_function(pd.DataFrame(transformed), scaler=scaler, train_keys=train_keys)
    prep_data = scaler_function(data.loc[:, 'x_mean':'m_var'], scaler=scaler, train_keys=train_keys)
    prep_data['Key'] = data['Key']
    return prep_data


def preprocess_face_text_combined(data: pd.DataFrame, scaler=StandardScaler(), train_keys=None) -> pd.DataFrame:
    prep_data = scaler_function(data.iloc[:, 1:-1], scaler=scaler, train_keys=train_keys)
    prep_data['Key'] = data['Key']
    return prep_data


def split_child_therapist(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    child, therapist = data[data['Person'] == 0], data[data['Person'] == 1]
    child = child.reset_index(drop=True)
    therapist = therapist.reset_index(drop=True)
    child = child.drop(columns='Person')
    therapist = therapist.drop(columns='Person')
    return child, therapist


def threshold_face_va(data: pd.DataFrame, conf_threshold=0.4) -> (pd.DataFrame, pd.DataFrame):
    child, therapist = split_child_therapist(data)
    # confidence thresholds of 0.4 eliminates 8.7% of child faces and 4.6% of therapist faces
    child, therapist = child[child['conf'] > conf_threshold], therapist[therapist['conf'] > conf_threshold]
    return child, therapist
