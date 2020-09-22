import os
import numpy as np
import pandas as pd

import util

LABELS = ['Affect_Anger', 'Affect_Anxiety_Fear', 'Affect_Pleasure', 'Affect_Sadness', 'MicroSphere', 'MacroSphere']


def train_test_splits(df_info: pd.DataFrame, data: pd.DataFrame):
    # The first line assures that only the feature extracted sessions are used
    subset = df_info.where(df_info['Key'].isin(data['Key'].unique())).dropna()
    id_list = subset['ID'].unique()
    for ID in id_list:
        yield subset[subset['ID'] != ID]['Key'], subset[subset['ID'] == ID]['Key']


def load_text_va(info_df: pd.DataFrame, dir_path) -> pd.DataFrame:
    data = pd.DataFrame(columns=['Key', 'v', 'a'])
    print('Loading text valence/arousal files!')
    for index, row in info_df.iterrows():
        file_path = os.path.join(dir_path, row['Initials'], f"Seans{row['Session']}Play{row['Segment_No']}")
        current = pd.read_csv(file_path, delim_whitespace=True, header=None)
        current.columns = ['v', 'a']
        current['Key'] = row['Key']
        data = data.append(current, ignore_index=True)
    return data


def load_face_va(info_df: pd.DataFrame, dir_path: str) -> pd.DataFrame:
    data = pd.DataFrame(columns=['Key', 'CameraNo', 'FrameNo', 'Person', 'v', 'a', 'conf'])
    print('Loading facial emotion valence/arousal files!')
    for index, row in info_df.iterrows():
        for camera_no in [1, 2]:
            file_path = os.path.join(dir_path, row['Initials'], "kayıtlar", str(row['Session']), f"{row['Session']}K{camera_no}Play{row['Segment_No']}.txt")
            try:
                current = pd.read_csv(file_path, delim_whitespace=True, header=None)
            except pd.errors.EmptyDataError:
                continue
            current.columns = ['FrameNo', 'Person', 'v', 'a', 'conf']
            current['Key'] = row['Key']
            current['CameraNo'] = camera_no
            data = data.append(current, ignore_index=True)
    return data


def _extract_optic_flow_features(info_df: pd.DataFrame, dir_path: str, pickle_path: str):
    data = None
    print('Extracting optic flow features!')
    for index, row in info_df.iterrows():
        for camera_no in [1, 2]:
            file_path = os.path.join(dir_path, row['Initials'], str(row['Session']), f"{row['Session']}K{camera_no}play{row['Segment_No']}.txt")
            current = pd.read_csv(file_path, delim_whitespace=True, header=None)
            child, therapist = current.iloc[:, :3].copy(), current.iloc[:, 3:]
            for p in range(2):
                cur_person = current.iloc[:, p*3:(p+1)*3].copy()
                cur_person.columns = ['x', 'y', 'm']
                cur_person['Key'] = row['Key']
                cur_feats = util.extract_features(cur_person[cur_person['x'] != -10000000], columns=['x', 'y', 'm'])
                cur_feats = cur_feats.reset_index()
                cur_feats['CameraNo'], cur_feats['Person'] = camera_no, p
                if data is None:
                    data = cur_feats.copy()
                else:
                    if len(cur_feats) == 0:
                        # Fill all zeros
                        prev_feats.loc[:, 'x_mean':'m_var'] = 0
                        prev_feats['CameraNo'], prev_feats['Person'] = camera_no, p
                        prev_feats['Key'] = row['Key']
                        data = data.append(prev_feats, ignore_index=True)
                    else:
                        data = data.append(cur_feats, ignore_index=True)
                        prev_feats = cur_feats.copy()
    data.to_pickle(pickle_path)
    return data


def load_optic_flow(info_df: pd.DataFrame, dir_path: str):
    pickle_path = os.path.join(dir_path, 'optic_flow_feats.pkl')
    if os.path.exists(pickle_path):
        print('Loading optic flow features!')
        return pd.read_pickle(pickle_path)
    else:
        return _extract_optic_flow_features(info_df, dir_path, pickle_path)


def load_labels(path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    :param path: path to the SPSS.csv
    :return: 2 dataframes with matching indices. (info: details about ID, initials and session, labels: )
    """
    print('Loading labels!')
    data = pd.read_csv(path)
    data['Key'] = data.index
    return data[['Key', 'ID', 'Initials', 'Session', 'Segment_No', 'Age', 'Gender', 'Diagnosis']], data[['Key'] + LABELS]

