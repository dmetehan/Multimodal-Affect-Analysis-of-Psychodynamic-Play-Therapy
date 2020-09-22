import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def _calc_baseline(all_preds, ground_truth):
    baseline = []
    for c, cols in enumerate(ground_truth):
        mean_preds = np.full(shape=np.shape(ground_truth.loc[all_preds.iloc[:, c].index, cols]), fill_value=np.mean(ground_truth.loc[all_preds.loc[:, cols].index, cols]), dtype=np.float32)
        baseline.append(mean_squared_error(mean_preds, ground_truth.loc[all_preds.loc[:, cols].index, cols]))
    return np.array(baseline).reshape(1, -1)


def _calc_performance(all_preds, ground_truth):
    pcc, p_vals, mse = [], [], []
    for c, cols in enumerate(ground_truth):
        corr, p = pearsonr(all_preds.loc[:, cols], ground_truth.loc[all_preds.loc[:, cols].index, cols])
        pcc.append(corr)
        p_vals.append(p)
        mse.append(mean_squared_error(all_preds.loc[:, cols], ground_truth.loc[all_preds.loc[:, cols].index, cols]))
    return pcc, p_vals, mse


def save_predictions(results, regressor, column, info):
    all_preds, ground_truth = results[regressor]
    preds = all_preds.loc[:, ['Key', column]]
    ground_truth = ground_truth.loc[:, [column]]
    ground_truth[column+'_gt'] = ground_truth[column]
    merged = pd.merge(left=preds, right=info, left_on='Key', right_on='Key')
    merged = pd.merge(left=merged, right=ground_truth[column+'_gt'], left_on='Key', right_index=True)
    merged = merged.loc[:, info.columns.append(pd.Index([preds.columns[-1], ground_truth.columns[-1]]))]
    merged.to_csv(os.path.join(r'D:\PyCharm\Projects\PsychotherapyResearch\results', f'{regressor}_{column}_face.csv'), index=False)


def save_all_predictions(all_results, info, name='text', all_tables=None):
    if all_tables is not None:
        for key in all_tables:
            all_tables[key].to_csv(os.path.join(r'D:\PyCharm\Projects\PsychotherapyResearch\results', name, f'{name}_scores_{key}.csv'))
    for key in all_results:
        merged = _merge_preds_and_gt(*all_results[key])
        merged = pd.merge(left=info, right=merged, left_on='Key', right_on='Key')
        merged.to_csv(os.path.join(r'D:\PyCharm\Projects\PsychotherapyResearch\results', name, f'{name}_preds_{key}.csv'))


def _merge_preds_and_gt(all_preds, ground_truth):
    preds = all_preds.groupby(level=0, axis=1).mean().copy()
    gt = ground_truth.copy()
    gt.columns = [str(col) + '_gt' for col in gt.columns]
    merged = pd.merge(left=preds, right=gt, left_on='Key', right_index=True)
    return merged


def calc_results(results):
    table = {'pcc': pd.DataFrame(index=results.keys(), columns=results[list(results.keys())[0]][1].columns),
             'p_vals': pd.DataFrame(index=results.keys(), columns=results[list(results.keys())[0]][1].columns),
             'mse': pd.DataFrame(index=results.keys(), columns=results[list(results.keys())[0]][1].columns)}
    for n, name in enumerate(results):
        all_preds, ground_truth = results[name]
        pcc, p_vals, mse = _calc_performance(all_preds, ground_truth)
        table['pcc'].loc[name, :] = pcc
        table['p_vals'].loc[name, :] = p_vals
        table['mse'].loc[name, :] = mse
        if n == len(results) - 1:
            table['mse'] = table['mse'].append(pd.DataFrame(_calc_baseline(all_preds, ground_truth), index=['Baseline'], columns=ground_truth.columns))
    return table


def print_results_table(results, tab_delimited):
    table = calc_results(results)
    print_dataframe(table['pcc'], tab_delimited=tab_delimited)
    print_dataframe(table['p_vals'], tab_delimited=tab_delimited, scientific=True)
    print_dataframe(table['mse'], tab_delimited=tab_delimited)


def print_dataframe(df: pd.DataFrame, tab_delimited=False, scientific=False):
    if tab_delimited:
        print('', end='\t')
        for col in df:
            print(f'{col}', end='\t')
        print()
        for i, row in df.iterrows():
            print(f'{i}', end='\t')
            for col, val in row.items():
                if scientific:
                    print(f'{val:.2E}', end='\t')
                else:
                    print(f'{val:.3f}', end='\t')
            print()
    else:
        first_col_width = max(list(map(len, df.index)))

        print(f'{" ":{first_col_width}}', end=' ')
        for col in df:
            print(f'{col:{max(5, len(col))}}', end=' ')
        print()
        for i, row in df.iterrows():
            print(f'{i:{first_col_width}}', end=' ')
            for col, val in row.items():
                print(f'{val:{max(5, len(col))}.3f}', end=' ')
            print()
