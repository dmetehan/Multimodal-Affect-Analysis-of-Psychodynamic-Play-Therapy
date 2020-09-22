import os
import pickle


def save_pickle(pickle_dir, pickle_abs_path, pickle_object):
    print('Saving pickle at {}'.format(pickle_abs_path))
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    with open(pickle_abs_path, 'wb') as handle:
        pickle.dump(pickle_object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(pickle_abs_path):
    print('Loading pickle from {}'.format(pickle_abs_path))
    if os.path.exists(pickle_abs_path):
        return pickle.load(open(pickle_abs_path, "rb"))
    else:
        return None
