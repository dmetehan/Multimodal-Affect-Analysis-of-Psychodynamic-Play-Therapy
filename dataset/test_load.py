import os
from unittest import TestCase

from dataset.load import load_text_va, load_face_va, load_optic_flow, load_labels


class Test(TestCase):
    def test_load(self):
        ROOT = r'D:\Datasets\Bilgi Universitesi'
        info, labels = load_labels(os.path.join(ROOT, 'SPSS.csv'))
        _ = load_text_va(info, os.path.join(ROOT, 'text_va'))
        _ = load_face_va(info, os.path.join(ROOT, 'face_va'))
        _ = load_optic_flow(info, os.path.join(ROOT, 'optic_flow'))
