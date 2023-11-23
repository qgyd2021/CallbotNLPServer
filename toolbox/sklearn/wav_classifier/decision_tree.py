#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from toolbox.python_speech_features.wave_features import calc_wave_features
from scipy.io import wavfile
from python_speech_features import sigproc
import scipy.signal

from project_settings import project_path


class DecisionTreeAudioSearch(object):
    def __init__(self, decision_tree_pkl_path: str, win_len: float = 2.0, win_step: float = 0.5):
        self.decision_tree_pkl_path = decision_tree_pkl_path
        self.win_len = win_len
        self.win_step = win_step

        with open(self.decision_tree_pkl_path, 'rb') as f:
            tree = pickle.load(f)
        self.tree: DecisionTreeClassifier = tree

    def search(self, signal, sample_rate):
        if len(signal) < self.win_len * sample_rate:
            raise AssertionError

        frames = sigproc.framesig(
            sig=signal,
            frame_len=self.win_len * sample_rate,
            frame_step=self.win_step * sample_rate,
        )
        frames = np.array(frames, dtype=np.int16)

        features = list()
        for frame in frames:
            feature = calc_wave_features(frame, sample_rate)
            features.append(feature)

        x = list()

        for feature in features:
            x.append([
                feature['mean'],
                feature['var'],
                feature['per1'],
                feature['per25'],
                feature['per50'],
                feature['per75'],
                feature['per99'],
                feature['silence_rate'],
                feature['mean_non_silence'],
                feature['silence_count'],
                feature['var_var_non_silence'],
                feature['var_non_silence'],
                feature['var_non_silence_rate'],
                feature['var_var_whole'],
            ])
        # 概率预测
        predict = self.tree.predict(x)
        return predict

    def search_by_filename(self, filename: str):
        sample_rate, signal = wavfile.read(filename)
        return self.search(signal, sample_rate)


def demo1():
    tree = DecisionTreeAudioSearch(
        decision_tree_pkl=os.path.join(project_path, 'data/audio/886/decision_tree.pkl'),
    )

    filename = os.path.join(project_path, 'data/audio/886/886_wav/bell/bell_20220120115502.wav')

    result = tree.search_by_filename(filename)
    print(result)
    return


if __name__ == '__main__':
    demo1()
