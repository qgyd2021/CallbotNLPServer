#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob
import os
from tqdm import tqdm
from typing import Dict, List

import cv2 as cv
import dtw
import numpy as np
from scipy.io import wavfile

from toolbox.cv2.misc import show_image
from toolbox.python_speech_features.misc import wave2spectrum, wave2spectrum_image


class DTWTemplateMatch(object):
    """
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple dtw==1.4.0

    dtw 是 python 实现的,
    匹配速度太慢
    """
    def __init__(self,
                 template_path: str,
                 sample_rate: int = 8000,
                 template_crop: float = 0.1,
                 threshold: float = 0.01,
                 ):
        self.template_path = template_path
        self.sample_rate = sample_rate
        self.template_crop = template_crop
        self.threshold = threshold

        label2templates, max_template_width = self._init_label2templates()

        self.label2templates: Dict[str, List[dict]] = label2templates

        # 输入音频的长度不能小于 max_template_width.
        self.max_template_width: int = max_template_width

    def _init_label2templates(self):
        filename_pattern = os.path.join(self.template_path, '*.wav')
        filename_list = glob(filename_pattern)
        label2templates = defaultdict(list)
        max_template_width = 0
        for filename in tqdm(filename_list):
            path, fn = os.path.split(filename)
            root_path, label = os.path.split(path)

            # wave, sample_rate = librosa.load(filename, sr=self.sample_rate)
            sample_rate, wave = wavfile.read(filename)
            if sample_rate != self.sample_rate:
                raise AssertionError('expected sample rate: {}, instead of: {}'.format(self.sample_rate, sample_rate))
            if wave.dtype != np.int16:
                raise AssertionError('expected wave dtype np.int16, instead of: {}'.format(wave.dtype))

            if wave.shape[0] < self.sample_rate:
                raise AssertionError('wave.shape: {}'.format(wave.shape))

            max_wave_value = 32768.0
            wave = wave / max_wave_value

            # template = wave2spectrum_image(
            #     wave=wave,
            #     sample_rate=self.sample_rate,
            #     n_low_freq=100
            # )
            # show_image(template)

            spectrum = wave2spectrum(
                wave=wave,
                sample_rate=self.sample_rate,
            )
            template = spectrum.T

            template_width, _ = template.shape
            if template_width > max_template_width:
                max_template_width = template_width
            label2templates[label].append({
                'filename': filename,
                'template': template,
            })

        return label2templates, max_template_width

    def _dtw_match_template(self, spectrum: np.ndarray):
        matches = list()

        for label, templates in self.label2templates.items():
            for templ in templates:
                filename = templ['filename']
                template = templ['template']

                cost, C, D1, path = dtw.dtw(spectrum, template, dist=lambda a, b: np.sum(a - b) ** 2)

                # print(cost)
                if cost < self.threshold:
                    matches.append({
                        'cost': cost,
                        'label': label,
                        'filename': filename,
                    })
        return matches

    def search(self, wave: np.ndarray):
        max_wave_value = 32768.0
        wave = wave / max_wave_value

        # spectrum = wave2spectrum_image(
        #     wave=wave,
        #     sample_rate=self.sample_rate,
        #     n_low_freq=100
        # )
        # show_image(spectrum.T)

        spectrum = wave2spectrum(
            wave=wave,
            sample_rate=self.sample_rate,
        )
        spectrum = spectrum.T
        matches = self._dtw_match_template(spectrum)
        return matches


def demo1():
    from project_settings import project_path

    template_path = os.path.join(project_path, 'server/call_monitor/data/template/ja-JP/voicemail')

    dtw_template_match = DTWTemplateMatch(
        template_path=template_path,
        sample_rate=8000,
        template_crop=0.0,
        threshold=100.0,
    )

    # filename_pattern = r'D:\程序员\ASR数据集\voicemail\origin_wav\ja-JP/*.wav'
    filename_pattern = os.path.join(project_path, 'server/call_monitor/data/badcase/data/template_match/ja-JP/*.wav')
    filename_list = glob(filename_pattern)

    for filename in tqdm(filename_list):
        sample_rate, signal = wavfile.read(filename)

        matches = dtw_template_match.search(signal)

        print(filename)
        print(matches)
        print(len(matches))

    return


def demo2():
    import shutil

    from project_settings import project_path

    template_path = os.path.join(project_path, 'server/call_monitor/data/template/ja-JP/voicemail')

    dtw_template_match = DTWTemplateMatch(
        template_path=template_path,
        sample_rate=8000,
        template_crop=0.05,
        threshold=100.0,
    )

    filename_pattern = r'D:\程序员\ASR数据集\voicemail\origin_wav\ja-JP/*.wav'
    # filename_pattern = os.path.join(project_path, 'server/call_monitor/data/badcase/data/template_match/ja-JP/*.wav')
    filename_list = glob(filename_pattern)

    for filename in tqdm(filename_list):
        path, fn = os.path.split(filename)
        sample_rate, signal = wavfile.read(filename)

        matches = dtw_template_match.search(signal)

        if len(matches) == 0:
            pass
            # to_path = os.path.join(path, 'wav_segmented/non_voicemail')
            # shutil.move(filename, to_path)
        else:
            to_path = os.path.join(path, 'wav_segmented/voicemail')
            shutil.move(filename, to_path)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
