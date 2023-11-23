#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from glob import glob
import os
from tqdm import tqdm
from typing import Dict, List

import cv2 as cv
import numpy as np
from scipy.io import wavfile

from toolbox.python_speech_features.misc import wave2spectrum_image
from toolbox.cv2.misc import show_image


class BasicTemplateMatch(object):
    """
    音频模板匹配
    """
    def __init__(self,
                 template_path: str,
                 sample_rate: int = 8000,
                 template_crop: float = 0.1,
                 threshold: float = 0.01,
                 ):
        super().__init__()
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

            template = wave2spectrum_image(
                wave=wave,
                sample_rate=self.sample_rate,
                n_low_freq=100
            )
            # show_image(template)

            template_width, _ = template.shape
            if template_width > max_template_width:
                max_template_width = template_width
            label2templates[label].append({
                'filename': filename,
                'template': template,
            })

        return label2templates, max_template_width

    def _shadow_match_template(self, spectrum: np.ndarray):
        matches = list()

        if spectrum.shape[0] < self.max_template_width:
            return matches

        for label, templates in self.label2templates.items():
            for templ in templates:
                filename = templ['filename']
                template = templ['template']

                tw, _ = template.shape[:2]
                c = int(tw * self.template_crop)
                if c != 0:
                    template = template[c: -c]

                tw, th = template.shape[:2]

                shadow_m = 10
                shadow_spect = spectrum[:, :shadow_m]
                shadow_templ = template[:, :shadow_m]

                sqdiff_normed = cv.matchTemplate(image=shadow_spect, templ=shadow_templ, method=cv.TM_SQDIFF_NORMED)
                min_val, _, min_loc, _ = cv.minMaxLoc(sqdiff_normed)
                # print(min_val, min_loc)
                if min_val > self.threshold:
                    continue

                # master
                _, x = min_loc
                match_spectrum = spectrum[x:x+tw, :]
                sqdiff_normed = cv.matchTemplate(image=match_spectrum, templ=template, method=cv.TM_SQDIFF_NORMED)

                min_val, _, min_loc, _ = cv.minMaxLoc(sqdiff_normed)
                # print(min_val, min_loc)
                if min_val > self.threshold:
                    continue

                matches.append({
                    'begin': x,
                    'width': tw,
                    'label': label,
                    'filename': filename,
                    'min_val': min_val,
                })
        return matches

    def search(self, wave: np.ndarray):
        max_wave_value = 32768.0
        wave = wave / max_wave_value

        spectrum = wave2spectrum_image(
            wave=wave,
            sample_rate=self.sample_rate,
            n_low_freq=100
        )
        # show_image(spectrum.T)
        matches = self._shadow_match_template(spectrum)
        return matches


def demo1():
    from project_settings import project_path

    template_path = os.path.join(project_path, 'server/call_monitor/data/template/ja-JP/voicemail')

    basic_template_match = BasicTemplateMatch(
        template_path=template_path,
        sample_rate=8000,
        template_crop=0.0,
        threshold=0.007,
    )

    # filename_pattern = r'D:\程序员\ASR数据集\voicemail\origin_wav\ja-JP/*.wav'
    filename_pattern = os.path.join(project_path, 'server/call_monitor/data/badcase/data/template_match/ja-JP/*.wav')
    filename_list = glob(filename_pattern)

    for filename in tqdm(filename_list):
        sample_rate, signal = wavfile.read(filename)

        matches = basic_template_match.search(signal)

        print(filename)
        print(matches)

    return


def demo2():
    import shutil

    from project_settings import project_path

    template_path = os.path.join(project_path, 'server/call_monitor/data/template/ja-JP/voicemail')

    basic_template_match = BasicTemplateMatch(
        template_path=template_path,
        sample_rate=8000,
        template_crop=0.05,
        threshold=0.007,
    )

    filename_pattern = r'D:\程序员\ASR数据集\voicemail\origin_wav\ja-JP/*.wav'
    # filename_pattern = os.path.join(project_path, 'server/call_monitor/data/badcase/data/template_match/ja-JP/*.wav')
    filename_list = glob(filename_pattern)

    for filename in tqdm(filename_list):
        path, fn = os.path.split(filename)
        sample_rate, signal = wavfile.read(filename)

        matches = basic_template_match.search(signal)

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
