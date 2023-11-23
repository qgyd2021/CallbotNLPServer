#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch

from toolbox.python_speech_features.misc import wave2spectrum_image
from toolbox.allennlp.modules.wave_feature_extractors.wave_feature_extractor import WaveFeatureExtractor


@WaveFeatureExtractor.register('wave_to_spectrum_gray')
class WaveToSpectrumGrayExtractor(WaveFeatureExtractor):
    def __init__(self,
                 sample_rate: int,
                 xmax: int = 10,
                 xmin: int = -50,
                 winlen: float = 0.025,
                 winstep: float = 0.01,
                 nfft: int = 512,
                 max_wave_value: float = 1.0,
                 n_low_freq: int = None,
                 ):
        super(WaveToSpectrumGrayExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.xmax = xmax
        self.xmin = xmin
        self.winlen = winlen
        self.winstep = winstep
        self.nfft = nfft
        self.max_wave_value = max_wave_value
        self.n_low_freq = n_low_freq

    def forward(self, wave: torch.FloatTensor):
        wave = wave.numpy()
        wave /= self.max_wave_value
        spectrum_gray = wave2spectrum_image(
            wave,
            sample_rate=self.sample_rate,
            xmax=self.xmax,
            xmin=self.xmin,
            winlen=self.winlen,
            winstep=self.winstep,
            nfft=self.nfft,
            n_low_freq=self.n_low_freq,
        )
        # from toolbox.cv2.misc import show_image
        # show_image(spectrum_gray.T)
        spectrum_gray = torch.tensor(spectrum_gray, dtype=torch.float32)
        return spectrum_gray


if __name__ == '__main__':
    pass
