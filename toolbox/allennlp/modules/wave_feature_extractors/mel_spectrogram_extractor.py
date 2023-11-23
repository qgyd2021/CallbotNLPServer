#!/usr/bin/python3
# -*- coding: utf-8 -*-
# import librosa
import torch

from toolbox.allennlp.modules.wave_feature_extractors.wave_feature_extractor import WaveFeatureExtractor
from toolbox.librosa import filters


@WaveFeatureExtractor.register('mel_spectrogram')
class MelSpectrogramExtractor(WaveFeatureExtractor):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 n_mels: int = 80,
                 fmin: int = 0,
                 fmax: int = None,
                 hop_size: int = 256,
                 win_size: int = 1024,
                 center: bool = False,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hop_size = hop_size
        self.win_size = win_size
        self.center = center

    def forward(self, inputs: torch.Tensor):
        if torch.min(inputs) < -1.:
            raise AssertionError()
        if torch.max(inputs) > 1.:
            raise AssertionError()

        device = inputs.device
        # mel = librosa.filters.mel(
        #     sr=self.sample_rate,
        #     n_fft=self.n_fft,
        #     n_mels=self.n_mels,
        #     fmin=self.fmin,
        #     fmax=self.fmax
        # )
        # linux 上安装 librosa 比较麻烦.
        mel = filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        mel = torch.from_numpy(mel).float().to(device)

        inputs = torch.nn.functional.pad(
            input=inputs,
            pad=(int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
            mode='reflect'
        )

        spec = torch.stft(
            inputs,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=torch.hann_window(self.win_size).to(device),
            center=self.center,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=False
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(mel, spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1)
        spec = torch.transpose(spec, dim0=1, dim1=2)
        return spec


def demo1():
    import os

    from scipy.io import wavfile
    from sphfile import SPHFile

    from toolbox.transformers.models.hifigan.modeling_hifigan import HiFiGAN
    from project_settings import project_path

    MAX_WAV_VALUE = 32768.0

    device = torch.device('cpu')
    torch.manual_seed(1234)

    model = HiFiGAN.from_pretrained(
        pretrained_model_name_or_path=os.path.join(project_path, 'pretrained/hifigan-lj-v1')
    )
    model.eval()
    model.remove_weight_norm()

    filename = os.path.join(project_path, 'datasets/asr/TIMIT/TEST/DR1/FAKS0/SA1.WAV')

    wave, sample_rate = librosa.load(filename, sr=22050)

    print(wave.shape)
    print(sample_rate)

    wave = torch.FloatTensor(wave).to(device)
    wave = torch.unsqueeze(wave, dim=0)

    mel_spectrogram_extractor = MelSpectrogramExtractor(
        n_fft=1024,
        n_mels=80,
        sample_rate=sample_rate,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000,
    )
    print(wave.shape)
    print(wave.dtype)
    x = mel_spectrogram_extractor.forward(wave)
    x = torch.transpose(x, dim0=1, dim1=2)

    with torch.no_grad():
        y_g_hat = model(x)
        wav = y_g_hat.squeeze()
        wav = wav * MAX_WAV_VALUE
        wav = wav.cpu().numpy().astype('int16')

        wavfile.write('test.wav', 22050, wav)

    return


if __name__ == '__main__':
    demo1()
