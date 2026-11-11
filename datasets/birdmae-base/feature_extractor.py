from transformers import SequenceFeatureExtractor
from transformers.tokenization_utils_base import BatchEncoding
from transformers.feature_extraction_utils import BatchFeature
from torchaudio.compliance.kaldi import fbank
import torch
import numpy as np
import torch.nn.functional as F

from typing import Union, List
from transformers.utils import PaddingStrategy


class BirdMAEFeatureExtractor(SequenceFeatureExtractor):
    _auto_class = "AutoFeatureExtractor"
    model_input_names = ["input_values"]

    def __init__(
            self,
            # process waveform
            feature_size: int = 1,
            sampling_rate: int = 32_000,
            padding_value: float = 0.0,
            return_attention_mask: bool = True,

            # fbank
            htk_compat: bool = True,
            use_energy: bool = False,
            window_type: str = "hanning",
            num_mel_bins: int = 128,
            dither: float = 0.0,
            frame_shift: int = 10,

            # pad and normalize
            target_length: int = 512,
            mean: float = -7.2,
            std: float = 4.43,

            **kwargs
    ):
        super().__init__(feature_size, sampling_rate, padding_value, **kwargs)
        # squence FE
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask

        # fbank
        self.htk_compat = htk_compat
        self.use_energy = use_energy
        self.window_type = window_type
        self.num_mel_bins = num_mel_bins
        self.dither = dither
        self.frame_shift = frame_shift

        # pad and normalize
        self.target_length = target_length
        self.mean = mean
        self.std = std

    def __call__(self,
                 waveform_batch: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
                 padding: Union[bool, str, PaddingStrategy] = "max_length",
                 max_length: int | None = None,
                 truncation: bool = True,
                 return_tensors: str = "pt"
                 ):

        if not torch.is_tensor(waveform_batch):
            #waveform_batch = torch.tensor((waveform_batch))
            waveform_batch = torch.from_numpy(np.array(waveform_batch))

        if len(waveform_batch.shape) == 1:
            waveform_batch = waveform_batch.unsqueeze(0)

        if len(waveform_batch.shape) != 2:
            raise ValueError("waveform_batch must be have 1 or 2 dimensions")

        waveform_batch = self._process_waveforms(waveform_batch, padding, truncation)

        fbank_features = self._compute_fbank_features(waveform_batch["input_values"])

        fbank_features = self._pad_and_normalize(fbank_features)

        return fbank_features.unsqueeze(1)

    def _process_waveforms(self,
                           waveforms,
                           padding: bool | str,
                           truncation: bool):
        clip_duration = 5 # TODO this is the clip duration used in training
        max_length = int(int(self.sampling_rate) * clip_duration)
        waveform_encoded = BatchFeature({"input_values": waveforms})

        waveform_batch = self.pad(
            waveform_encoded,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=self.return_attention_mask
        )

        attention_mask = waveform_batch.get("attention_mask")

        if attention_mask is not None:
            waveform_batch["attention_mask"] = attention_mask

        #waveform_batch["input_values"] = waveform_batch["input_values"] - waveform_batch["input_values"].mean(axis=1, keepdims=True)
        return waveform_batch

    def _compute_fbank_features(self, waveforms):
        fbank_features = [
            fbank(
                waveform.unsqueeze(0),
                htk_compat=self.htk_compat,
                sample_frequency=self.sampling_rate,
                use_energy=self.use_energy,
                window_type=self.window_type,
                num_mel_bins=self.num_mel_bins,
                dither=self.dither,
                frame_shift=self.frame_shift
            )
            for waveform in waveforms
        ]
        return torch.stack(fbank_features)

    def _pad_and_normalize(self, fbank_features):
        # fbank_features shape: [batch, time_frames, mel_bins]

        # Truncate if too long
        if fbank_features.shape[1] > self.target_length:
            fbank_features = fbank_features[:, :self.target_length, :]

        # Pad if too short
        elif fbank_features.shape[1] < self.target_length:
            difference = self.target_length - fbank_features.shape[1]
            min_value = fbank_features.min()
            padding = (0, 0, 0, difference)  # (left, right, top, bottom)
            fbank_features = F.pad(
                fbank_features,
                padding,
                value=min_value
            )

        # Normalize
        fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return fbank_features