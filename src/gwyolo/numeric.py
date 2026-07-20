from __future__ import annotations

import os
import random
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .factory import synthesize_scene
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .provenance import SceneRecipe, read_recipe_manifest

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised in dependency-minimal installations
    torch = None
    nn = None
    torch_functional = None
    DataLoader = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Numeric training requires the optional train dependencies, including torch")


class NumericRecipeDataset:
    def __init__(
        self,
        recipes: list[SceneRecipe],
        tensor_config: dict[str, Any],
        cache_in_memory: bool = False,
    ):
        self.recipes = recipes
        self.tensor_config = tensor_config
        self.cache: list[tuple[np.ndarray, np.ndarray] | None] | None = (
            [None] * len(recipes) if cache_in_memory else None
        )

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is not None and self.cache[index] is not None:
            return self.cache[index]  # type: ignore[return-value]
        arrays = synthesize_scene(self.recipes[index], self.tensor_config)
        features = arrays["features"].reshape(-1, *arrays["features"].shape[-2:])
        masks = np.stack([arrays["chirp_mask"], arrays["glitch_mask"]])
        masks = masks.reshape(2, -1, *masks.shape[-2:])
        item = features.astype(np.float32), masks.astype(np.float32)
        if self.cache is not None:
            self.cache[index] = item
        return item


if nn is not None:

    class _ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            groups = min(8, output_channels)
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, output_channels),
                nn.SiLU(),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)


    class MultiIFOQNet(nn.Module):
        """Compact numeric baseline retaining per-IFO/per-Q class masks."""

        def __init__(self, input_channels: int, base_channels: int = 24):
            super().__init__()
            self.input_channels = input_channels
            self.encoder = _ConvBlock(input_channels, base_channels)
            self.bottleneck = _ConvBlock(base_channels, base_channels * 2)
            self.decoder = _ConvBlock(base_channels * 3, base_channels)
            self.head = nn.Conv2d(base_channels, 2 * input_channels, 1)

        def decoded_features(self, value: Any) -> Any:
            encoded = self.encoder(value)
            low = self.bottleneck(torch_functional.max_pool2d(encoded, 2))
            up = torch_functional.interpolate(low, size=encoded.shape[-2:], mode="bilinear", align_corners=False)
            return self.decoder(torch.cat([encoded, up], dim=1))

        def forward(self, value: Any) -> Any:
            decoded = self.decoded_features(value)
            logits = self.head(decoded)
            return logits.reshape(value.shape[0], 2, self.input_channels, *value.shape[-2:])


    class DetectorSetQNet(nn.Module):
        """Shared-IFO encoder with explicit availability-masked set fusion."""

        def __init__(self, ifo_count: int, q_count: int, base_channels: int = 24):
            super().__init__()
            if ifo_count < 2 or q_count < 1:
                raise ValueError("DetectorSetQNet requires at least two IFO slots and one Q plane")
            self.ifo_count = int(ifo_count)
            self.q_count = int(q_count)
            self.input_channels = self.ifo_count * self.q_count
            self.base_channels = int(base_channels)
            self.shared_encoder = _ConvBlock(self.q_count, base_channels)
            self.attention_score = nn.Conv2d(base_channels, 1, 1)
            self.bottleneck = _ConvBlock(base_channels, base_channels * 2)
            self.shared_decoder = _ConvBlock(base_channels * 3, base_channels)
            self.shared_head = nn.Conv2d(base_channels, 2 * self.q_count, 1)

        def forward(self, value: Any, detector_availability: Any) -> Any:
            if value.ndim != 4 or value.shape[1] != self.input_channels:
                raise ValueError(
                    "DetectorSetQNet input must have shape [batch, IFO*Q, frequency, time]"
                )
            if detector_availability.ndim != 2 or tuple(detector_availability.shape) != (
                value.shape[0],
                self.ifo_count,
            ):
                raise ValueError("detector availability must have shape [batch, IFO]")
            availability = detector_availability.to(device=value.device, dtype=value.dtype)
            if not torch.all((availability == 0) | (availability == 1)):
                raise ValueError("detector availability must be binary")
            if torch.any(availability.sum(dim=1) < 1):
                raise ValueError("every sample requires at least one available detector")
            batch, _, frequency, time_bins = value.shape
            planes = value.reshape(
                batch, self.ifo_count, self.q_count, frequency, time_bins
            )
            encoded = self.shared_encoder(
                planes.reshape(batch * self.ifo_count, self.q_count, frequency, time_bins)
            ).reshape(batch, self.ifo_count, self.base_channels, frequency, time_bins)
            attention_logits = self.attention_score(
                encoded.reshape(
                    batch * self.ifo_count, self.base_channels, frequency, time_bins
                )
            ).reshape(batch, self.ifo_count, 1, frequency, time_bins)
            available_map = availability[:, :, None, None, None].to(dtype=torch.bool)
            attention = torch.softmax(
                attention_logits.masked_fill(~available_map, -torch.inf), dim=1
            )
            fused = torch.sum(attention * encoded, dim=1)
            low = self.bottleneck(torch_functional.max_pool2d(fused, 2))
            up = torch_functional.interpolate(
                low, size=(frequency, time_bins), mode="bilinear", align_corners=False
            )
            repeated_up = up[:, None].expand(-1, self.ifo_count, -1, -1, -1)
            decoded = self.shared_decoder(
                torch.cat([encoded, repeated_up], dim=2).reshape(
                    batch * self.ifo_count,
                    self.base_channels * 3,
                    frequency,
                    time_bins,
                )
            )
            logits = self.shared_head(decoded).reshape(
                batch,
                self.ifo_count,
                2,
                self.q_count,
                frequency,
                time_bins,
            )
            logits = logits.permute(0, 2, 1, 3, 4, 5)
            logits = torch.where(
                availability[:, None, :, None, None, None].to(dtype=torch.bool),
                logits,
                torch.full_like(logits, -20.0),
            )
            return logits.reshape(batch, 2, self.input_channels, frequency, time_bins)


    class GlitchEmbeddingNet(nn.Module):
        """Shared single-IFO Q encoder for known-family attribution and OOD scoring."""

        def __init__(
            self,
            q_count: int,
            class_count: int,
            base_channels: int = 24,
            embedding_dim: int = 32,
        ):
            super().__init__()
            if q_count < 1 or class_count < 2 or embedding_dim < 2:
                raise ValueError("glitch embedding dimensions are invalid")
            self.q_count = int(q_count)
            self.class_count = int(class_count)
            self.embedding_dim = int(embedding_dim)
            self.encoder = _ConvBlock(self.q_count, base_channels)
            self.projection = nn.Linear(2 * base_channels, self.embedding_dim)
            self.classifier = nn.Linear(self.embedding_dim, self.class_count)

        def forward(self, value: Any) -> tuple[Any, Any]:
            if value.ndim != 4 or value.shape[1] != self.q_count:
                raise ValueError("glitch embedding input must have shape [batch, Q, F, T]")
            encoded = self.encoder(value)
            pooled = torch.cat(
                [encoded.mean(dim=(2, 3)), encoded.amax(dim=(2, 3))], dim=1
            )
            embedding = torch_functional.normalize(
                self.projection(pooled), p=2, dim=1
            )
            return self.classifier(embedding), embedding


    class CoalescenceTimingNet(nn.Module):
        """Candidate timing refiner with a mask-compatible convolutional backbone."""

        def __init__(self, input_channels: int, base_channels: int = 24):
            super().__init__()
            self.backbone = MultiIFOQNet(input_channels, base_channels)
            self.backbone.head.requires_grad_(False)
            groups = min(8, base_channels)
            self.timing_head = nn.Sequential(
                nn.Conv1d(base_channels, base_channels, 5, padding=2, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(base_channels, 1, 1),
            )

        def forward(self, value: Any) -> Any:
            decoded = self.backbone.decoded_features(value)
            temporal = torch.amax(decoded, dim=2)
            return self.timing_head(temporal)[:, 0]


    class DetectorArrivalTimingNet(nn.Module):
        """Shared time-domain encoder with explicit detector-set fusion and per-IFO timing."""

        def __init__(self, detector_count: int, base_channels: int = 32):
            super().__init__()
            if detector_count < 2 or base_channels < 4:
                raise ValueError("arrival timing network dimensions are invalid")
            self.detector_count = detector_count
            groups = min(8, base_channels)
            self.encoder = nn.Sequential(
                nn.Conv1d(1, base_channels, 17, stride=4, padding=8, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(
                    base_channels, base_channels, 9, stride=2, padding=4, bias=False
                ),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(base_channels, base_channels, 7, padding=3, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
            )
            self.head = nn.Sequential(
                nn.Conv1d(2 * base_channels, base_channels, 5, padding=2, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(base_channels, 1, 1),
            )

        def forward(self, value: Any, availability: Any) -> Any:
            if value.ndim != 3 or value.shape[1] != self.detector_count:
                raise ValueError("arrival timing strain must have shape [batch, IFO, time]")
            if availability.shape != value.shape[:2]:
                raise ValueError("arrival timing availability shape differs from strain")
            batch, detectors, samples = value.shape
            encoded = self.encoder(value.reshape(batch * detectors, 1, samples))
            encoded = encoded.reshape(batch, detectors, encoded.shape[1], encoded.shape[2])
            mask = availability.to(encoded.dtype)[:, :, None, None]
            if torch.any(mask.sum(dim=1) < 1):
                raise ValueError("arrival timing batch contains no available detector")
            network = (encoded * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            )
            network = network.expand(-1, detectors, -1, -1)
            fused = torch.cat([encoded, network], dim=2).reshape(
                batch * detectors, 2 * encoded.shape[2], encoded.shape[3]
            )
            logits = self.head(fused).reshape(batch, detectors, -1)
            return logits.masked_fill(~availability.to(torch.bool)[:, :, None], -torch.inf)


    class _DilatedResidual1D(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            groups = min(8, channels)
            self.layers = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(groups, channels),
                nn.SiLU(),
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.GroupNorm(groups, channels),
            )

        def forward(self, value: Any) -> Any:
            return torch_functional.silu(value + self.layers(value))


    class DetectorArrivalTimingContextNet(nn.Module):
        """Long-context timing head spanning the inspiral while preserving 7.8 ms bins."""

        def __init__(self, detector_count: int, base_channels: int = 32):
            super().__init__()
            if detector_count < 2 or base_channels < 4:
                raise ValueError("arrival timing context network dimensions are invalid")
            self.detector_count = detector_count
            groups = min(8, base_channels)
            self.encoder = nn.Sequential(
                nn.Conv1d(1, base_channels, 17, stride=4, padding=8, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(
                    base_channels, base_channels, 9, stride=2, padding=4, bias=False
                ),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
            )
            self.long_context = nn.Sequential(
                *(
                    _DilatedResidual1D(base_channels, dilation)
                    for dilation in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                )
            )
            self.head = nn.Sequential(
                nn.Conv1d(2 * base_channels, base_channels, 5, padding=2, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(base_channels, 1, 1),
            )

        def forward(self, value: Any, availability: Any) -> Any:
            if value.ndim != 3 or value.shape[1] != self.detector_count:
                raise ValueError("arrival timing strain must have shape [batch, IFO, time]")
            if availability.shape != value.shape[:2]:
                raise ValueError("arrival timing availability shape differs from strain")
            batch, detectors, samples = value.shape
            encoded = self.long_context(
                self.encoder(value.reshape(batch * detectors, 1, samples))
            )
            encoded = encoded.reshape(batch, detectors, encoded.shape[1], encoded.shape[2])
            mask = availability.to(encoded.dtype)[:, :, None, None]
            if torch.any(mask.sum(dim=1) < 1):
                raise ValueError("arrival timing batch contains no available detector")
            network = (encoded * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            )
            fused = torch.cat(
                [encoded, network.expand(-1, detectors, -1, -1)], dim=2
            ).reshape(batch * detectors, 2 * encoded.shape[2], encoded.shape[3])
            logits = self.head(fused).reshape(batch, detectors, -1)
            return logits.masked_fill(~availability.to(torch.bool)[:, :, None], -torch.inf)


    class DetectorArrivalSpectrogramNet(nn.Module):
        """High-time-resolution numeric spectrogram head for chirp endpoint timing."""

        def __init__(self, detector_count: int, base_channels: int = 32):
            super().__init__()
            if detector_count < 2 or base_channels < 4:
                raise ValueError("arrival spectrogram network dimensions are invalid")
            self.detector_count = detector_count
            self.n_fft = 256
            self.hop_length = 8
            self.output_bins = 1024
            self.register_buffer("stft_window", torch.hann_window(self.n_fft))
            groups = min(8, base_channels)
            self.spectral_encoder = nn.Sequential(
                nn.Conv2d(
                    1,
                    base_channels,
                    (9, 5),
                    stride=(2, 1),
                    padding=(4, 2),
                    bias=False,
                ),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    (7, 5),
                    padding=(3, 2),
                    groups=base_channels,
                    bias=False,
                ),
                nn.Conv2d(base_channels, base_channels, 1, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    (5, 5),
                    padding=(2, 2),
                    groups=base_channels,
                    bias=False,
                ),
                nn.Conv2d(base_channels, base_channels, 1, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
            )
            self.frequency_projection = nn.Sequential(
                nn.Conv1d(2 * base_channels, base_channels, 1, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
            )
            self.temporal_context = nn.Sequential(
                *(
                    _DilatedResidual1D(base_channels, dilation)
                    for dilation in (1, 2, 4, 8)
                )
            )
            self.head = nn.Sequential(
                nn.Conv1d(2 * base_channels, base_channels, 5, padding=2, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(base_channels, 1, 1),
            )

        def forward(self, value: Any, availability: Any) -> Any:
            if value.ndim != 3 or value.shape[1] != self.detector_count:
                raise ValueError("arrival timing strain must have shape [batch, IFO, time]")
            if availability.shape != value.shape[:2]:
                raise ValueError("arrival timing availability shape differs from strain")
            batch, detectors, samples = value.shape
            flat = value.reshape(batch * detectors, samples)
            spectrum = torch.stft(
                flat,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.stft_window,
                center=True,
                return_complex=True,
            )
            if spectrum.shape[-1] < self.output_bins:
                raise ValueError("arrival spectrogram has too few temporal bins")
            power = torch.log1p(spectrum[..., : self.output_bins].abs().square())
            encoded_2d = self.spectral_encoder(power[:, None])
            frequency_pooled = torch.cat(
                [encoded_2d.mean(dim=2), encoded_2d.amax(dim=2)], dim=1
            )
            encoded = self.temporal_context(
                self.frequency_projection(frequency_pooled)
            ).reshape(batch, detectors, -1, self.output_bins)
            mask = availability.to(encoded.dtype)[:, :, None, None]
            if torch.any(mask.sum(dim=1) < 1):
                raise ValueError("arrival timing batch contains no available detector")
            network = (encoded * mask).sum(dim=1, keepdim=True) / mask.sum(
                dim=1, keepdim=True
            )
            fused = torch.cat(
                [encoded, network.expand(-1, detectors, -1, -1)], dim=2
            ).reshape(batch * detectors, 2 * encoded.shape[2], self.output_bins)
            logits = self.head(fused).reshape(batch, detectors, self.output_bins)
            return logits.masked_fill(~availability.to(torch.bool)[:, :, None], -torch.inf)


    class CandidateLocalSpectrogramRefiner(nn.Module):
        """Score and time every local candidate with aligned detector context."""

        def __init__(
            self,
            detector_count: int,
            output_bins: int = 640,
            base_channels: int = 16,
        ):
            super().__init__()
            if detector_count < 2 or output_bins < 32 or base_channels < 4:
                raise ValueError("candidate local refiner dimensions are invalid")
            self.detector_count = int(detector_count)
            self.output_bins = int(output_bins)
            self.n_fft = 128
            self.hop_length = 4
            self.register_buffer("stft_window", torch.hann_window(self.n_fft))
            groups = min(8, base_channels)
            self.spectral_encoder = nn.Sequential(
                nn.Conv2d(
                    1,
                    base_channels,
                    (7, 5),
                    stride=(2, 1),
                    padding=(3, 2),
                    bias=False,
                ),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    (5, 5),
                    padding=(2, 2),
                    groups=base_channels,
                    bias=False,
                ),
                nn.Conv2d(base_channels, base_channels, 1, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
            )
            self.frequency_projection = nn.Sequential(
                nn.Conv1d(2 * base_channels, base_channels, 1, bias=False),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
            )
            self.temporal_context = nn.Sequential(
                *(
                    _DilatedResidual1D(2 * base_channels, dilation)
                    for dilation in (1, 2, 4, 8, 16)
                )
            )
            self.timing_head = nn.Sequential(
                nn.Conv1d(
                    2 * base_channels, base_channels, 5, padding=2, bias=False
                ),
                nn.GroupNorm(groups, base_channels),
                nn.SiLU(),
                nn.Conv1d(base_channels, 1, 1),
            )
            self.presence_head = nn.Sequential(
                nn.Linear(4 * base_channels, 2 * base_channels),
                nn.SiLU(),
                nn.Linear(2 * base_channels, 1),
            )

        def forward(
            self, value: Any, availability: Any, candidate_ifo_index: Any
        ) -> tuple[Any, Any]:
            if value.ndim != 3 or value.shape[1] != self.detector_count:
                raise ValueError("candidate refiner strain must be [batch, IFO, time]")
            if availability.shape != value.shape[:2]:
                raise ValueError("candidate refiner availability differs from strain")
            if candidate_ifo_index.shape != (value.shape[0],):
                raise ValueError("candidate refiner IFO index must be [batch]")
            batch, detectors, samples = value.shape
            if torch.any(candidate_ifo_index < 0) or torch.any(
                candidate_ifo_index >= detectors
            ):
                raise ValueError("candidate refiner IFO index is outside model slots")
            candidate_available = availability.to(torch.bool).gather(
                1, candidate_ifo_index[:, None]
            )
            if torch.any(~candidate_available):
                raise ValueError("candidate refiner proposal uses an unavailable detector")
            spectrum = torch.stft(
                value.reshape(batch * detectors, samples),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.stft_window,
                center=True,
                return_complex=True,
            )
            if spectrum.shape[-1] < self.output_bins:
                raise ValueError("candidate refiner crop has too few time bins")
            power = torch.log1p(spectrum[..., : self.output_bins].abs().square())
            encoded_2d = self.spectral_encoder(power[:, None])
            encoded = self.frequency_projection(
                torch.cat(
                    [encoded_2d.mean(dim=2), encoded_2d.amax(dim=2)], dim=1
                )
            ).reshape(batch, detectors, -1, self.output_bins)
            mask = availability.to(encoded.dtype)[:, :, None, None]
            if torch.any(mask.sum(dim=1) < 1):
                raise ValueError("candidate refiner batch has no available detector")
            network = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
            batch_indices = torch.arange(batch, device=value.device)
            local = encoded[batch_indices, candidate_ifo_index]
            fused = self.temporal_context(torch.cat([local, network], dim=1))
            timing = self.timing_head(fused)[:, 0]
            pooled = torch.cat([fused.mean(dim=2), fused.amax(dim=2)], dim=1)
            presence = self.presence_head(pooled)[:, 0]
            return presence, timing


    class CandidateEndpointWarmRefiner(nn.Module):
        """Local refiner whose timing backbone is compatible with the dense endpoint arm."""

        def __init__(
            self,
            detector_count: int,
            output_bins: int = 640,
            base_channels: int = 16,
        ):
            super().__init__()
            self.detector_count = int(detector_count)
            self.output_bins = int(output_bins)
            self.n_fft = 256
            self.hop_length = 4
            self.register_buffer("stft_window", torch.hann_window(self.n_fft))
            endpoint = DetectorArrivalSpectrogramNet(detector_count, base_channels)
            self.spectral_encoder = endpoint.spectral_encoder
            self.frequency_projection = endpoint.frequency_projection
            self.temporal_context = endpoint.temporal_context
            self.timing_head = endpoint.head
            self.presence_head = nn.Sequential(
                nn.Linear(4 * base_channels, 2 * base_channels),
                nn.SiLU(),
                nn.Linear(2 * base_channels, 1),
            )

        def load_endpoint_backbone(self, endpoint_state: dict[str, Any]) -> None:
            source = DetectorArrivalSpectrogramNet(
                self.detector_count, self.timing_head[0].in_channels // 2
            )
            source.load_state_dict(endpoint_state)
            self.spectral_encoder.load_state_dict(source.spectral_encoder.state_dict())
            self.frequency_projection.load_state_dict(
                source.frequency_projection.state_dict()
            )
            self.temporal_context.load_state_dict(source.temporal_context.state_dict())
            self.timing_head.load_state_dict(source.head.state_dict())

        def forward(
            self, value: Any, availability: Any, candidate_ifo_index: Any
        ) -> tuple[Any, Any]:
            if value.ndim != 3 or value.shape[1] != self.detector_count:
                raise ValueError("candidate refiner strain must be [batch, IFO, time]")
            if availability.shape != value.shape[:2]:
                raise ValueError("candidate refiner availability differs from strain")
            if candidate_ifo_index.shape != (value.shape[0],):
                raise ValueError("candidate refiner IFO index must be [batch]")
            batch, detectors, samples = value.shape
            if torch.any(candidate_ifo_index < 0) or torch.any(
                candidate_ifo_index >= detectors
            ):
                raise ValueError("candidate refiner IFO index is outside model slots")
            candidate_available = availability.to(torch.bool).gather(
                1, candidate_ifo_index[:, None]
            )
            if torch.any(~candidate_available):
                raise ValueError("candidate refiner proposal uses an unavailable detector")
            spectrum = torch.stft(
                value.reshape(batch * detectors, samples),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.stft_window,
                center=True,
                return_complex=True,
            )
            if spectrum.shape[-1] < self.output_bins:
                raise ValueError("candidate refiner crop has too few time bins")
            power = torch.log1p(spectrum[..., : self.output_bins].abs().square())
            encoded_2d = self.spectral_encoder(power[:, None])
            encoded = self.temporal_context(
                self.frequency_projection(
                    torch.cat(
                        [encoded_2d.mean(dim=2), encoded_2d.amax(dim=2)], dim=1
                    )
                )
            ).reshape(batch, detectors, -1, self.output_bins)
            mask = availability.to(encoded.dtype)[:, :, None, None]
            if torch.any(mask.sum(dim=1) < 1):
                raise ValueError("candidate refiner batch has no available detector")
            network = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
            batch_indices = torch.arange(batch, device=value.device)
            local = encoded[batch_indices, candidate_ifo_index]
            fused = torch.cat([local, network], dim=1)
            timing = self.timing_head(fused)[:, 0]
            pooled = torch.cat([fused.mean(dim=2), fused.amax(dim=2)], dim=1)
            presence = self.presence_head(pooled)[:, 0]
            return presence, timing

else:

    class MultiIFOQNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class CoalescenceTimingNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class DetectorSetQNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class DetectorArrivalTimingNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class DetectorArrivalTimingContextNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class DetectorArrivalSpectrogramNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class CandidateLocalSpectrogramRefiner:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class CandidateEndpointWarmRefiner:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()

    class GlitchEmbeddingNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any):
            _require_torch()


def initialize_detector_set_from_early_fusion(
    model: Any,
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    """Warm-start a detector-set arm from a fixed-channel MultiIFOQNet state."""
    _require_torch()
    source = checkpoint_state.get("model", checkpoint_state)
    if not isinstance(source, dict):
        raise ValueError("early-fusion checkpoint state must be a state dictionary")
    expected_channels = model.ifo_count * model.q_count
    first = source.get("encoder.layers.0.weight")
    head_weight = source.get("head.weight")
    head_bias = source.get("head.bias")
    if first is None or tuple(first.shape[1:2]) != (expected_channels,):
        raise ValueError("early-fusion encoder channel count differs from detector-set model")
    if head_weight is None or head_bias is None or head_weight.shape[0] != 2 * expected_channels:
        raise ValueError("early-fusion head shape differs from detector-set model")
    target = model.state_dict()
    mapped = []
    first_reshaped = first.reshape(
        first.shape[0], model.ifo_count, model.q_count, *first.shape[2:]
    )
    target["shared_encoder.layers.0.weight"] = first_reshaped.mean(dim=1)
    mapped.append("encoder first convolution averaged across configured IFO slots")
    for suffix in (
        "layers.1.weight",
        "layers.1.bias",
        "layers.3.weight",
        "layers.4.weight",
        "layers.4.bias",
    ):
        target[f"shared_encoder.{suffix}"] = source[f"encoder.{suffix}"].clone()
    for prefix in ("bottleneck",):
        for key, value in source.items():
            if key.startswith(f"{prefix}."):
                target[key] = value.clone()
    for key, value in source.items():
        if key.startswith("decoder."):
            target[f"shared_decoder.{key[len('decoder.'):]}"] = value.clone()
    target["shared_head.weight"] = head_weight.reshape(
        2, model.ifo_count, model.q_count, *head_weight.shape[1:]
    ).mean(dim=1).reshape(2 * model.q_count, *head_weight.shape[1:])
    target["shared_head.bias"] = head_bias.reshape(
        2, model.ifo_count, model.q_count
    ).mean(dim=1).reshape(2 * model.q_count)
    target["attention_score.weight"].zero_()
    target["attention_score.bias"].zero_()
    model.load_state_dict(target)
    return {
        "status": "detector_set_warm_start",
        "source_architecture": "MultiIFOQNet",
        "target_architecture": "DetectorSetQNet",
        "ifo_count": model.ifo_count,
        "q_count": model.q_count,
        "input_channels": expected_channels,
        "mapping": mapped
        + [
            "shared encoder/bottleneck/decoder copied where shapes match",
            "class/Q head averaged across configured IFO slots",
            "set-attention logits initialized uniformly",
        ],
    }


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    model_ifos: tuple[str, ...],
    q_values: tuple[float, ...],
) -> tuple[Any, str]:
    """Construct a mask model without silently changing its detector contract."""
    _require_torch()
    architecture = str(checkpoint.get("architecture", "fixed_channel"))
    expected_channels = len(model_ifos) * len(q_values)
    if int(checkpoint["input_channels"]) != expected_channels:
        raise ValueError(
            f"checkpoint has {checkpoint['input_channels']} channels; "
            f"requested detector/Q contract requires {expected_channels}"
        )
    if "model_ifos" in checkpoint and tuple(checkpoint["model_ifos"]) != model_ifos:
        raise ValueError("checkpoint detector ordering differs from the requested model_ifos")
    if "q_values" in checkpoint and tuple(float(x) for x in checkpoint["q_values"]) != q_values:
        raise ValueError("checkpoint Q ordering differs from the requested q_values")
    base_channels = int(checkpoint["base_channels"])
    if architecture == "fixed_channel":
        model = MultiIFOQNet(expected_channels, base_channels)
    elif architecture == "detector_set":
        model = DetectorSetQNet(len(model_ifos), len(q_values), base_channels)
    else:
        raise ValueError(f"unsupported checkpoint architecture: {architecture}")
    model.load_state_dict(checkpoint["model"])
    return model, architecture


def _dice_loss(logits: Any, targets: Any, class_weights: Any | None = None) -> Any:
    probabilities = torch.sigmoid(logits)
    axes = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + targets.sum(dim=axes)
    losses = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    if class_weights is None:
        return losses.mean()
    weights = class_weights.reshape(1, -1)
    return (losses * weights).sum() / (weights.sum() * losses.shape[0])


def _batch_counts(
    logits: Any,
    targets: Any,
    thresholds: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    threshold_tensor = torch.as_tensor(thresholds, device=logits.device).reshape(1, 2, 1, 1, 1)
    predicted = torch.sigmoid(logits) >= threshold_tensor
    expected = targets >= 0.5
    axes = tuple(range(2, predicted.ndim))
    true_positive = (predicted & expected).sum(dim=axes)
    false_positive = (predicted & ~expected).sum(dim=axes)
    false_negative = (~predicted & expected).sum(dim=axes)
    return torch.stack([true_positive, false_positive, false_negative], dim=-1).sum(dim=0).cpu().numpy()


def _metrics_from_counts(counts: np.ndarray) -> dict[str, Any]:
    class_names = ("chirp", "glitch")
    result: dict[str, Any] = {}
    ious = []
    for index, name in enumerate(class_names):
        true_positive, false_positive, false_negative = (float(item) for item in counts[index])
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        iou = true_positive / max(true_positive + false_positive + false_negative, 1.0)
        dice = 2.0 * true_positive / max(2.0 * true_positive + false_positive + false_negative, 1.0)
        result[name] = {"precision": precision, "recall": recall, "iou": iou, "dice": dice}
        ious.append(iou)
    result["mean_iou"] = float(np.mean(ious))
    return result


def _run_epoch(
    model: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    positive_weights: tuple[float, float],
    dice_weights: tuple[float, float],
    thresholds: tuple[float, float] = (0.5, 0.5),
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    counts = np.zeros((2, 3), dtype=np.int64)
    positive_weight_tensor = torch.as_tensor(positive_weights, device=device).reshape(2, 1, 1, 1)
    dice_weight_tensor = torch.as_tensor(dice_weights, device=device)
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            bce = torch_functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weight_tensor
            )
            loss = bce + _dice_loss(logits, targets, dice_weight_tensor)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
        counts += _batch_counts(logits.detach(), targets, thresholds)
    return {
        "loss": total_loss / max(batches, 1),
        **_metrics_from_counts(counts),
    }


def _calibrate_thresholds(
    model: Any,
    loader: Any,
    device: Any,
    grid: tuple[float, ...],
) -> tuple[tuple[float, float], dict[str, Any]]:
    model.eval()
    logits_batches = []
    target_batches = []
    with torch.no_grad():
        for features, targets in loader:
            logits_batches.append(model(features.to(device)).cpu())
            target_batches.append(targets.cpu())
    logits = torch.cat(logits_batches)
    targets = torch.cat(target_batches)
    selected = []
    curves: dict[str, Any] = {"chirp": [], "glitch": []}
    for class_index, class_name in enumerate(("chirp", "glitch")):
        best_threshold = grid[0]
        best_iou = -1.0
        for threshold in grid:
            thresholds = (threshold, 1.0) if class_index == 0 else (1.0, threshold)
            counts = _batch_counts(logits, targets, thresholds)
            class_metrics = _metrics_from_counts(counts)[class_name]
            curves[class_name].append({"threshold": threshold, **class_metrics})
            if float(class_metrics["iou"]) > best_iou:
                best_iou = float(class_metrics["iou"])
                best_threshold = threshold
        selected.append(best_threshold)
    return (float(selected[0]), float(selected[1])), curves


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def train_numeric_model(
    config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    seed_override: int | None = None,
) -> dict[str, Any]:
    _require_torch()
    config = load_yaml(config_path)
    settings = config["numeric_training"]
    if seed_override is not None:
        settings["seed"] = int(seed_override)
    seed = int(settings.get("seed", 0))
    family_config = deepcopy(config)
    family_config["numeric_training"].pop("seed", None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    recipes = read_recipe_manifest(manifest_path)
    by_split = {split: [recipe for recipe in recipes if recipe.split == split] for split in ("train", "val", "test")}
    if any(not items for items in by_split.values()):
        raise ValueError(f"Manifest must contain non-empty train/val/test splits: { {key: len(value) for key, value in by_split.items()} }")
    tensor_config = settings["tensor"]
    batch_size = int(settings.get("batch_size", 8))
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            NumericRecipeDataset(
                items,
                tensor_config,
                cache_in_memory=bool(settings.get("cache_in_memory", False)),
            ),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=int(settings.get("workers", 0)),
            generator=generator if split == "train" else None,
        )
        for split, items in by_split.items()
    }
    first_recipe = recipes[0]
    input_channels = len(first_recipe.ifos) * len(first_recipe.q_values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiIFOQNet(input_channels, int(settings.get("base_channels", 24))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_numeric.pt"
    history = []
    positive_weights = tuple(float(item) for item in settings.get("positive_weights", [1.0, 1.0]))
    dice_weights = tuple(float(item) for item in settings.get("dice_weights", [1.0, 1.0]))
    if len(positive_weights) != 2 or len(dice_weights) != 2:
        raise ValueError("positive_weights and dice_weights require [chirp, glitch]")
    best_metric = -1.0
    best_epoch = None
    started = time.time()
    for epoch in range(1, int(settings.get("epochs", 10)) + 1):
        train_metrics = _run_epoch(
            model, loaders["train"], device, optimizer, positive_weights, dice_weights
        )
        validation_metrics = _run_epoch(
            model, loaders["val"], device, None, positive_weights, dice_weights
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if float(validation_metrics["mean_iou"]) > best_metric:
            best_metric = float(validation_metrics["mean_iou"])
            best_epoch = epoch
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_mean_iou": best_metric,
                    "input_channels": input_channels,
                    "base_channels": int(settings.get("base_channels", 24)),
                    "config_hash": canonical_hash(config),
                    "manifest_sha256": file_sha256(manifest_path),
                    "seed": seed,
                },
            )
        atomic_write_json(output / "history.json", history)

    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    threshold_grid = tuple(float(item) for item in settings.get("threshold_grid", [0.5]))
    thresholds, threshold_curves = _calibrate_thresholds(
        model, loaders["val"], device, threshold_grid
    )
    calibrated_validation_metrics = _run_epoch(
        model,
        loaders["val"],
        device,
        None,
        positive_weights,
        dice_weights,
        thresholds,
    )
    evaluate_test = bool(settings.get("evaluate_test", False))
    test_metrics = None
    if evaluate_test:
        test_metrics = _run_epoch(
            model,
            loaders["test"],
            device,
            None,
            positive_weights,
            dice_weights,
            thresholds,
        )
    report = {
        "status": "synthetic_engineering_baseline",
        "scientific_claim_allowed": False,
        "device": str(device),
        "seed": seed,
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        "config_family_hash": canonical_hash(family_config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "split_counts": {key: len(value) for key, value in by_split.items()},
        "cache_in_memory": bool(settings.get("cache_in_memory", False)),
        "best_epoch": best_epoch,
        "best_validation_mean_iou": best_metric,
        "positive_weights": positive_weights,
        "dice_weights": dice_weights,
        "validation_selected_thresholds": {"chirp": thresholds[0], "glitch": thresholds[1]},
        "threshold_curves": threshold_curves,
        "calibrated_validation": calibrated_validation_metrics,
        "test_evaluated": evaluate_test,
        "test": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    atomic_write_json(output / "numeric_training_report.json", report)
    return report


def evaluate_numeric_checkpoint(
    config_path: str | Path,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    thresholds: tuple[float, float],
    output_path: str | Path,
) -> dict[str, Any]:
    _require_torch()
    if split not in {"val", "test"}:
        raise ValueError("numeric evaluation split must be val or test")
    config = load_yaml(config_path)
    settings = config["numeric_training"]
    recipes = [recipe for recipe in read_recipe_manifest(manifest_path) if recipe.split == split]
    if not recipes:
        raise ValueError(f"No recipes found for split {split}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiIFOQNet(
        int(checkpoint["input_channels"]), int(checkpoint["base_channels"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        NumericRecipeDataset(recipes, settings["tensor"]),
        batch_size=int(settings.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(settings.get("workers", 0)),
    )
    positive_weights = tuple(float(item) for item in settings.get("positive_weights", [1.0, 1.0]))
    dice_weights = tuple(float(item) for item in settings.get("dice_weights", [1.0, 1.0]))
    metrics = _run_epoch(
        model,
        loader,
        device,
        None,
        positive_weights,
        dice_weights,
        thresholds,
    )
    report = {
        "status": "synthetic_engineering_baseline",
        "scientific_claim_allowed": False,
        "split": split,
        "scene_count": len(recipes),
        "thresholds": {"chirp": thresholds[0], "glitch": thresholds[1]},
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "manifest_sha256": file_sha256(manifest_path),
        "config_hash": canonical_hash(config),
    }
    atomic_write_json(output_path, report)
    return report
