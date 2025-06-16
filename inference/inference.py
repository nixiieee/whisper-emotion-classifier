import os
import re
import logging
from typing import Union, List, Tuple, Dict, Generator, Optional
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForAudioClassification,
)
from pyannote.audio import Pipeline

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

import gigaam

# Configure module-level logger
logger = logging.getLogger(__name__)

# Pre-compile stop-phrase regex patterns
def _build_stop_patterns(stop_list: List[str]) -> List[re.Pattern]:
    return [re.compile(re.escape(p), flags=re.IGNORECASE) for p in stop_list]

stop_phrases = [
    'Редактор субтитров А.Семкин Корректор А.Егорова',
    'Субтитры добавил DimaTorzok',
    'Аллах Акбар, Иблис Аллах',
    'Редактор субтитров А.Синецкая Корректор А.Егорова',
    'Субтитры делал DimaTorzok',
    'ПОКА!',
    'Продолжение следует...',
    'Спасибо за просмотр!',
    'Удачи!'
]
_stop_patterns = _build_stop_patterns(stop_phrases)

def remove_stop_phrases(text: str) -> str:
    """Remove configured stop phrases from a given text."""
    for pat in _stop_patterns:
        text = pat.sub('', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file, resample if needed, and return a mono waveform tensor.

    Args:
        path: Path to audio file.
        target_sr: Desired sampling rate.

    Returns:
        Tensor of shape (1, N) at target_sr.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def chunk_audio(
    audio: torch.Tensor,
    chunk_length_s: float,
    sr: int = 16000
) -> Generator[torch.Tensor, None, None]:
    """Yield fixed-length chunks (seconds) of an audio tensor."""
    chunk_size = int(chunk_length_s * sr)
    total_len = audio.size(1)
    for start in range(0, total_len, chunk_size):
        yield audio[:, start: start + chunk_size]


class WhisperInferencer:
    """
    Wrapper for Whisper ASR with speaker diarization and optional emotion classification.
    """
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: Union[str, torch.device] = "cpu",
        language: str = "Russian",
        task: str = "transcribe",
        chunk_length_s: float = 30.0,
        vad_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: str = "",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Diarization parameters
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Initialize ASR model and diarization
        try:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="sdpa"
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{model_name}': {e}")

        try:
            self.vad_pipeline = Pipeline.from_pretrained(vad_name, use_auth_token=hf_token)
            self.vad_pipeline.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization pipeline '{vad_name}': {e}")

        self.model.eval()

        self.processor = WhisperProcessor.from_pretrained(
            model_name, language=language, task=task
        )
        self.chunk_length_s = chunk_length_s
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )

    def _run_diarization(self, audio: torch.Tensor) -> List[Tuple[float, float, str]]:
        """Run speaker diarization with optional min/max speaker constraints."""
        sr = self.processor.feature_extractor.sampling_rate
        params = {'waveform': audio, 'sample_rate': sr}
        kwargs = {}
        if self.min_speakers is not None:
            kwargs['min_speakers'] = self.min_speakers
        if self.max_speakers is not None:
            kwargs['max_speakers'] = self.max_speakers
        diarization = self.vad_pipeline(params, **kwargs)
        return [(seg.start, seg.end, label)
                for seg, _, label in diarization.itertracks(yield_label=True)]

    def _decode_segment(self, segment: torch.Tensor) -> str:
        """Decode an audio segment to text, removing stop phrases."""
        sr = self.processor.feature_extractor.sampling_rate
        features = self.processor.feature_extractor(
            segment.squeeze(0).numpy(), sampling_rate=sr, return_tensors="pt"
        ).input_features.half().to(self.device)

        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type):
            with torch.no_grad():
                gen_ids = self.model.generate(
                    features,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    forced_decoder_ids=self.forced_decoder_ids,
                )
                raw = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return remove_stop_phrases(raw)

    def transcribe(self, audio: torch.Tensor) -> List[Dict]:
        """Transcribe audio tensor into speaker-labeled segments."""
        logger.debug("Audio shape: %s, mean abs: %.6f", audio.shape, audio.abs().mean().item())

        segments = self._run_diarization(audio)
        results = []
        prev = {'start': None, 'end': None, 'speaker': None, 'text': ""}

        for start, end, speaker in tqdm(segments, desc="Diarization segments", unit="segment"):
            seg_audio = audio[:, int(start * self.processor.feature_extractor.sampling_rate):
                                   int(end * self.processor.feature_extractor.sampling_rate)]
            text = self._decode_segment(seg_audio)
            if not text:
                continue

            can_merge = (
                prev['speaker'] == speaker or
                (prev['end'] is not None and start - prev['end'] < 0.001)
            ) and (prev['start'] is not None and (end - prev['start'] <= self.chunk_length_s))

            if prev['start'] is None:
                prev.update(start=start, end=end, speaker=speaker, text=text)
            elif can_merge:
                prev['text'] += ' ' + text
                prev['end'] = max(prev['end'], end)
            else:
                results.append(prev.copy())
                prev = {'start': start, 'end': end, 'speaker': speaker, 'text': text}

        # Append last segment
        if prev['start'] is not None and prev['text']:
            results.append(prev)

        # Free GPU cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return results

    def predict_emotion(self, audio: torch.Tensor) -> Tuple[int, float]:
        """Predict emotion index and confidence from audio tensor."""
        sr = self.processor.feature_extractor.sampling_rate
        features = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_features=features)
            probs = F.softmax(outputs.logits, dim=-1)
        idx = int(torch.argmax(probs, dim=-1).item())
        return idx, float(probs[0, idx].item())

    def inference_transcription(
        self,
        inputs: Union[str, List[str]]
    ) -> Union[List[Dict], Dict]:
        """
        Transcribe one or multiple audio file paths.

        Args:
            inputs: File path or list of file paths.
        Returns:
            Single transcription dict or list thereof.
        """
        single = False
        if isinstance(inputs, str):
            inputs = [inputs]
            single = True

        all_results = []
        for path in inputs:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Audio file not found: {path}")
            audio = load_audio(path, target_sr=self.processor.feature_extractor.sampling_rate)
            all_results.append(self.transcribe(audio))

        return all_results[0] if single else all_results
