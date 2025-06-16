import torch
import librosa
import gradio as gr
from inference import WhisperInferencer
from tqdm import tqdm
import tempfile
import json
import os
import ffmpeg
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = ""

EMOTION_LABELS = ['neutral', 'angry', 'positive', 'sad', 'other']

class AudioConversionError(Exception):
    """If conversion to WAV failed."""
    pass

def prepare_audio(file_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        (
            ffmpeg
            .input(file_path)
            .output(wav_path,
                    format='wav',
                    acodec='pcm_s16le',  
                    ar='16000',          
                    ac=1)                
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        try: os.remove(wav_path)
        except OSError: pass

        stderr = e.stderr.decode('utf-8', errors='ignore')
        raise AudioConversionError(f"Failed to convert file:\n{stderr}") from e

    return wav_path

def analyze(file_path: str, min_spk: int, max_spk: int):
    try:
        audio_path = prepare_audio(file_path=file_path)
    except AudioConversionError as e:
        raise gr.Error(str(e))

    transcriber = WhisperInferencer(model_name='openai/whisper-large-v3-turbo', 
                                device=device,
                                language="Russian",
                                task="transcribe",
                                hf_token=HF_TOKEN,
                                min_speakers=min_spk,
                                max_speakers=max_spk if max_spk > 0 else 1)
    
    segments = transcriber.inference_transcription(audio_path)

    waveform, sr = librosa.load(audio_path, sr=16000)

    audio_emotion_model = WhisperInferencer(model_name='nixiieee/whisper-small-emotion-classifier-dusha',
                                device=device,
                                language="Russian",
                                task="transcribe")

    records = []
    for seg in tqdm(segments):
        start, end, text = seg['start'], seg['end'], seg['text']
        s_sample = int(start * sr)
        e_sample = int(end * sr)
        wave_seg = waveform[s_sample:e_sample]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            wav_tmp_path = tmp.name
            sf.write(wav_tmp_path, wave_seg, samplerate=sr)
            emo, score = audio_emotion_model.predict_emotion(wav_tmp_path)
        records.append({
            'start': start,
            'end': end,
            'speaker': seg['speaker'],
            'text': text,
            'emotion_name': EMOTION_LABELS[emo],
            'emotion_confidence_score': score
        })

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False ) as tmp:
        json.dump(records, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    os.remove(audio_path)
    return records, tmp_path

def main():
    description = (
        "Загрузите аудио или видеофайл.\n"
    )

    iface = gr.Interface(
        fn=analyze,
        inputs=[
            gr.File(type="filepath", label="Аудио (wav/mp3) или видео (mp4)"),
            gr.Number(value=0, label="Минимальное число спикеров", precision=0),
            gr.Number(value=1, label="Максимальное число спикеров", precision=0),
        ],
        outputs=[
            gr.JSON(label="Транскрипция"),
            gr.File(label="Скачать JSON")
        ],
        title="Транскрипция и детекция эмоций аудио- и видеофайлов",
        description=description
    )

    iface.launch()

if __name__ == "__main__":
    main()
