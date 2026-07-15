import os
import json
import wave
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "ro")
MODEL_ONNX = os.path.join(MODEL_DIR, "ro_RO-mihai-medium.onnx")
MODEL_JSON = os.path.join(MODEL_DIR, "ro_RO-mihai-medium.onnx.json")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

_voice = None
_config = None


def _load_voice():
    global _voice, _config
    if _voice is not None:
        return _voice, _config

    import onnxruntime
    from pathlib import Path
    from piper import PiperVoice, PiperConfig
    from piper.config import PhonemeType

    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _config = PiperConfig(
        num_symbols=cfg["num_symbols"],
        num_speakers=cfg["num_speakers"],
        sample_rate=cfg["audio"]["sample_rate"],
        espeak_voice=cfg["espeak"]["voice"],
        phoneme_id_map=cfg["phoneme_id_map"],
        phoneme_type=PhonemeType[cfg["phoneme_type"].upper()],
        speaker_id_map=cfg.get("speaker_id_map", {}),
        piper_version=cfg.get("piper_version"),
        length_scale=cfg["inference"]["length_scale"],
        noise_scale=cfg["inference"]["noise_scale"],
        noise_w_scale=cfg["inference"]["noise_w"],
    )

    session = onnxruntime.InferenceSession(MODEL_ONNX)
    _voice = PiperVoice(session, _config, download_dir=Path("."))
    return _voice, _config


def generate_scenario_wav(scenario_id, text):
    if not text or not text.strip():
        return False

    try:
        voice, config = _load_voice()
        sentences = voice.phonemize(text)
        audio_chunks = []
        for sentence_phonemes in sentences:
            phoneme_ids = voice.phonemes_to_ids(sentence_phonemes)
            audio = voice.phoneme_ids_to_audio(phoneme_ids)
            audio_chunks.append(audio)

        full_audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
        if len(full_audio) == 0:
            return False

        os.makedirs(ASSETS_DIR, exist_ok=True)
        output_path = os.path.join(ASSETS_DIR, f"{scenario_id}.wav")

        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.sample_rate)
            wav_file.writeframes((full_audio * 32767).astype(np.int16).tobytes())

        return True
    except Exception as e:
        import logging
        logging.error(f"TTS generation error for scenario {scenario_id}: {e}")
        return False
