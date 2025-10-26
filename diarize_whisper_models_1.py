#!/usr/bin/env python3

from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import pvfalcon
import librosa
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import os
import soundfile as sf

audio_path = "audio for transcription test.mp3"
output_dir = "transcripts"
FALCON_ACCESS_KEY = "INSERT_KEY_HERE"

models = {
    "rishabh": "rishabhjain16/whisper_medium_en_to_myst_pf",
    "kid": "aadel4/kid-whisper-medium-en-myst",
    "openai": "openai/whisper-medium.en"
}

os.makedirs(output_dir, exist_ok=True)

print(f"Processing: {audio_path}")

audio, sr = librosa.load(audio_path, sr=16000, mono=True)
audio = np.array(audio, dtype=np.float32)

print(f"Audio loaded: {len(audio)} samples at {sr}Hz")

falcon = pvfalcon.create(access_key=FALCON_ACCESS_KEY)
speaker_segments = falcon.process_file(audio_path)

print(f"Found {len(speaker_segments)} speaker segments")

def match_speaker(transcript_segment, speaker_segments):
    ts_start, ts_end = transcript_segment["start"], transcript_segment["end"]
    best_speaker = None
    best_score = 0
    
    for speaker_segment in speaker_segments:
        overlap_start = max(ts_start, speaker_segment.start_sec)
        overlap_end = min(ts_end, speaker_segment.end_sec)
        overlap = max(0, overlap_end - overlap_start)
        
        if ts_end > ts_start:
            score = overlap / (ts_end - ts_start)
            if score > best_score:
                best_score = score
                best_speaker = speaker_segment.speaker_tag
    
    return best_speaker, best_score

def generate_elan(transcripts, elan_path):
    root = ET.Element('ANNOTATION_DOCUMENT')
    root.set('AUTHOR', '')
    root.set('DATE', '')
    root.set('FORMAT', '2.8')
    root.set('VERSION', '2.8')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://www.mpi.nl/tools/elan/EAFv2.8.xsd')

    header = ET.SubElement(root, 'HEADER')
    header.set('MEDIA_FILE', '')
    header.set('TIME_UNITS', 'milliseconds')

    absolute_audio_path = os.path.abspath(audio_path).replace('\\', '/')
    media_descriptor = ET.SubElement(header, 'MEDIA_DESCRIPTOR')
    media_descriptor.set('MEDIA_URL', 'file:///' + absolute_audio_path)
    media_descriptor.set('MIME_TYPE', 'audio/x-wav')
    media_descriptor.set('RELATIVE_MEDIA_URL', './' + os.path.basename(audio_path))

    time_order = ET.SubElement(root, 'TIME_ORDER')

    time_slot_id = 1
    time_slots = {}
    for _, start, end, _ in transcripts:
        if start not in time_slots:
            time_slots[start] = f'ts{time_slot_id}'
            time_slot_id += 1
        if end not in time_slots:
            time_slots[end] = f'ts{time_slot_id}'
            time_slot_id += 1

    for time, ts_id in time_slots.items():
        time_slot = ET.SubElement(time_order, 'TIME_SLOT')
        time_slot.set('TIME_SLOT_ID', ts_id)
        time_slot.set('TIME_VALUE', str(int(time * 1000)))

    linguistic_type = ET.SubElement(root, 'LINGUISTIC_TYPE')
    linguistic_type.set('GRAPHIC_REFERENCES', 'false')
    linguistic_type.set('LINGUISTIC_TYPE_ID', 'default-lt')
    linguistic_type.set('TIME_ALIGNABLE', 'true')

    tier = ET.SubElement(root, 'TIER')
    tier.set('TIER_ID', 'transcription')
    tier.set('LINGUISTIC_TYPE_REF', 'default-lt')

    for speaker, start, end, transcript in transcripts:
        annotation = ET.SubElement(tier, 'ANNOTATION')
        alignable_annotation = ET.SubElement(annotation, 'ALIGNABLE_ANNOTATION')
        alignable_annotation.set('ANNOTATION_ID', f'a{time_slot_id}')
        alignable_annotation.set('TIME_SLOT_REF1', time_slots[start])
        alignable_annotation.set('TIME_SLOT_REF2', time_slots[end])
        annotation_value = ET.SubElement(alignable_annotation, 'ANNOTATION_VALUE')
        annotation_value.text = f'{speaker}: {transcript}'
        time_slot_id += 1

    tree = ET.ElementTree(root)
    tree.write(elan_path, xml_declaration=True, encoding='UTF-8')

processor = AutoProcessor.from_pretrained("openai/whisper-medium.en")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for name, model_name in models.items():
    print(f"Processing with {name} model")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )

    result = asr_pipeline(audio, return_timestamps=True)
    transcript_segments = result.get("segments", [])

    transcripts = []
    txt_output = []

    for t_segment in transcript_segments:
        speaker, score = match_speaker(t_segment, speaker_segments)
        
        if speaker is None:
            speaker = "Unknown"

        transcripts.append((speaker, t_segment["start"], t_segment["end"], t_segment["text"]))
        txt_output.append(f"[{t_segment['start']:.2f}-{t_segment['end']:.2f}] {speaker}: {t_segment['text']}")

    txt_file = os.path.join(output_dir, f"{name}_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_output))

    eaf_file = os.path.join(output_dir, f"{name}_{timestamp}.eaf")
    generate_elan(transcripts, eaf_file)

    print(f"Completed {name} model")

print("Done")
