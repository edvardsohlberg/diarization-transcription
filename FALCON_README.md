# Falcon-Based Speaker Diarization with Whisper Models

This script (`diarize_whisper_models_falcon.py`) uses **Falcon** for speaker diarization first, then applies **Whisper transcription** to each diarized segment individually.

## 🎯 Key Benefits

- **Preserves speaker assignments**: Speaker labels from Falcon are maintained exactly as detected
- **Preserves segment boundaries**: Transcription boundaries match diarization results
- **Handles short segments**: Configurable minimum duration to skip problematic segments
- **Better speaker consistency**: No speaker switching within segments

## 🔄 How It Works

1. **Falcon Diarization**: Runs first to identify speaker segments with timestamps
2. **Segment Analysis**: Analyzes segment durations and filters out very short ones
3. **Whisper Transcription**: Applies transcription to each valid diarized segment
4. **Output Generation**: Creates both TXT and EAF (ELAN) format outputs

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements_falcon.txt
```

### 2. Get Falcon Access Key
1. Visit [Picovoice Console](https://console.picovoice.ai/)
2. Create an account and get your access key
3. Update `FALCON_ACCESS_KEY` in the script

### 3. Pre-download Whisper Models (Optional but Recommended)
```bash
python -c "
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-medium.en')
AutoProcessor.from_pretrained('openai/whisper-medium.en')
AutoModelForSpeechSeq2Seq.from_pretrained('rishabhjain16/whisper_medium_en_to_myst_pf')
AutoModelForSpeechSeq2Seq.from_pretrained('aadel4/kid-whisper-medium-en-myst')
"
```

## 🚀 Usage

### Basic Usage
```bash
python diarize_whisper_models_falcon.py
```

### Configuration
Edit the configuration section at the top of the script:
```python
# Audio file to process
audio_path = "your_audio_file.mp3"

# Output directory
output_dir = "transcripts"

# Minimum segment duration (seconds)
MIN_SEGMENT_DURATION = 0.5

# Your Falcon access key
FALCON_ACCESS_KEY = "your_access_key_here"
```

## 📊 Output

The script generates:
- **TXT files**: Human-readable transcripts with timestamps and speaker labels
- **EAF files**: ELAN annotation format for further analysis

## 🔧 Troubleshooting

### Common Issues

1. **Short Segments**: If Falcon creates many short segments (<0.5s), increase `MIN_SEGMENT_DURATION`
2. **Empty Transcripts**: Check if segments are too short for Whisper to process
3. **Falcon Errors**: Verify your access key and account credits

### Testing Integration
Run the test script to verify everything works:
```bash
python test_falcon_integration.py
```

## 📈 Performance Tips

- **Segment Duration**: Aim for segments >1 second for best Whisper results
- **Audio Quality**: Higher quality audio improves both diarization and transcription
- **Model Caching**: Pre-download models to avoid repeated downloads

## 🔍 Differences from Pyannote Version

| Feature | Pyannote Version | Falcon Version |
|---------|------------------|----------------|
| Diarization Order | First | First |
| Speaker Consistency | Per segment | Per segment |
| Segment Boundaries | Diarization-based | Diarization-based |
| Short Segment Handling | Limited | Configurable filtering |
| Dependencies | HuggingFace + Pyannote | HuggingFace + Falcon |

## 📝 Example Output

```
🔄 Processing segment 1/4: Speaker 1 [0.00-2.45] (duration: 2.45s)
    🔊 Speaker 1: Hello, how are you today? I hope you're doing well.
🔄 Processing segment 2/4: Speaker 2 [2.45-4.12] (duration: 1.67s)
    🔊 Speaker 2: I'm doing great, thank you for asking.
```

## 🆘 Support

- **Falcon Issues**: Check [Picovoice Documentation](https://picovoice.ai/docs/)
- **Whisper Issues**: Check [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- **Script Issues**: Review error messages and check configuration
