# WhisperModels: Speaker Diarization and Transcription Pipeline

A Python toolkit for speaker diarization and transcription using multiple Whisper models. This project compares different Whisper model variants and provides both Pyannote and Falcon-based speaker diarization approaches.

## Overview

This repository contains several different approaches for:
- **Speaker Diarization**: Identifying who spoke when using Pyannote or Falcon
- **Speech Transcription**: Converting speech to text using various Whisper models
- **Model Comparison**: Comparing different Whisper variants (OpenAI, Rishabh, Kid-Whisper)
- **Output Formats**: Generating both TXT and ELAN (.eaf) annotation files

## Project Structure

```
WhisperModels/
├── Core Scripts
│   ├── diarize_whisper_models.py          # Main Pyannote-based diarization
│   ├── diarize_whisper_models_falcon.py   # Falcon-based diarization (robust)
│   ├── diarize_whisper_models_simple.py   # Simplified Falcon version
│   ├── diarize_whisper_models_1.py        # Basic Falcon implementation
│   ├── compare_whisper_models.py           # Simple transcription comparison
├── Configuration
│   ├── requirements_hf.txt                 # HuggingFace + Pyannote dependencies
│   ├── requirements_falcon.txt            # Falcon-specific dependencies
├── Documentation
│   ├── FALCON_README.md                   # Falcon-specific documentation
├── Audio Files
│   └── audio for transcription test.mp3   # Sample audio file
└── Outputs
    └── transcripts/                       # Generated transcriptions
        ├── *.txt                          # Text transcripts
        └── *.eaf                          # ELAN annotation files
```

## Quick Start

### Installation

Choose the approach that fits your needs:

**For Pyannote-based diarization:**
```bash
pip install -r requirements_hf.txt
```

**For Falcon-based diarization:**
```bash
pip install -r requirements_falcon.txt
```


### Configuration

**For Pyannote approach:**
- Get HuggingFace token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
- Update `HF_TOKEN` in the script

**For Falcon approach:**
- Get Falcon access key from [console.picovoice.ai](https://console.picovoice.ai/)
- Update `FALCON_ACCESS_KEY` in the script

### Basic Usage

**Simple transcription comparison:**
```bash
python compare_whisper_models.py
```

**Full diarization with Pyannote:**
```bash
python diarize_whisper_models.py
```

**Full diarization with Falcon:**
```bash
python diarize_whisper_models_falcon.py
```

## Available Scripts

### Core Scripts

| Script | Purpose | Diarization | Features |
|--------|---------|-------------|----------|
| `compare_whisper_models.py` | Simple transcription | None | Basic Whisper comparison |
| `diarize_whisper_models.py` | Full pipeline | Pyannote | Robust error handling |
| `diarize_whisper_models_falcon.py` | Full pipeline | Falcon | Production-ready |
| `diarize_whisper_models_simple.py` | Minimal pipeline | Falcon | Simplified version |
| `diarize_whisper_models_1.py` | Basic pipeline | Falcon | Clean implementation |


## Supported Models

The project tests three Whisper model variants:

1. **OpenAI Whisper Medium** (`openai/whisper-medium.en`)
   - Original OpenAI model
   - General-purpose transcription

2. **Rishabh Whisper** (`rishabhjain16/whisper_medium_en_to_myst_pf`)
   - Fine-tuned for specific domain
   - May perform better on specialized content

3. **Kid Whisper** (`aadel4/kid-whisper-medium-en-myst`)
   - Fine-tuned for children's speech
   - Optimized for higher-pitched voices


## Stanford SC Shell Server Deployment

For running on Stanford's SC Shell server, use these commands:

```bash
cd /juice6/scr6/nlp/child-child-comm/logs
conda activate whisperenv
export HF_HOME=/juice6/scr6/nlp/child-child-comm/logs/hf_cache/
srun --account=nlp --partition=jag-standard --gres=gpu:1 --mem=32G --cpus-per-task=4 --pty bash -c "source $PWD/miniconda3/etc/profile.d/conda.sh && conda activate whisperenv && python diarize_whisper_models_falcon_new.py"
```

**Setup Steps:**
1. Create conda environment: `conda create -n whisperenv python=3.9`
2. Activate environment: `conda activate whisperenv`
3. Install dependencies: `pip install -r requirements_falcon.txt`
4. Pre-download models to avoid timeout issues
5. Upload your audio files to the server
6. Update file paths in the scripts

**Server Considerations:**
- Use absolute paths for audio files
- Ensure sufficient disk space for model downloads (~2-8GB)
- Monitor GPU memory usage for large audio files
- Consider chunking very long audio files (>30 minutes)

## Diarization Approaches

### Pyannote Approach
- **Pros**: Free, open-source, good accuracy
- **Cons**: Requires HuggingFace token, slower processing
- **Best for**: Research, academic use, cost-sensitive projects

### Falcon Approach
- **Pros**: Fast, high accuracy, commercial-grade
- **Cons**: Requires API key, usage-based pricing
- **Best for**: Production use, time-sensitive projects

## Performance Tips

### Model Selection
- **General speech**: Use OpenAI Whisper Medium
- **Children's speech**: Use Kid Whisper
- **Specialized content**: Use Rishabh Whisper

### Audio Optimization
- **Sample rate**: 16kHz recommended
- **Format**: WAV or MP3 supported
- **Duration**: Chunk files >30 minutes for better processing

### Resource Management
- **Memory**: 16GB+ recommended for full pipeline
- **GPU**: CUDA-compatible GPU for faster processing
- **Storage**: 10GB+ free space for models and outputs

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Missing dependencies
pip install -r requirements_falcon.txt

# CUDA issues
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Audio Loading Issues:**
- Check file format (MP3, WAV supported)
- Verify file permissions
- Ensure sufficient disk space

**Model Download Issues:**
- Check internet connection
- Verify HuggingFace token (Pyannote)
- Verify Falcon access key (Falcon)

**Memory Issues:**
- Reduce batch size
- Use CPU-only processing
- Process shorter audio segments

## Example Workflows

### Research Workflow
1. Run small-scale tests with real audio
2. Deploy to server for full processing
3. Compare outputs across different models

### Production Workflow
1. Use Falcon-based scripts for speed
2. Pre-download models to avoid delays
3. Implement error handling and retry logic
4. Monitor API usage and costs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## License

This project is open source. Please check individual model licenses:
- OpenAI Whisper: MIT License
- Pyannote: MIT License
- Falcon: Commercial License (check Picovoice terms)

## Resources

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [Picovoice Falcon](https://picovoice.ai/docs/)
- [ELAN Annotation Format](https://archive.mpi.nl/tla/elan)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Open an issue with detailed error logs
3. Include your system configuration and audio file details
