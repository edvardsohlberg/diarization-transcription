# Testing Guide for Whisper Models Scripts

This guide shows you how to test the Whisper transcription scripts locally without downloading the large AI models.

## What This Testing Approach Achieves

- **Tests script logic** without downloading models (saves GB of data)
- **Verifies file operations** (TXT and EAF generation)
- **Tests audio processing** pipeline
- **Validates XML generation** for ELAN files
- **Checks error handling** and edge cases
- **Fast execution** (seconds instead of minutes)

## Quick Start

### 1. Use Your System Python (Recommended)
Your PyCharm CE setup already has all the necessary packages installed in Python 3.13:
```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 test_diarize_whisper_models.py
```

### 2. Or Use Virtual Environment (Alternative)
```bash
source .venv/bin/activate
python3 test_diarize_whisper_models.py
```

## What Gets Tested

### Core Functionality
- **Audio loading** (with fallback to mock data)
- **Speaker diarization** (mocked with realistic segments)
- **Model pipeline creation** (mocked transformers)
- **Transcription processing** (mocked results)
- **File output generation** (TXT and EAF)

### File Operations
- **Directory creation** and management
- **Text file writing** with proper encoding
- **XML generation** for ELAN annotation files
- **Timestamp handling** and file naming

### Error Handling
- **Missing audio files** (graceful fallback)
- **Audio loading failures** (mock data substitution)
- **Model loading errors** (mocked responses)

## How the Mocking Works

### 1. Model Loading
```python
# Instead of downloading models:
with patch('transformers.AutoModelForSpeechSeq2Seq.from_pretrained') as mock_loader:
    mock_model = Mock()
    mock_loader.return_value = mock_model
```

### 2. Transcription Pipeline
```python
# Instead of running actual transcription:
mock_pipeline = Mock()
mock_pipeline.return_value = {'text': '[MOCK] Test transcription'}
```

### 3. Speaker Diarization
```python
# Instead of running pyannote:
class MockDiarization:
    def itertracks(self, yield_label=True):
        mock_segments = [
            ("SPEAKER_00", 0.0, 5.0),
            ("SPEAKER_01", 5.0, 10.0),
            # ... more segments
        ]
```

## Expected Output

After running the tests, you'll find:

```
transcripts/
├── rishabh_20241220_143022_TEST.txt
├── kid_20241220_143022_TEST.txt
├── openai_20241220_143022_TEST.txt
├── rishabh_20241220_143022_TEST.eaf
├── kid_20241220_143022_TEST.eaf
└── openai_20241220_143022_TEST.eaf
```

## Testing Different Scenarios

### Test with Real Audio File
1. Place your audio file in the project directory
2. Update `audio_path` in the test script
3. Run the test - it will use real audio data

### Test with Missing Audio
1. Remove or rename the audio file
2. Run the test - it will use mock audio data
3. Verify graceful fallback behavior

### Test File Permissions
1. Make the output directory read-only
2. Run the test - verify error handling
3. Check that appropriate error messages appear

## What to Look For

### Success Indicators
- Script runs without errors
- Output files are created with `_TEST` suffix
- File sizes are reasonable (not 0 bytes)
- XML files are well-formed
- Timestamps are current

### Warning Signs
- Import errors (missing dependencies)
- Permission errors (file access issues)
- Empty output files
- Malformed XML

## Troubleshooting

### Common Issues

**Import Error: No module named 'librosa'**
```bash
pip install librosa
```

**Permission Denied**
```bash
chmod 755 transcripts/
```

**Audio File Not Found**
- Check file path in script
- Ensure audio file exists
- Verify file permissions

### Debug Mode
Add more verbose logging by modifying the mock functions:
```python
def mock_transcription(audio):
    print(f"Mock transcription called with {len(audio)} samples")
    return {'text': '[DEBUG] Mock result'}
```

## Performance Comparison

| Test Type | Time | Disk Usage | Network |
|-----------|------|------------|---------|
| **Mock Testing** | ~5 seconds | ~1 MB | 0 MB |
| **Real Models** | 5-15 minutes | 2-8 GB | 2-8 GB |

## Next Steps After Testing

1. **Verify script logic** works as expected
2. **Check file outputs** are correctly formatted
3. **Test with real audio** (if available)
4. **Deploy to server** with confidence
5. **Run full models** on server for actual transcription

## Related Files

- `test_diarize_whisper_models.py` - Test version of diarization script
- `requirements_test.txt` - Minimal dependencies for testing (optional)
- `TESTING_GUIDE.md` - This guide

## Pro Tips

- **Keep test files** for comparison with real outputs
- **Use version control** to track test script changes
- **Test regularly** when modifying the main scripts
- **Mock external APIs** to avoid rate limits during testing
- **Use different audio files** to test various scenarios
