# Streaming Diarization Eval

Benchmark realtime speaker diarization systems (DIART, Streaming Sortformer) on standard datasets.

## Quick Start

### Running Evaluation

```bash
# Activate your virtual environment
source venv-310/bin/activate  # Linux/Mac
# or
.\venv-310\Scripts\Activate.ps1  # Windows PowerShell

# Run with default config
python -m src.runner

# Run with specific config
python -m src.runner config_callhome.yaml
```

### Project Structure

```
streaming-diarization-eval/
├── src/                    # Source code
│   ├── runner.py          # Main evaluation orchestrator
│   ├── config.py          # Configuration classes
│   ├── dataset/           # Dataset providers
│   ├── systems/           # Diarization system adapters
│   └── evaluator/         # Metrics computation
├── data/                  # Dataset cache
├── config.yaml            # Configuration files
└── results/               # Evaluation outputs
```

## Windows Setup Notes

### TorchCodec Compatibility Issue

PyTorch 2.8.0+ requires torchcodec 0.7+, but torchcodec 0.7 has FFmpeg DLL dependency issues on Windows. The error you'll see:

```
RuntimeError: Could not load libtorchcodec. Could not find module 'libtorchcodec_core[X].dll'
```

**Workaround:**

1. **Use torchcodec 0.7**:
   ```bash
   pip uninstall torchcodec
   pip install torchcodec==0.7
   ```

2. **Add FFmpeg DLLs** to torchcodec directory:
   - Download [FFmpeg 7.1.1 shared build](https://github.com/BtbN/FFmpeg-Builds/releases) (Windows, shared, full)
   - Extract all `.dll` files from the `bin/` folder
   - Copy them to: `<venv>\Lib\site-packages\torchcodec\`

## Model Access

### HuggingFace Models

To use DIART models, you must first visit each model on HuggingFace and accept the usage agreement.

## Sortformer Configuration

### Configuration Presets

Source: [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)

| Configuration | Latency | RTF | chunk_len | subsampling_factor | chunk_right_context | chunk_left_context | fifo_len | spkcache_len | spkcache_update_period | log |
|---|---|---|---|---|---|---|---|---|---|---|
| **Default** | - | - | 188 | 8 | 1 | 1 | 0 | 188 | 188 | False |
| **Very High Latency** | 30.4s | 0.002 | 340 | 8 | 40 | 1 | 40 | 188 | 300 | False |
| **High Latency** | 10.0s | 0.005 | 124 | 8 | 1 | 1 | 124 | 188 | 124 | False |
| **Low Latency** | 1.04s | 0.093 | 6 | 8 | 7 | 1 | 188 | 188 | 144 | False |
| **Ultra Low Latency** | 0.32s | 0.180 | 3 | 8 | 1 | 1 | 188 | 188 | 144 | False |
| **WhisperLiveKit** | ~1.0s | - | 10 | 10 | 0 | 10 | 188 | 188 | 144 | False |

#### Latency Calculation

**Input Buffer Latency** (seconds) = `(chunk_len + chunk_right_context) × subsampling_factor × window_stride`

Where:
- `chunk_len` + `chunk_right_context` = total frames buffered before processing
- `subsampling_factor` = frame subsampling (typically 8)
- `window_stride` = 0.01s (10ms hop length for mel spectrogram preprocessing)

**Example:** Very High Latency configuration:
- (340 + 40) × 8 × 0.01 = 30.4 seconds

**Note:** This latency reflects only the input buffering time before processing begins. It does not include computational processing time (see RTF for processing speed).
   


