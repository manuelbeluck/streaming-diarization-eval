# Streaming Diarization Eval

Benchmark realtime speaker diarization systems (DIART, Streaming Sortformer) on standard datasets.

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



