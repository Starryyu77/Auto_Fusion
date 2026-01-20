# M4-Optimized Environment Setup & MPS Verification Plan

I will generate the necessary scripts to set up and verify the MM-CoT baseline on your MacBook Pro M4, ensuring full utilization of Metal Performance Shaders (MPS).

## 1. Environment Setup Script (`setup_m4_track_a.sh`)
I will create a shell script that:
- Detects `conda` or `mamba`.
- Creates a dedicated environment `mmcot_mps` (Python 3.10 recommended for compatibility).
- Installs **PyTorch for macOS ARM64** (MPS enabled).
- Installs key libraries: `transformers`, `accelerate`, `sentencepiece`, `datasets`, `psutil` (for memory monitoring), `pylint`, and `black`.

## 2. MPS Verification Script (`verify_mps_baseline.py`)
I will create a Python script to validate the hardware acceleration:
- **Device Detection**: Robust check for `mps` availability.
- **Model Test**: Loads `google/t5-v1_1-base`, moves it to the MPS device.
- **Inference Test**: Runs a forward pass with dummy text and image tensors (handling `float32` casting for MPS stability).
- **Resource Monitoring**: Uses `psutil` to report RAM usage (since `torch.cuda` memory tools don't apply to MPS).

## 3. Monkey Patch for MPS (`src/mps_patch.py`)
To avoid rewriting existing CUDA code, I will create a module `src/mps_patch.py` that:
- Intercepts `torch.cuda.is_available()` to return `True` (simulated) or redirect logic.
- Patches `torch.Tensor.cuda()` to execute `.to("mps")` instead.
- Redirects `torch.device("cuda")` calls to `torch.device("mps")`.

## 4. Documentation
- I will update `.ai_status.md` to record these environment configuration tools.
