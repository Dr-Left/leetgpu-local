# LeetGPU Local Testing Repo

<img width="1136" height="577" alt="image" src="https://github.com/user-attachments/assets/bdb37d54-bfcd-4f40-97ca-f5670eae5dc3" />

## Local Testing

Test Triton solutions locally against any challenge using `scripts/test_local.py`:

```bash
# Setup (one-time)
conda create -n leetgpu python=3.11 -y
conda activate leetgpu
pip install torch triton

# Run all tests (functional + performance)
python scripts/test_local.py challenges/easy/1_vector_add solution.triton.py

# Functional tests only
python scripts/test_local.py challenges/easy/1_vector_add solution.triton.py --functional-only

# Performance benchmark only
python scripts/test_local.py challenges/easy/1_vector_add solution.triton.py --perf-only --iterations 500
```

The harness compares your solution against the challenge's `reference_impl` for correctness, and benchmarks performance against the PyTorch baseline.

# About LeetGPU
This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases, and starter templates for multiple GPU programming frameworks.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for each supported framework


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## License

This problem set is licensed under [CC BY‑NC‑ND 4.0 license](LICENSE).

© 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.
