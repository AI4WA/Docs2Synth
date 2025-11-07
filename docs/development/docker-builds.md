# Docker Builds

This document explains how to build and test Docker images for Docs2Synth.

## Overview

Docs2Synth provides Docker images for both CPU and GPU environments. Due to the large size of GPU images (~11GB), Docker builds are **not run in CI/CD** to avoid resource constraints on GitHub Actions runners.

## Building Images Locally

### Quick Start

Use the provided build script:

```bash
# Build and test CPU image
./scripts/build-docker.sh cpu

# Build and test GPU image
./scripts/build-docker.sh gpu

# Build both images
./scripts/build-docker.sh all
```

### What the Script Does

The build script automatically:

1. **Builds** the Docker image with appropriate build arguments
2. **Verifies** Python version and package installation
3. **Tests** the CLI and runs pytest
4. **Reports** image size and provides usage instructions

### Manual Building

You can also build images manually:

```bash
# CPU image
docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .

# GPU image
docker build --build-arg BUILD_TYPE=gpu -t docs2synth:gpu .
```

## Image Specifications

### CPU Image

- **Size**: ~2.5 GB
- **Base**: `python:3.11-slim`
- **Use Cases**: Development, testing, CPU-only deployment
- **Build Time**: ~10-15 minutes

**Includes:**
- Python 3.11
- CPU-only PyTorch
- PaddleOCR and dependencies
- All application code and tests

### GPU Image

- **Size**: ~11 GB
- **Base**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Use Cases**: GPU training, GPU inference, production with CUDA
- **Build Time**: ~15-25 minutes

**Includes:**
- Python 3.11 (installed via deadsnakes PPA)
- GPU-enabled PyTorch with CUDA 11.8
- NVIDIA CUDA libraries (cublas, cudnn, etc.)
- PaddleOCR and dependencies
- All application code and tests

## Why No CI for Docker Builds?

### Problem

GitHub Actions runners have limited disk space (~14GB usable). Building the GPU image requires:
- ~11GB for the image itself
- ~3-5GB for build cache and layers
- **Total: ~16GB** (exceeds available space)

Even with aggressive cleanup strategies, builds consistently failed with "no space left on device" errors.

### Solution

We moved Docker builds to a manual script that developers and maintainers can run locally or on machines with sufficient resources.

### Benefits

- ✅ No CI failures due to disk space
- ✅ Faster CI pipelines (focus on code quality)
- ✅ More control over when/where to build large images
- ✅ Reduced GitHub Actions costs

### Trade-offs

- ⚠️ Docker images not automatically tested on every commit
- ⚠️ Requires manual building before releases
- ⚠️ Assumes Dockerfile changes are less frequent

## Testing Docker Images

### CPU Image Tests

```bash
# Run all tests
docker run --rm docs2synth:cpu pytest tests/ -v

# Test specific module
docker run --rm docs2synth:cpu pytest tests/test_datasets.py -v

# Interactive shell
docker run -it docs2synth:cpu /bin/bash
```

### GPU Image Tests

```bash
# Check CUDA availability
docker run --rm --gpus all docs2synth:gpu python -c "import torch; print(torch.cuda.is_available())"

# Run tests with GPU
docker run --rm --gpus all docs2synth:gpu pytest tests/ -v

# Interactive shell with GPU
docker run --gpus all -it docs2synth:gpu /bin/bash
```

**Note:** GPU tests require:
- Linux host with NVIDIA GPU
- NVIDIA drivers installed
- NVIDIA Container Toolkit installed

## Production Deployment

### Building for Production

```bash
# Tag for your registry
docker build --build-arg BUILD_TYPE=cpu -t your-registry/docs2synth:v0.1.0-cpu .
docker build --build-arg BUILD_TYPE=gpu -t your-registry/docs2synth:v0.1.0-gpu .

# Push to registry
docker push your-registry/docs2synth:v0.1.0-cpu
docker push your-registry/docs2synth:v0.1.0-gpu
```

### Running in Production

```bash
# CPU deployment
docker run -d \
  --name docs2synth \
  -v /data:/app/data \
  -v /logs:/app/logs \
  your-registry/docs2synth:v0.1.0-cpu \
  python -m docs2synth.cli process --config /app/config.yml

# GPU deployment
docker run -d \
  --gpus all \
  --name docs2synth-gpu \
  -v /data:/app/data \
  -v /logs:/app/logs \
  your-registry/docs2synth:v0.1.0-gpu \
  python -m docs2synth.cli process --config /app/config.yml
```

## Troubleshooting

### Build Fails with "No Space Left"

**Solution**: Free up disk space before building:

```bash
# Remove unused Docker images
docker image prune -a

# Remove unused containers
docker container prune

# Remove build cache
docker builder prune
```

### GPU Image Fails to Run

**Check NVIDIA setup:**

```bash
# Verify nvidia-smi works
nvidia-smi

# Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Slow Build Times

**Use build cache:**

```bash
# BuildKit automatically caches layers
DOCKER_BUILDKIT=1 docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .
```

**Or use docker-compose cache:**

```bash
docker compose build docs2synth-cpu
```

## CI/CD Integration

While we don't build Docker images in GitHub Actions, you can integrate the build script into your own CI/CD:

```yaml
# Example: GitLab CI with Docker-in-Docker
build-docker:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - ./scripts/build-docker.sh cpu
  only:
    - tags
```

Or use dedicated runners with more disk space:

```yaml
# Example: Self-hosted GitHub Actions runner
build-docker:
  runs-on: self-hosted  # Your own runner with 50GB+ disk
  steps:
    - uses: actions/checkout@v4
    - name: Build Docker images
      run: ./scripts/build-docker.sh all
```

## Best Practices

1. **Build before releases** - Always build and test Docker images before tagging a release
2. **Version your images** - Tag images with version numbers (e.g., `v0.1.0`, `v0.1.0-cpu`)
3. **Test locally first** - Build and test on your local machine before deploying
4. **Monitor image size** - Keep track of image sizes and optimize when possible
5. **Document changes** - Update this doc when making Dockerfile changes

## References

- [Dockerfile](https://github.com/AI4WA/Docs2Synth/blob/main/Dockerfile)
- [Build Script](https://github.com/AI4WA/Docs2Synth/blob/main/scripts/build-docker.sh)
- [Docker Compose](https://github.com/AI4WA/Docs2Synth/blob/main/docker-compose.yml)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
