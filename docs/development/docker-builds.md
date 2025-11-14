# Docker Builds

Build and test Docker images for CPU and GPU environments.

## Quick Start

```bash
# Build and test CPU image
./scripts/build-docker.sh cpu

# Build and test GPU image
./scripts/build-docker.sh gpu

# Build both
./scripts/build-docker.sh all
```

---

## Images

### CPU Image

```bash
docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .
```

- **Size**: ~2.5 GB
- **Base**: `python:3.11-slim`
- **Use**: Development, testing, CPU-only deployment
- **Build time**: ~10-15 min

### GPU Image

```bash
docker build --build-arg BUILD_TYPE=gpu -t docs2synth:gpu .
```

- **Size**: ~11 GB
- **Base**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Use**: GPU training, GPU inference, production with CUDA
- **Build time**: ~15-25 min

---

## Testing

### CPU Tests

```bash
# Run all tests
docker run --rm docs2synth:cpu pytest tests/ -v

# Interactive shell
docker run -it docs2synth:cpu /bin/bash
```

### GPU Tests

```bash
# Check CUDA
docker run --rm --gpus all docs2synth:gpu python -c "import torch; print(torch.cuda.is_available())"

# Run tests with GPU
docker run --rm --gpus all docs2synth:gpu pytest tests/ -v

# Interactive shell
docker run --gpus all -it docs2synth:gpu /bin/bash
```

**Requirements for GPU:**
- Linux with NVIDIA GPU
- NVIDIA drivers installed
- NVIDIA Container Toolkit installed

---

## Production

```bash
# Build and tag
docker build --build-arg BUILD_TYPE=cpu -t your-registry/docs2synth:v0.1.0-cpu .
docker build --build-arg BUILD_TYPE=gpu -t your-registry/docs2synth:v0.1.0-gpu .

# Push
docker push your-registry/docs2synth:v0.1.0-cpu
docker push your-registry/docs2synth:v0.1.0-gpu
```

### Run in Production

```bash
# CPU
docker run -d \
  --name docs2synth \
  -v /data:/app/data \
  -v /logs:/app/logs \
  your-registry/docs2synth:v0.1.0-cpu \
  python -m docs2synth.cli process --config /app/config.yml

# GPU
docker run -d \
  --gpus all \
  --name docs2synth-gpu \
  -v /data:/app/data \
  -v /logs:/app/logs \
  your-registry/docs2synth:v0.1.0-gpu \
  python -m docs2synth.cli process --config /app/config.yml
```

---

## Why No CI for Docker?

GitHub Actions runners have limited disk space (~14GB). GPU image needs ~16GB total:
- ~11GB image
- ~5GB build cache

**Solution**: Manual builds locally or on dedicated machines.

---

## Troubleshooting

### No Space Left

```bash
docker image prune -a
docker container prune
docker builder prune
```

### GPU Image Fails

```bash
# Verify NVIDIA setup
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Slow Builds

```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build --build-arg BUILD_TYPE=cpu -t docs2synth:cpu .
```

---

## References

- [Dockerfile](https://github.com/AI4WA/Docs2Synth/blob/main/Dockerfile)
- [Build Script](https://github.com/AI4WA/Docs2Synth/blob/main/scripts/build-docker.sh)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
