# Docker images

| File | Architecture | Compose service |
|------|----------------|-----------------|
| [`Dockerfile.amd64`](Dockerfile.amd64) | `linux/amd64` (x86_64) | `clasp-amd64` → image `clasp:amd64` |
| [`Dockerfile.arm64`](Dockerfile.arm64) | `linux/arm64` (Apple Silicon, ARM servers) | `clasp-arm64` → image `clasp:arm64` |

Build from the **repository root** (context must be `.` so `COPY pyproject.toml`, `src/`, and `scripts/` resolve):

```bash
docker compose build clasp-amd64   # x86_64
docker compose build clasp-arm64   # ARM64
```

Optional dependency group (LaBSE, datasets, etc.) is set in [`docker-compose.yml`](../docker-compose.yml) as `UV_EXTRA_GROUPS=voxpopuli`. For a lighter SPIRAL-only image, override with `realdata`:

```bash
docker build -f docker/Dockerfile.amd64 -t clasp:spiral-eval \
  --build-arg UV_EXTRA_GROUPS=realdata .
```
