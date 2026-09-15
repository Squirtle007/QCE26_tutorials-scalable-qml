#!/usr/bin/env bash
# Local .venv setup for the QCE26 tutorial, driven by uv.
#
#   bash setup-venv.sh
#
# This is the third install path in this repository, and the only one that does
# not need root:
#
#   Dockerfile    the primary, self-contained path — builds the tutorial image
#   run.sh        the same steps applied in place, inside an existing CUDA-Q
#                 container (needs sudo/apt for the CUDA 13.2 toolchain)
#   setup-venv.sh this file — a plain virtualenv on the host, no apt, no sudo
#
# Scope: it installs everything in requirements.txt, which is enough for
# notebooks 02 and 03. It does NOT provide CUDA-Q, cudaq_einsum or the
# multi-stream cuTENSOR backend — those are installed natively / built from
# source by the Dockerfile, so notebooks 00, 01 and 04 still need the image.
#
# The interesting problem this solves
# -----------------------------------
# qkan compiles its CuTe extension against the *installed torch*, and its
# setup.py refuses to build (falling back to a pure-Python qkan) when the
# system nvcc's CUDA version differs from torch.version.cuda. The pinned torch
# is +cu132, so a host with, say, CUDA 12.8 from apt silently yields a qkan
# without `qkan._C` — the failure is quiet, and the tutorial's whole point is
# the fused kernel.
#
# Rather than install a second CUDA toolkit system-wide, this script takes the
# toolchain from PyPI: nvidia-cuda-nvcc / -cccl / -crt / nvvm, pinned to
# torch.version.cuda so they cannot drift apart, and points CUDA_HOME at the
# wheel tree. Two gaps have to be papered over for that tree to work as a
# CUDA_HOME:
#
#   - the runtime wheels ship only versioned sonames (libcudart.so.13) with no
#     libcudart.so, so `-lcudart` fails to link. Section 4 adds the symlinks.
#   - the nvcc wheel has no CCCL headers, so <nv/target> is missing and every
#     translation unit fails. nvidia-cuda-cccl supplies them.
#
# Re-running is cheap: every step is idempotent and skips work already done.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

VENV="${VENV:-$PWD/.venv}"
# Matches the Dockerfile's CUTLASS_REF. qkan would otherwise clone main.
CUTLASS_REF="${CUTLASS_REF:-v4.7.1}"
CUTLASS_PATH="${CUTLASS_PATH:-$HOME/.cache/qce26/cutlass}"
# Wheel-naming family and PyTorch index tag, as in run.sh / the Dockerfile.
CUDA_FAMILY="${CUDA_FAMILY:-cu13}"
TORCH_TAG="${TORCH_TAG:-cu132}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_TAG}"

if ! command -v uv >/dev/null; then
    echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 1. The virtualenv. An existing one is reused rather than recreated, so a
#    half-finished run can be resumed without re-downloading torch.
if [ ! -x "$VENV/bin/python" ]; then
    echo "==> creating $VENV"
    uv venv --python "${PYTHON_VERSION:-3.12}" "$VENV"
else
    echo "==> reusing $VENV ($("$VENV/bin/python" -V))"
fi
export VIRTUAL_ENV="$VENV"
PY="$VENV/bin/python"

uvpip() { uv pip install --index-strategy unsafe-best-match "$@"; }

# 2. torch first, on its own. qkan reads torch.version.cuda and the C++11 ABI
#    flag off the installed torch to pick nvcc's gencode flags, so it has to be
#    present before qkan builds — this is also why qkan is the one package
#    installed without build isolation further down.
echo "==> torch (${TORCH_TAG})"
uvpip --extra-index-url "$TORCH_INDEX" \
    $(grep -E '^(torch|torchvision)==' requirements.txt)
uvpip "setuptools<82" wheel ninja

# 3. The CUDA toolchain, as wheels, pinned to whatever CUDA torch was built
#    against. Deriving the version from torch rather than hardcoding it is the
#    point: these cannot drift from the torch pin.
TORCH_CUDA="$("$PY" -c 'import torch; print(torch.version.cuda)')"
echo "==> CUDA ${TORCH_CUDA} toolchain from PyPI (matching torch)"
# nvcc alone is not enough: -crt and nvvm are its own back end, and -cccl
# supplies <nv/target> and the rest of libcu++ that CUTLASS includes.
uvpip \
    "nvidia-cuda-nvcc~=${TORCH_CUDA}.0" \
    "nvidia-cuda-cccl~=${TORCH_CUDA}.0" \
    "nvidia-cuda-crt~=${TORCH_CUDA}.0" \
    "nvidia-nvvm~=${TORCH_CUDA}.0"

SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
# CUDA 13 gathers every component under one nvidia/cu13 prefix, which is what
# makes it usable as a CUDA_HOME at all. CUDA 12 wheels scatter themselves over
# per-component directories (nvidia/cuda_runtime, nvidia/cublas, ...) with no
# single root, so this route is cu13-only; on cu12 install a real toolkit and
# set CUDA_HOME yourself before running this script.
CUDA_HOME="${CUDA_HOME:-$SITE/nvidia/${CUDA_FAMILY}}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "No nvcc under $CUDA_HOME."
    [ "$CUDA_FAMILY" = "cu13" ] || echo "  (CUDA_FAMILY=$CUDA_FAMILY: only cu13 ships a single wheel prefix; set CUDA_HOME to a real toolkit)"
    exit 1
fi
export CUDA_HOME CUDA_PATH="$CUDA_HOME" CUDA_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
echo "    CUDA_HOME=$CUDA_HOME  ($("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1))"

# 4. Linker symlinks. The runtime wheels ship libfoo.so.N only; ld needs a bare
#    libfoo.so to resolve -lfoo. The Dockerfile does the same thing for
#    cutensor, for the same reason.
made=0
for f in "$CUDA_HOME"/lib/lib*.so.*; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    link="$CUDA_HOME/lib/${base%%.so.*}.so"
    [ -e "$link" ] || { ln -s "$base" "$link"; made=$((made + 1)); }
done
echo "==> linker symlinks: ${made} added"

# 5. CUTLASS headers, pinned, in a cache outside the repo so a `git clean` here
#    does not throw away 33 MB that never changes. qkan's runtime
#    GPU-compatibility guard rebuilds its kernels from source when the prebuilt
#    SASS does not match the local GPU, so this is kept, not deleted after use.
if [ ! -f "$CUTLASS_PATH/include/cute/tensor.hpp" ]; then
    echo "==> CUTLASS ${CUTLASS_REF} -> $CUTLASS_PATH"
    mkdir -p "$(dirname "$CUTLASS_PATH")"
    rm -rf "$CUTLASS_PATH"
    git clone --depth 1 --branch "$CUTLASS_REF" --filter=blob:none --sparse \
        https://github.com/NVIDIA/cutlass.git "$CUTLASS_PATH" >/dev/null 2>&1
    git -C "$CUTLASS_PATH" sparse-checkout set include >/dev/null
else
    echo "==> CUTLASS already at $CUTLASS_PATH"
fi
export CUTLASS_PATH

# 6. Everything else, including qkan. Build only for the GPU actually in this
#    machine — the image ships 80;90;100;120 because it has to serve every
#    participant, but locally one architecture is a much shorter build.
#    QKAN_FORCE_BUILD=TRUE skips the prebuilt-wheel lookup (the pin is a dev
#    revision with no release) and turns a failed CUDA build into a hard error
#    instead of a silent pure-Python fallback.
ARCHS="${QKAN_CUDA_ARCHS:-$("$PY" -c 'import torch;c=torch.cuda.get_device_capability();print(f"{c[0]}{c[1]}")' 2>/dev/null || echo "80;90;100;120")}"
echo "==> qkan + remaining requirements (QKAN_CUDA_ARCHS=$ARCHS)"
QKAN_FORCE_BUILD=TRUE \
QKAN_CUDA_ARCHS="$ARCHS" \
NVCC_THREADS="${NVCC_THREADS:-8}" \
MAX_JOBS="${MAX_JOBS:-4}" \
    uvpip --extra-index-url "$TORCH_INDEX" \
        --no-build-isolation-package qkan \
        -r requirements.txt

# 7. Self-check. Mirrors the qkan section of the image's qce26-verify: a
#    pure-Python qkan imports perfectly well and is simply slow, so importing
#    qkan is not evidence of anything — qkan._C has to load, and the kernel has
#    to agree with the reference solver.
echo
"$PY" - <<'PYEOF'
import importlib, sys
import importlib.metadata as md
import torch

ok = True
def line(k, v):
    print(f"  {k:<22} {v}")

line("python", sys.version.split()[0])
line("torch", f"{torch.__version__} (CUDA {torch.version.cuda})")
line("qkan", md.version("qkan"))
if torch.cuda.is_available():
    line("GPU", f"{torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability()))}")
else:
    line("GPU", "not available"); ok = False

try:
    importlib.import_module("qkan._C")
    line("qkan._C", "present")
except Exception as exc:
    line("qkan._C", f"MISSING -- {exc}"); ok = False

if ok:
    from qkan import QKANLayer
    torch.manual_seed(0)
    sample = torch.randn(8, 4, device="cuda")
    ref = QKANLayer(4, 3, reps=3, device="cuda", solver="exact")
    alt = QKANLayer(4, 3, reps=3, device="cuda", solver="cute")
    alt.load_state_dict(ref.state_dict())
    with torch.no_grad():
        d = (ref(sample) - alt(sample)).abs().max().item()
    if d < 1e-4:
        line("qkan CuTe kernel", f"OK (matches exact to {d:.1e})")
    else:
        line("qkan CuTe kernel", f"DISAGREES by {d:.1e}"); ok = False

print()
print("venv OK" if ok else "venv has problems (see above)")
print("note: CUDA-Q, cudaq_einsum and the multi-stream backend are not in this")
print("      venv -- notebooks 00, 01 and 04 need the Docker image.")
sys.exit(0 if ok else 1)
PYEOF
