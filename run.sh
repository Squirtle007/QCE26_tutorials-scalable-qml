#!/usr/bin/env bash
# Environment setup for the QCE26 tutorial, targeting CUDA 13.2.
#
# The Dockerfile in this repository is the primary, self-contained path:
#
#   docker build -t cudaq-qce26:cu132 .
#   docker run -it --gpus all -v "$(pwd):/workspace" cudaq-qce26:cu132
#
# This script is the manual equivalent, for when you already have a CUDA-Q
# container and want to set it up in place:
#
#   docker run -it --gpus all \
#       -v "$(pwd):/workspace" -w /workspace \
#       --name cudaq-qce26 \
#       nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.15.1
#   # or, if the container already exists: docker start -ai cudaq-qce26
#
#   bash run.sh
#
# It performs the same steps as the Dockerfile, and each is commented there in
# more detail. In brief:
#   - the published cuda-quantum:cu13-* images ship CUDA *13.0*, so the CUDA
#     13.2 compiler and dev headers are installed first (without them, qkan
#     silently drops its CuTe extension and falls back to pure Python);
#   - uv is used rather than pip, because Ubuntu's python3-pip is dpkg-owned
#     with no RECORD file and cannot upgrade itself;
#   - an import-order hook is installed, because CUDA-Q and PyTorch's bundled
#     triton each embed an LLVM and the second to load aborts the interpreter.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

CUDA_SUFFIX=13-2
CUDA_DIR_VERSION=13.2
# Wheel-naming family, PyTorch index tag and cupy package, mirroring the
# Dockerfile build args of the same names. The requirements template is
# rendered with these, so they must match the variant being installed.
CUDA_FAMILY=cu13
TORCH_TAG=cu132
CUPY_PKG=cupy-cuda13x
CUTLASS_REF=v4.7.1
CUTLASS_PATH="${CUTLASS_PATH:-/opt/cutlass}"
export CUTLASS_PATH
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# 0. Guard against requirements.txt drifting from the Dockerfile's inlined copy,
#    which is authoritative.
if [ -f Dockerfile ]; then
    sed -n "/^COPY <<'REQEOF'/,/^REQEOF$/p" Dockerfile | sed '1d;$d' \
        | sed -e "s/@TORCH_TAG@/${TORCH_TAG}/g" \
              -e "s/@CUDA_FAMILY@/${CUDA_FAMILY}/g" \
              -e "s/@CUPY_PKG@/${CUPY_PKG}/g" > /tmp/req-from-dockerfile.txt
    if ! diff -u --label "Dockerfile (rendered for ${TORCH_TAG})" --label requirements.txt \
             <(grep -v '^--extra-index-url' /tmp/req-from-dockerfile.txt) \
             <(sed -n '/^# --- PyTorch/,$p' requirements.txt); then
        echo "WARNING: requirements.txt differs from the pin block inlined in Dockerfile."
        echo "         The Dockerfile is authoritative; using requirements.txt anyway."
    fi
    rm -f /tmp/req-from-dockerfile.txt
fi

# 1. CUDA 13.2 toolchain and C++ build tools. The base image removes the NVIDIA
#    apt repo, so re-add it via the keyring package.
if ! command -v nvcc >/dev/null || ! nvcc --version | grep -q "release ${CUDA_DIR_VERSION}"; then
    . /etc/os-release
    distro="${ID}${VERSION_ID//./}"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends ca-certificates curl
    curl -fsSL -o /tmp/cuda-keyring.deb \
        "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${NVARCH:?set by the CUDA base image; x86_64 or sbsa}/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        "cuda-nvcc-${CUDA_SUFFIX}" \
        "cuda-nvtx-${CUDA_SUFFIX}" \
        "cuda-profiler-api-${CUDA_SUFFIX}" \
        "cuda-nvrtc-dev-${CUDA_SUFFIX}" \
        "libnvjitlink-dev-${CUDA_SUFFIX}" \
        "libcublas-dev-${CUDA_SUFFIX}" \
        "libcusparse-dev-${CUDA_SUFFIX}" \
        "libcusolver-dev-${CUDA_SUFFIX}"
    sudo ln -sfn "/usr/local/cuda-${CUDA_DIR_VERSION}" /usr/local/cuda
    sudo ldconfig
fi
sudo apt-get install -y --no-install-recommends g++ make cmake git ninja-build

export CUDA_HOME="/usr/local/cuda-${CUDA_DIR_VERSION}"
export CUDA_PATH="$CUDA_HOME"
export CUDA_ROOT="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 2. CUTLASS headers for qkan's CuTe kernels. Kept afterwards: qkan rebuilds
#    them at runtime when the prebuilt SASS does not match the local GPU.
if [ ! -f "$CUTLASS_PATH/include/cute/tensor.hpp" ]; then
    sudo git clone --depth 1 --branch "$CUTLASS_REF" --filter=blob:none --sparse \
        https://github.com/NVIDIA/cutlass.git "$CUTLASS_PATH"
    sudo git -C "$CUTLASS_PATH" sparse-checkout set include
fi

# 3. uv. Installing pip's replacement rather than upgrading pip sidesteps
#    "Cannot uninstall pip 24.0, RECORD file not found" on dpkg-owned pip.
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

# The container's dist-packages is root-owned (the base image pip-installs as
# root), but this script runs as the non-root `cudaq` user, so installs that
# replace an existing system file fail with EACCES. The Dockerfile does not hit
# this because it builds as root. Route every system-site install through sudo,
# preserving the UV_* settings; uv's absolute path is needed because sudo resets
# PATH and uv lives under $HOME/.local/bin.
UV_BIN="$(command -v uv)"
uvpip() { sudo -E "$UV_BIN" pip install "$@"; }

# 4. PyTorch first, on its own: qkan compiles a CUDA extension against the
#    installed torch, reading torch.version.cuda and the C++11 ABI flag off it.
uvpip --index-strategy unsafe-best-match \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_TAG}" \
    $(grep -E '^(torch|torchvision)==' requirements.txt)
uvpip "setuptools<82" wheel ninja
python3 -c "import torch; assert torch.version.cuda == '13.2', torch.version.cuda"

# 5. Everything else. --no-build-isolation-package qkan lets only qkan build
#    against the installed torch. QKAN_FORCE_BUILD=TRUE skips the pre-built-wheel
#    lookup (the pinned commit is a dev revision with no release) and turns a
#    failed CUDA build into a hard error rather than a silent pure-Python
#    fallback.
QKAN_FORCE_BUILD=TRUE \
QKAN_CUDA_ARCHS="${QKAN_CUDA_ARCHS:-80;90;100;120}" \
NVCC_THREADS="${NVCC_THREADS:-8}" \
MAX_JOBS="${MAX_JOBS:-4}" \
    uvpip --index-strategy unsafe-best-match \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_TAG}" \
        --no-build-isolation-package qkan \
        -r requirements.txt

# 6. LLVM import-order hook. CUDA-Q and triton (bundled with PyTorch, pulled in
#    by torchvision and qkan) each embed an LLVM; whichever loads second aborts
#    with "Option 'debug-counter' registered more than once!". The hook pulls
#    triton in just before the first `import cudaq`, so notebooks need no import
#    discipline. Extracted from the Dockerfile to keep one copy of the source.
sed -n "/^COPY <<'PYEOF' .*qce26_import_order.py$/,/^PYEOF$/p" Dockerfile \
    | sed '1d;$d' | sudo tee "$SITE_PACKAGES/qce26_import_order.py" >/dev/null
echo "import qce26_import_order" | sudo tee "$SITE_PACKAGES/zz-qce26-import-order.pth" >/dev/null

# 7. cudaq_einsum and the multi-stream cuTENSOR backend, both needed by
#    04_cutn-qsvm.ipynb and neither on PyPI. Three one-line compatibility fixes
#    are applied because the repository targets an older CUDA-Q / cuQuantum; see
#    the Dockerfile section 6 comments for what each one is. Every patch is
#    verified, so this fails loudly if upstream moves the line.
export CUDAQ_ROOT=/opt/nvidia/cudaq
EINSUM_REPO="${EINSUM_REPO:-https://github.com/gilbert12tw/Einsum-Simulator.git}"
EINSUM_REF="${EINSUM_REF:-b40e3d1b371d4eaa5563afc3c2871964bb588257}"
EINSUM_SRC="${EINSUM_SRC:-/opt/einsum-simulator}"

if [ ! -d "$EINSUM_SRC" ]; then
    sudo git clone --filter=blob:none "$EINSUM_REPO" "$EINSUM_SRC"
    sudo git -C "$EINSUM_SRC" checkout --detach "$EINSUM_REF"
    sudo chown -R "$(id -u):$(id -g)" "$EINSUM_SRC"
fi
(
    cd "$EINSUM_SRC"
    sed -i 's/void setNoiseModel(cudaq::noise_model& noise) override {/void setNoiseModel(cudaq::noise_model\& noise) {/' cpp/EinsumSimulator.cpp
    ! grep -q 'setNoiseModel(cudaq::noise_model& noise) override' cpp/EinsumSimulator.cpp
    sed -i 's/^\( *\)from cuquantum import contract$/\1from cuquantum.tensornet import contract/' src/cudaq_einsum/batched.py
    sed -i 's/^\( *\)from cuquantum import Network, NetworkOptions$/\1from cuquantum.tensornet import Network, NetworkOptions/' src/cudaq_einsum/qsvm_cudaq_cpp_backend.py
    ! grep -rqn '^ *from cuquantum import ' src/cudaq_einsum/
    uvpip .
)
sudo -E python3 -c "import cudaq_einsum; cudaq_einsum.install_cudaq_target()"
python3 -c "import cudaq; cudaq.set_target('einsum')"

# The multi-stream backend links -lcutensor, but the cutensor wheel ships only
# the runtime soname; add the linker symlink. Must come after step 5, which is
# what installs the cutensor being linked against.
sudo ln -sfn libcutensor.so.2 "$CUTENSOR_ROOT/lib/libcutensor.so"
sudo ln -sfn libcutensorMg.so.2 "$CUTENSOR_ROOT/lib/libcutensorMg.so"
uvpip pybind11
( cd "$EINSUM_SRC/cpp_backend" && CUDA_ROOT="$CUDA_HOME" python3 setup.py build_ext --inplace )
sudo cp "$EINSUM_SRC"/cpp_backend/qsvm_cutensor_backend*.so "$SITE_PACKAGES/"

# 04_cutn-qsvm.ipynb imports qsvm_cudaq_cpp_backend flat; the repository ships it
# inside the cudaq_einsum package. Alias so both spellings resolve to one object.
printf '%s\n' \
    '"""Alias: `import qsvm_cudaq_cpp_backend` -> cudaq_einsum.qsvm_cudaq_cpp_backend."""' \
    'import sys' \
    'from cudaq_einsum import qsvm_cudaq_cpp_backend as _m' \
    'sys.modules[__name__] = _m' \
    | sudo tee "$SITE_PACKAGES/qsvm_cudaq_cpp_backend.py" >/dev/null

# 8. CUDA forward-compatibility guard.
#
# The base image ships /usr/local/cuda-13.0/compat containing a bundled libcuda
# that may be NEWER than the host driver. At container start the NVIDIA runtime
# drops an /etc/ld.so.conf.d/00-compat-*.conf pointing at it, so libcuda.so.1
# resolves to the compat copy. On a data-centre GPU that is the intended forward
# compatibility path; on GeForce it is unsupported and every CUDA call fails with
# error 804 ("forward compatibility was attempted on non supported HW").
#
# The Dockerfile sidesteps this by repointing /usr/local/cuda at 13.2, which has
# no compat dir, so the runtime never injects the conf. This script cannot: the
# conf already exists by the time it runs. So probe cuInit and disable the compat
# path only when it is actually broken, leaving legitimate forward compatibility
# on older data-centre drivers alone.
CUINIT_PROBE='import ctypes
try:
    print(ctypes.CDLL("libcuda.so.1").cuInit(0))
except Exception:
    print(999)'
cuinit_status() { python3 -c "$CUINIT_PROBE" 2>/dev/null || echo 999; }

if [ "$(cuinit_status)" = "804" ]; then
    echo
    echo "NOTE: CUDA forward compatibility is active but unsupported on this GPU"
    echo "      (cuInit returned 804). Disabling the bundled compat libcuda so the"
    echo "      host driver is used instead."
    sudo rm -f /etc/ld.so.conf.d/00-compat-*.conf
    sudo ldconfig
    st=$(cuinit_status)
    if [ "$st" = "0" ]; then
        echo "      Resolved: libcuda.so.1 -> $(ldconfig -p | grep -m1 'libcuda.so.1' | sed 's/.*=> //')"
    else
        echo "      WARNING: cuInit still returns $st; the GPU may be unusable here."
    fi
fi

# 9. Self-check, using the same script the image ships.
sed -n "/^COPY <<'PYEOF' \/usr\/local\/bin\/qce26-verify$/,/^PYEOF$/p" Dockerfile \
    | sed '1d;$d' | sudo tee /usr/local/bin/qce26-verify >/dev/null
sudo chmod +x /usr/local/bin/qce26-verify
echo
qce26-verify
