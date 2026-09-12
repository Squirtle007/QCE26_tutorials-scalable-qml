# syntax=docker/dockerfile:1.7
#
# QCE26 tutorial image — "Scalable Validation and Optimized Simulation for
# Quantum Machine Learning", on CUDA 13.2.
#
#   docker build -t cudaq-qce26:cu132 .
#   docker run -it --gpus all -v "$(pwd):/workspace" cudaq-qce26:cu132
#   # inside: qce26-verify   (environment self-check)
#   #         jupyter lab --ip=0.0.0.0 --no-browser --allow-root
#
# Self-contained: the dependency pins, the import-order fix and the self-check
# are all inlined below, so no other file from the repository is required. The
# build context is only consulted for the optional cudaq_einsum source tree
# (section 6). Requires BuildKit (default since Docker 23) for heredocs and the
# bind mount.
#
# Build cost is dominated by compiling qkan's CuTe kernels for four GPU
# architectures (~14 min). Pass --build-arg QKAN_CUDA_ARCHS=80 for an
# A100-only image.
#
#
# Five things this image has to work around
# -----------------------------------------
# 1. No CUDA-Q image ships CUDA 13.2. Every published
#    nvcr.io/nvidia/quantum/cuda-quantum:cu13-* tag is CUDA *13.0*. Installing
#    +cu132 PyTorch on top is not sufficient: qkan's setup.py compares system
#    nvcc against torch.version.cuda and silently drops its CuTe extension when
#    they differ, leaving a pure-Python qkan. Section 1 installs the CUDA 13.2
#    compiler and the dev headers reachable from ATen/cuda/CUDAContext.h, then
#    repoints CUDA_HOME.
#
# 2. `pip install --upgrade pip` cannot work here. Ubuntu's python3-pip is
#    dpkg-owned with no RECORD file, so pip refuses to uninstall itself
#    ("Cannot uninstall pip 24.0"). uv installs alongside it instead, which
#    avoids the problem entirely — one of the reasons this image uses uv.
#
# 3. `qkan[cute]` fails on every CUDA version. The extra expands to
#    nvidia-cublas-cu13 / nvidia-cuda-runtime-cu13 / etc.; NVIDIA renamed those
#    wheels for CUDA 13 (now unsuffixed, under nvidia/cu13/) and the old names
#    are stub sdists whose setup.py calls sys.exit(1) on any wheel build. The
#    extra is not requested; its useful member, ninja, is pinned directly.
#
# 4. `cuquantum-python` (the meta sdist) imports pkg_resources at build time,
#    which setuptools >= 82.0.0 no longer ships. The -cu13 wheel is pinned
#    directly instead. Its component libraries are pinned too — see the note in
#    the requirements block.
#
# 5. CUDA-Q and triton each embed an LLVM, and whichever loads second aborts
#    the interpreter. Section 5 installs a lazy import hook that fixes the
#    ordering globally.
#
# Driver requirement: NVIDIA_REQUIRE_CUDA is deliberately left at the base
# image's `cuda>=13.0`. CUDA minor version compatibility lets +cu132 wheels run
# on any driver supporting CUDA 13.0 (>= 580); this image was verified on a
# 580 driver. Set INSTALL_CUDA_COMPAT=true to add cuda-compat-13-2 for full
# 13.2 driver features on data-centre GPUs with an older driver; leave it false
# on GeForce, where forward compatibility is not supported.

# Two variants are built from this one file. Defaults target CUDA 13.2; pass
# the CUDA 12.6 set for hosts whose driver predates CUDA 13 (see the driver note
# below). Every cuXX-specific string is derived from these args.
#
#   CUDA 13.2 (driver >= 580)      -- the defaults
#   CUDA 12.6 (driver >= 525):
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.15.1 \
#     --build-arg CUDA_SUFFIX=12-6 --build-arg CUDA_DIR_VERSION=12.6 \
#     --build-arg CUDA_FAMILY=cu12 --build-arg TORCH_TAG=cu126 \
#     --build-arg CUPY_PKG=cupy-cuda12x
ARG BASE_IMAGE=nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.15.1
ARG UV_VERSION=0.12.11

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# CUDA 13.2 apt package suffix and the matching /usr/local/cuda-<ver> prefix.
ARG CUDA_SUFFIX=13-2
ARG CUDA_DIR_VERSION=13.2
# Wheel-naming family for the cuQuantum / cupy packages, and the PyTorch index
# holding the matching +cuXXX local versions.
ARG CUDA_FAMILY=cu13
ARG TORCH_TAG=cu132
ARG CUPY_PKG=cupy-cuda13x
# Adds cuda-compat-13-2 (forward compatibility for data-centre GPUs on older
# drivers). Not supported on GeForce — leave false unless you need it.
ARG INSTALL_CUDA_COMPAT=false
# CUTLASS supplies cute/tensor.hpp and cutlass/float8.h for qkan's kernels.
# Pinned so the image is reproducible; qkan would otherwise clone main.
ARG CUTLASS_REF=v4.7.1
# A100 (sm_80), H100 (sm_90), B200 (sm_100), RTX 50 (sm_120), plus PTX for
# newer chips. Narrow this to shorten the build, e.g. "80" for Brev A100 only.
ARG QKAN_CUDA_ARCHS="80;90;100;120"
ARG NVCC_THREADS=8
ARG MAX_JOBS=4

# ARG, not ENV: apt needs it during the build, but baking it into the image
# would silently change apt's behaviour for anyone working inside a container.
ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. CUDA 13.2 toolchain
#
# The base image deletes /etc/apt/sources.list.d/cuda.list, so the NVIDIA repo
# is re-added via the keyring package. Only the compiler and the dev headers
# reachable from ATen/cuda/CUDAContext.h (cublas / cusparse / cusolver) are
# installed — the cuda-toolkit-13-2 meta-package drags in Nsight and the
# documentation for several extra GB. cuda-nvcc pulls cudart-dev, cccl and
# build-essential itself.
# ---------------------------------------------------------------------------
RUN set -euo pipefail \
 && . /etc/os-release \
 && distro="${ID}${VERSION_ID//./}" \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL -o /tmp/cuda-keyring.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${NVARCH:?set by the CUDA base image; x86_64 or sbsa}/cuda-keyring_1.1-1_all.deb" \
 && dpkg -i /tmp/cuda-keyring.deb \
 && rm -f /tmp/cuda-keyring.deb \
 && apt-get update \
# The base image apt-mark holds its own libcublas so nothing upgrades it by
# accident. When the variant's CUDA minor equals the base image's (the cu12
# case: base is 12.6 and we want the 12-6 dev headers) the held runtime package
# and the -dev package's dependency are the same package, and the hold makes the
# install unsatisfiable:
#   libcublas-dev-12-6 : Depends: libcublas-12-6 (>= 12.6.4.1)
#                        but 12.6.0.22-1 is to be installed
# --allow-change-held-packages does NOT resolve this; the hold has to be
# released. Only the one package is unheld, and only for the CUDA minor being
# installed, so libnccl2 and any other hold stay put. A no-op when not held,
# which is the cu13 case (it holds 13-0 and installs 13-2).
 && apt-mark unhold libcublas-${CUDA_SUFFIX} >/dev/null 2>&1 || true \
 && apt-get install -y --no-install-recommends \
      cuda-nvcc-${CUDA_SUFFIX} \
      cuda-nvtx-${CUDA_SUFFIX} \
      cuda-profiler-api-${CUDA_SUFFIX} \
      cuda-nvrtc-dev-${CUDA_SUFFIX} \
      libnvjitlink-dev-${CUDA_SUFFIX} \
      libcublas-dev-${CUDA_SUFFIX} \
      libcusparse-dev-${CUDA_SUFFIX} \
      libcusolver-dev-${CUDA_SUFFIX} \
 && if [ "${INSTALL_CUDA_COMPAT}" = "true" ]; then \
      apt-get install -y --no-install-recommends cuda-compat-${CUDA_SUFFIX}; \
    fi \
 && apt-get install -y --no-install-recommends g++ make cmake git ninja-build \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
# Reclaim ~1.2 GB of math-library static archives that nothing here links.
# libcudart_static.a is deliberately kept: CMake defaults CUDA_RUNTIME_LIBRARY
# to Static, so the cudaq_einsum build in section 6 may need it.
 && cudadir="/usr/local/cuda-${CUDA_DIR_VERSION}" \
 && find "$cudadir" -name 'libcublas*_static.a' -delete \
 && find "$cudadir" -name 'libcusparse*_static.a' -delete \
 && find "$cudadir" -name 'libcusolver*_static.a' -delete \
# Repoint /usr/local/cuda, which /etc/ld.so.conf.d/nvidia.conf and the base
# image's LD_LIBRARY_PATH both reference, at 13.2.
 && ln -sfn "$cudadir" /usr/local/cuda \
 && ldconfig

# nvcc must resolve to 13.2 ahead of the base image's cuda-13.0 PATH entry, and
# CUDA_HOME is what qkan's setup.py probes for the bare-metal CUDA version.
ENV CUDA_VERSION=${CUDA_DIR_VERSION} \
    CUDA_INSTALL_PREFIX=/usr/local/cuda-${CUDA_DIR_VERSION} \
    CUDA_HOME=/usr/local/cuda-${CUDA_DIR_VERSION} \
    CUDA_ROOT=/usr/local/cuda-${CUDA_DIR_VERSION} \
    CUDA_PATH=/usr/local/cuda-${CUDA_DIR_VERSION}
ENV PATH=/usr/local/cuda-${CUDA_DIR_VERSION}/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_DIR_VERSION}/lib64:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# 2. CUTLASS headers
#
# Header-only, so a sparse checkout of include/ is enough. Kept in the image on
# purpose: qkan's runtime GPU-compatibility guard JIT-rebuilds its kernels from
# the .cu sources shipped in the wheel when the prebuilt SASS does not match the
# local GPU, and that rebuild needs CUTLASS, nvcc and ninja to still be here.
# ---------------------------------------------------------------------------
# Recorded so qce26-verify and the build-time checks assert against the variant
# actually built, rather than a hardcoded 13.2.
ENV QCE26_CUDA_VERSION=${CUDA_DIR_VERSION}
ENV CUTLASS_PATH=/opt/cutlass
RUN set -euo pipefail \
 && git clone --depth 1 --branch "${CUTLASS_REF}" --filter=blob:none --sparse \
      https://github.com/NVIDIA/cutlass.git "${CUTLASS_PATH}" \
 && git -C "${CUTLASS_PATH}" sparse-checkout set include \
 && rm -rf "${CUTLASS_PATH}/.git" \
 && test -f "${CUTLASS_PATH}/include/cute/tensor.hpp"

# ---------------------------------------------------------------------------
# 3. uv
#
# UV_SYSTEM_PYTHON / UV_BREAK_SYSTEM_PACKAGES target the container's system
# interpreter, which is the one CUDA-Q's cudaq.pth is registered against.
# ---------------------------------------------------------------------------
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

# ---------------------------------------------------------------------------
# 4. Python dependencies
#
# The pins live here rather than in a copied requirements.txt so this file is
# self-contained. A copy is left at /opt/qce26/requirements.txt in the image.
# ---------------------------------------------------------------------------
COPY <<'REQEOF' /opt/qce26/requirements.txt.in
# --- PyTorch, built against CUDA 13.2 -------------------------------------
# torch 2.11.0 has no cu132 build; that index starts at 2.12.0. 2.12.1 is
# the smallest step up from the version the notebooks were developed against.
# Both wheels pull nvidia's cuda-toolkit==13.2.1 runtime wheels.
torch==2.12.1+@TORCH_TAG@
torchvision==0.27.1+@TORCH_TAG@

tqdm==4.67.3
scikit-learn==1.8.0

# --- cuQuantum -------------------------------------------------------------
# The -cu13 wheel, not the `cuquantum-python` meta sdist: that sdist imports
# pkg_resources at build time, removed in setuptools 82.0.0. 26.3.x is also the
# series matching CUDA-Q 0.15.x; 26.1.0 pairs with 0.14.x and conflicts on
# cudensitymat.
cuquantum-python-@CUDA_FAMILY@==26.3.2

# Component pins. These matter because CUDA-Q is installed natively in the base
# image (/opt/nvidia/cudaq), not as a pip package, so the resolver cannot see
# its constraints — and the image puts dist-packages/cuquantum/lib first on
# LD_LIBRARY_PATH, so these wheels are the libraries CUDA-Q actually loads.
# Unpinned, cuquantum-python's loose upper bounds (<2, <3, <0.6) pull
# custatevec 1.14 / cutensornet 2.13 / cupy 14, all outside what the metadata
# for cuda-quantum-cu13 0.15.1 declares it supports.
custatevec-@CUDA_FAMILY@==1.13.1
cutensornet-@CUDA_FAMILY@==2.12.2
cudensitymat-@CUDA_FAMILY@==0.5.2
@CUPY_PKG@==13.6.0

# --- build tooling ---------------------------------------------------------
# cmake: cudaq_einsum's build shells out to it (also installed from apt, so a
# `cmake` is on PATH regardless of install order).
cmake==4.3.2
# ninja: needed to build qkan._C below, and again at runtime — qkan's GPU
# compatibility guard JIT-rebuilds its kernels when the prebuilt SASS does not
# match the local GPU.
ninja
# setuptools and wheel are installed by the Dockerfile layer below, before this
# file is read: qkan builds with --no-build-isolation-package and so needs them
# already present, not merely listed here.

# --- qkan ------------------------------------------------------------------
# Built from source: the pinned commit is a 0.2.4dev revision, so no pre-built
# wheel exists on the GitHub release page.
#
# The [cute] extra is deliberately NOT requested — it expands to deprecated
# nvidia-*-cu13 stub sdists that abort on wheel build, so `pip install
# qkan[cute]` fails outright on any CUDA version. It only existed to pull
# runtime CUDA libraries into environments lacking them; this image has them
# from both the apt CUDA 13.2 toolkit and torch's cuda-toolkit==13.2.1 wheels.
qkan @ git+https://github.com/Jim137/qkan.git@549c1aa6bbff7f7264a9e60d25d38165176dc194

# --- notebook interface ----------------------------------------------------
jupyterlab
ipywidgets

# NOTE: cudaq_einsum and the multi-stream cuTENSOR backend that 04_cutn-qsvm.ipynb
# needs are not on PyPI. They are built from the pinned Einsum-Simulator
# repository in section 6 of the Dockerfile, not installed from here.
REQEOF

# torch first, in its own layer: qkan compiles a CUDA extension against the
# installed torch, reading torch.version.cuda and the C++11 ABI flag off it to
# pick nvcc's gencode flags. unsafe-best-match is required because uv's default
# first-index strategy would only look at PyPI for `torch`, which does not carry
# the +cu132 local versions.
RUN set -euo pipefail \
 && sed -e "s/@TORCH_TAG@/${TORCH_TAG}/g" \
        -e "s/@CUDA_FAMILY@/${CUDA_FAMILY}/g" \
        -e "s/@CUPY_PKG@/${CUPY_PKG}/g" \
        /opt/qce26/requirements.txt.in > /opt/qce26/requirements.txt \
# Check for unsubstituted @TOKEN@ placeholders specifically — a bare '@' also
# matches the legitimate `qkan @ git+https://...` PEP 508 direct reference.
 && ! grep -qE '@[A-Z_]+@' /opt/qce26/requirements.txt \
 && uv pip install --index-strategy unsafe-best-match \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_TAG}" \
      $(grep -E '^(torch|torchvision)==' /opt/qce26/requirements.txt) \
 && uv pip install "setuptools<82" wheel ninja \
 && python3 -c "import torch, os; assert torch.version.cuda == os.environ['CUDA_DIR_VERSION'], torch.version.cuda; print(torch.__version__)"

# --no-build-isolation-package qkan is uv's targeted form of pip's global
# --no-build-isolation: only qkan builds against the installed torch/setuptools/
# ninja, everything else keeps a clean isolated build.
#
# QKAN_FORCE_BUILD=TRUE skips the pre-built-wheel lookup (the pinned dev commit
# has no matching release) and makes a failed CUDA build a hard error instead of
# a silent pure-Python fallback.
RUN set -euo pipefail \
 && QKAN_FORCE_BUILD=TRUE \
    QKAN_CUDA_ARCHS="${QKAN_CUDA_ARCHS}" \
    NVCC_THREADS="${NVCC_THREADS}" \
    MAX_JOBS="${MAX_JOBS}" \
    uv pip install --index-strategy unsafe-best-match \
      --extra-index-url "https://download.pytorch.org/whl/${TORCH_TAG}" \
      --no-build-isolation-package qkan \
      -r /opt/qce26/requirements.txt

# ---------------------------------------------------------------------------
# 5. LLVM import-order fix
#
# CUDA-Q embeds an LLVM/MLIR, and so does triton (bundled with PyTorch, pulled
# in by torchvision and qkan). Whichever loads second re-registers LLVM's global
# command-line options and aborts the interpreter:
#
#     CommandLine Error: Option 'debug-counter' registered more than once!
#     LLVM ERROR: inconsistency in registered CommandLine options
#
# `import triton, cudaq` is fine; `import cudaq, triton` dies. This is inherent
# to having CUDA-Q and PyTorch in one environment — not specific to CUDA 13.2 —
# and it is a nasty thing to hit live, because `import cudaq, torch` looks fine
# (torch does not load triton eagerly) right up until a cell adds torchvision.
#
# The .pth below arms a meta-path hook at interpreter start that pulls triton in
# just before the first `import cudaq`. It costs nothing until cudaq is imported
# and does nothing when triton is absent, so notebooks need no import discipline.
# ---------------------------------------------------------------------------
COPY <<'PYEOF' /usr/local/lib/python3.12/dist-packages/qce26_import_order.py
"""Make `import cudaq` safe no matter when torch/triton is imported.

CUDA-Q and triton each embed their own LLVM. Whichever loads second aborts the
interpreter with "Option 'debug-counter' registered more than once!". Loading
triton first is fine, so this hook watches for the first `import cudaq` and
pulls triton in ahead of it.
"""

import sys
from importlib.abc import MetaPathFinder


class _TritonBeforeCudaq(MetaPathFinder):
    _armed = True

    def find_spec(self, fullname, path=None, target=None):
        if not self._armed or fullname.partition(".")[0] != "cudaq":
            return None
        # Disarm first so the nested import cannot recurse back into here.
        self._armed = False
        try:
            import triton  # noqa: F401
        except Exception:  # triton absent or broken: nothing to order against
            pass
        return None  # never claim the module; defer to the real finders


if not any(isinstance(f, _TritonBeforeCudaq) for f in sys.meta_path):
    sys.meta_path.insert(0, _TritonBeforeCudaq())
PYEOF

COPY <<'PTHEOF' /usr/local/lib/python3.12/dist-packages/zz-qce26-import-order.pth
import qce26_import_order
PTHEOF

# ---------------------------------------------------------------------------
# 6. cudaq_einsum + the multi-stream cuTENSOR backend
#
# 04_cutn-qsvm.ipynb needs both: `cudaq_einsum` from Section 1 onward, and
# `qsvm_cudaq_cpp_backend` for the Section 3 multi-stream comparison. Neither is
# on PyPI; both live in the Einsum-Simulator repository, pinned here.
#
# That repository targets an older CUDA-Q / cuQuantum, so three one-line
# compatibility fixes are applied below. Each patch is verified to have changed
# something, so the build fails loudly if upstream moves the line rather than
# silently producing an image that breaks at notebook runtime. These belong
# upstream — once they land, drop the patches and bump EINSUM_REF.
# ---------------------------------------------------------------------------
ARG EINSUM_REPO=https://github.com/gilbert12tw/Einsum-Simulator.git
ARG EINSUM_REF=b40e3d1b371d4eaa5563afc3c2871964bb588257
ENV CUDAQ_ROOT=/opt/nvidia/cudaq

RUN set -euo pipefail \
 && git clone --filter=blob:none "${EINSUM_REPO}" /opt/einsum-simulator \
 && git -C /opt/einsum-simulator checkout --detach "${EINSUM_REF}" \
 && cd /opt/einsum-simulator \
# patch 1 — CUDA-Q >= 0.15 removed the CircuitSimulator::setNoiseModel virtual,
# so the `override` no longer compiles ("marked override, but does not
# override"). The method is unreachable either way: this simulator rejects noise
# models, and 0.15 routes noise through applyNoise() on the base class.
 && sed -i 's/void setNoiseModel(cudaq::noise_model& noise) override {/void setNoiseModel(cudaq::noise_model\& noise) {/' cpp/EinsumSimulator.cpp \
 && ! grep -q 'setNoiseModel(cudaq::noise_model& noise) override' cpp/EinsumSimulator.cpp \
# patches 2 and 3 — cuQuantum 26.x moved Network / NetworkOptions / contract out
# of the top-level namespace into cuquantum.tensornet. Both are function-local
# imports, so they fail at call time rather than on `import cudaq_einsum`.
 && sed -i 's/^\( *\)from cuquantum import contract$/\1from cuquantum.tensornet import contract/' src/cudaq_einsum/batched.py \
 && sed -i 's/^\( *\)from cuquantum import Network, NetworkOptions$/\1from cuquantum.tensornet import Network, NetworkOptions/' src/cudaq_einsum/qsvm_cudaq_cpp_backend.py \
 && ! grep -rqn '^ *from cuquantum import ' src/cudaq_einsum/ \
# Build the NVQIR plugin against CUDA-Q, then copy libnvqir-einsum.so and
# einsum.yml into the CUDA-Q tree — pip bundles them inside the Python package
# only, and without this step cudaq.set_target("einsum") reports
# "Invalid target name (einsum)".
 && uv pip install . \
 && python3 -c "import cudaq_einsum; cudaq_einsum.install_cudaq_target()" \
 && python3 -c "import cudaq; cudaq.set_target('einsum')"

# The multi-stream backend is a separate pybind11 extension. It links -lcutensor,
# but the cutensor wheel ships only the runtime soname (libcutensor.so.2) with no
# linker symlink, so one is added here. This has to happen after section 4, which
# is what installs the cutensor version being linked against.
RUN set -euo pipefail \
 && ln -sfn libcutensor.so.2 "${CUTENSOR_ROOT}/lib/libcutensor.so" \
 && ln -sfn libcutensorMg.so.2 "${CUTENSOR_ROOT}/lib/libcutensorMg.so" \
 && uv pip install pybind11 \
 && cd /opt/einsum-simulator/cpp_backend \
 && CUDA_ROOT="${CUDA_HOME}" python3 setup.py build_ext --inplace \
 && site=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
 && cp qsvm_cutensor_backend*.so "$site/" \
# 04_cutn-qsvm.ipynb imports `qsvm_cudaq_cpp_backend` flat, after putting a
# bundle directory on sys.path; the repository ships it inside the cudaq_einsum
# package instead. Alias the module so both spellings resolve to one object.
 && printf '%s\n' \
      '"""Alias: `import qsvm_cudaq_cpp_backend` -> cudaq_einsum.qsvm_cudaq_cpp_backend."""' \
      'import sys' \
      'from cudaq_einsum import qsvm_cudaq_cpp_backend as _m' \
      'sys.modules[__name__] = _m' \
      > "$site/qsvm_cudaq_cpp_backend.py" \
 && python3 -c "import qsvm_cudaq_cpp_backend, qsvm_cutensor_backend"

# ---------------------------------------------------------------------------
# 7. Self-check
# ---------------------------------------------------------------------------
COPY <<'PYEOF' /usr/local/bin/qce26-verify
#!/usr/bin/env python3
"""Report the state of the QCE26 tutorial environment."""

import importlib
import os
import shutil
import subprocess
import sys

ok = True


def line(label, value):
    print(f"{label:<26} {value}")


def version(name):
    return getattr(importlib.import_module(name), "__version__", "unknown")


def fail(msg):
    global ok
    print(f"  !! {msg}")
    ok = False


want_cuda = os.environ.get("QCE26_CUDA_VERSION", "13.2")

nvcc = shutil.which("nvcc")
if nvcc:
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    line("nvcc", next((l.strip() for l in out.splitlines() if "release" in l), "?"))
    if f"release {want_cuda}" not in out:
        fail(f"expected CUDA {want_cuda}")
else:
    line("nvcc", "NOT FOUND")
    ok = False

import torch  # noqa: E402

line("torch", f"{torch.__version__} (CUDA {torch.version.cuda})")
if torch.version.cuda != want_cuda:
    fail(f"expected a CUDA {want_cuda} torch build, got {torch.version.cuda}")

for label, mod in [
    ("torchvision", "torchvision"),
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
]:
    line(label, version(mod))

import qkan  # noqa: E402

try:
    importlib.import_module("qkan._C")
    line("qkan", f"{qkan.__version__} (CuTe extension present)")
except ImportError as exc:
    line("qkan", f"{qkan.__version__} (CuTe extension MISSING)")
    fail(f"qkan._C did not load: {exc}")

import cudaq  # noqa: E402

line("cuda-quantum", getattr(cudaq, "__version__", "unknown"))

import cuquantum  # noqa: E402
from cuquantum import tensornet  # noqa: E402,F401

line("cuquantum-python", cuquantum.__version__)
if not cuquantum.__version__.startswith("26.3"):
    fail("cuquantum-python 26.3.x is required by CUDA-Q 0.15.x")

try:
    import cudaq_einsum  # noqa: F401

    line("cudaq_einsum", "installed")
except ImportError as exc:
    line("cudaq_einsum", "MISSING")
    fail(f"04_cutn-qsvm.ipynb needs it: {exc}")

try:
    import qsvm_cudaq_cpp_backend  # noqa: F401
    import qsvm_cutensor_backend  # noqa: F401

    line("multi-stream backend", "qsvm_cutensor_backend built")
except ImportError as exc:
    line("multi-stream backend", "MISSING")
    fail(f"04_cutn-qsvm.ipynb section 3 needs it: {exc}")

# The LLVM double-registration hazard, as a live regression test rather than a
# caveat in a comment: cudaq-before-triton is the order that used to abort.
clash = subprocess.run(
    [sys.executable, "-c", "import cudaq, triton"], capture_output=True, text=True
)
if clash.returncode == 0:
    line("cudaq/triton LLVM order", "handled by qce26_import_order hook")
else:
    line("cudaq/triton LLVM order", "BROKEN")
    fail("`import cudaq, triton` aborts; the qce26_import_order hook is not active")

def driver_cuda_version():
    """CUDA version the *driver* supports, as an int like 13020, or None."""
    import ctypes

    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None  # no driver in this container at all
    v = ctypes.c_int()
    return v.value if lib.cuDriverGetVersion(ctypes.byref(v)) == 0 else None


print()
if torch.cuda.is_available():
    line("GPU", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    line("compute capability", "sm_%d%d" % cap)

    # Actually compute. Querying the device is not enough: a torch build whose
    # kernels do not cover this GPU reports the name and capability happily and
    # then dies with "no kernel image is available for execution on the device"
    # the moment real work is issued. Same for a driver too old for the runtime.
    # Both cases previously reported a healthy environment.
    try:
        a = torch.randn(512, 512, device="cuda")
        err = (a @ a).cpu().sub_(a.cpu() @ a.cpu()).abs().max().item()
        assert err < 1e-2, f"matmul disagrees with CPU by {err}"
        line("torch GPU matmul", "OK")
    except Exception as exc:  # noqa: BLE001
        line("torch GPU matmul", "FAILED")
        arch = ", ".join(torch.cuda.get_arch_list())
        fail(f"{exc}")
        if "no kernel image" in str(exc):
            fail(f"this torch covers [{arch}] but the GPU is sm_{cap[0]}{cap[1]}")

    try:
        importlib.import_module("qkan._C")
        from qkan import QKANLayer

        # Not named `x`: the cudaq kernel below uses the bare gate names h/x/mz,
        # and a torch tensor called x in the enclosing scope shadows the Pauli-X.
        sample_in = torch.randn(8, 4, device="cuda")
        ref = QKANLayer(4, 3, reps=3, device="cuda", solver="exact")
        alt = QKANLayer(4, 3, reps=3, device="cuda", solver="cute")
        alt.load_state_dict(ref.state_dict())
        d = (alt(sample_in) - ref(sample_in)).abs().max().item()
        assert d < 1e-4, f"cute vs exact differ by {d}"
        line("qkan CuTe kernel", f"OK (matches exact solver to {d:.1e})")
    except Exception as exc:  # noqa: BLE001
        line("qkan CuTe kernel", "FAILED")
        fail(str(exc))

    for target in ("nvidia", "einsum"):
        try:
            cudaq.set_target(target)
            line(f"cudaq target {target}", cudaq.get_target().name)
        except Exception as exc:  # noqa: BLE001
            line(f"cudaq target {target}", "FAILED")
            fail(str(exc))

    try:
        cudaq.set_target("nvidia")

        @cudaq.kernel
        def _bell():
            q = cudaq.qvector(2)
            h(q[0])
            x.ctrl(q[0], q[1])
            mz(q)

        counts = cudaq.sample(_bell, shots_count=500)
        clean = counts.count("00") + counts.count("11")
        assert clean > 450, f"only {clean}/500 shots in the Bell subspace"
        line("cudaq GPU sample", f"OK ({clean}/500 shots on |00>+|11>)")
    except Exception as exc:  # noqa: BLE001
        line("cudaq GPU sample", "FAILED")
        fail(str(exc))
else:
    # Distinguish "no GPU requested" (fine, CPU-only) from "GPU is there but
    # CUDA cannot be used" (fatal). The second case used to report OK, which is
    # how a driver too old for CUDA 13 slipped through as a healthy environment.
    dv = driver_cuda_version()
    if dv is None:
        line("GPU", "no driver visible — CPU only (add `--gpus all` for GPU)")
    else:
        line("GPU", "PRESENT BUT UNUSABLE")
        line("driver supports CUDA", f"{dv // 1000}.{(dv % 1000) // 10}")
        need_major = int(want_cuda.split(".")[0])
        if dv // 1000 < need_major:
            fail(
                f"driver supports CUDA {dv // 1000}.{(dv % 1000) // 10} but this "
                f"image needs CUDA {need_major}.x. Use the cu12 build of this "
                "image, or a host with a newer driver."
            )
        else:
            fail("CUDA driver present but torch cannot use it")

print()
print("environment OK" if ok else "environment has problems (see !! lines above)")
sys.exit(0 if ok else 1)
PYEOF

RUN set -euo pipefail \
 && chmod +x /usr/local/bin/qce26-verify \
 && nvcc --version | grep -q "release ${CUDA_DIR_VERSION}" \
 && python3 -c "import torch, os; assert torch.version.cuda == os.environ['QCE26_CUDA_VERSION'], torch.version.cuda" \
 && python3 -c "import torch, torchvision, sklearn, tqdm, qkan, qkan._C" \
 && python3 -c "import cudaq, cuquantum, cuquantum.tensornet" \
 && python3 -c "import cudaq, triton" \
 && python3 -c "import cuquantum; assert cuquantum.__version__.startswith('26.3'), cuquantum.__version__" \
 && python3 -c "import cudaq_einsum, qsvm_cudaq_cpp_backend, qsvm_cutensor_backend" \
 && python3 -c "import cudaq; cudaq.set_target('einsum'); assert cudaq.get_target().name == 'einsum'" \
 && echo "[qce26] build-time checks passed"

# ---------------------------------------------------------------------------
# 9. Runtime
#
# Runs as root on purpose. Brev (and plain `docker run -v`) bind-mounts host
# directories that keep their host ownership; a non-root uid inside the
# container then cannot write them, which breaks the MNIST download in 03 and
# every saved figure. Root sidesteps the uid mismatch entirely, which is the
# usual trade for a disposable single-user GPU instance.
# ---------------------------------------------------------------------------
RUN mkdir -p /workspace

USER root
WORKDIR /workspace
EXPOSE 8888

# The base image's ENTRYPOINT is ["bash", "-l"], so CMD supplies *arguments to
# that shell*, not a command of its own — hence the "-c" form. A bare
# CMD ["jupyter", ...] would run `bash -l jupyter lab ...`, i.e. bash trying to
# execute a script named "jupyter".
#
#   docker run --gpus all -p 8888:8888 IMAGE          -> serves Jupyter
#   docker run -it --gpus all IMAGE -c bash           -> interactive shell
#   docker run --gpus all IMAGE -c qce26-verify       -> one-off command
#
# No token is set here: Jupyter generates one and prints it. Pass
# -e JUPYTER_TOKEN=... to pin it (Jupyter Server reads that variable itself),
# which is what a Brev launchable wants so the tunnel URL works directly.
# --allow-root is required because section 9 runs as root; Jupyter otherwise
# refuses to start ("Running as root is not recommended. Use --allow-root").
CMD ["-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.root_dir=/workspace"]
