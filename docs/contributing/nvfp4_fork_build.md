# Prime fork wheels

The fork-only `nvfp4-wheels.yml` workflow builds one vLLM wheel and one
FlashInfer wheel per architecture (x86_64 for B300/H200, aarch64 for
GB200/GB300) on GitHub-hosted Ubuntu 24.04 CPUs. Every production serving lane
installs the same vLLM build. It does not deploy a service or publish a
release/container. Pushes to `prime/v*` branches (e.g. `prime/v0.31.0`: the
v0.31.0 release plus our production patches) run it; rerun an existing run from
Actions or `gh run rerun`. Manual dispatch is declared, but GitHub requires
the workflow to exist on the default branch before it appears in that UI.

## Outputs

- `flashinfer-nvfp4-cu130-linux-<arch>`: FlashInfer source wheel containing the
  validated B300 kernel/planner changes and native JIT sources; separately
  compiled SM100a and SM103a libraries, build commands, hashes, and manifests.
- `vllm-prime-cu130-linux-<arch>`: vLLM's native-NVFP4 integration wheel, using
  CUDA/Rust binaries from the pinned upstream release (`VLLM_BINARY_COMMIT`).
  The build refuses reuse if native sources or build configuration changed and
  verifies packaged native binaries against that upstream wheel.

The toolchain uses Python 3.12, PyTorch 2.13.0/cu130, and CUDA 13.0. Kernel
compilation pins TVM-FFI 0.1.11 and CUTLASS DSL 4.7.1 to match vLLM's runtime
requirements. The
FlashInfer revision and vLLM binary-parent revision are pinned in the workflow.
Each artifact records resolved versions and source/binary hashes. Dependency
lists are recorded, not yet a fully locked or hermetic environment.

For another kernel revision, update `FLASHINFER_REF` or the manual-dispatch
input. Native vLLM changes require a full source build or a new verified matching
binary parent; do not bypass the source-diff guard or use an unrelated wheel.

## Install and validate

Download the two artifacts into the same directory:

```bash
gh run download RUN_ID --repo stu-cao/vllm --dir nvfp4-build
uv venv --python 3.12 .venv-nvfp4
uv pip install --python .venv-nvfp4/bin/python 'torch==2.13.0' \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-nvfp4/bin/python \
  nvfp4-build/flashinfer-nvfp4-cu130-linux-x86_64/flashinfer/*.whl \
  nvfp4-build/vllm-prime-cu130-linux-x86_64/*.whl
```

Use a clean environment. The fork FlashInfer wheel has a local version suffix;
do not retain an official `flashinfer-jit-cache` package with a mismatched
version or disable its version checks. Other FlashInfer kernels will use JIT.
The source wheel supports ordinary first-use JIT compilation with CUDA 13.0.

The separate kernel output is a **compile-test artifact, not an automatically
installed, relocatable JIT cache or an AOT provider**. Its reference Ninja file
contains CI paths. Do not copy it into a runtime cache and assume that first-use
compilation is eliminated. Validate host architecture, compiler/runtime ABI,
TVM-FFI, kernel source digest, and GPU target before loading its `.so`; normal
JIT may rebuild after relocation. A validated portable-cache installer is a
follow-up, not a claim of this first trial. Experimental kernels remain out of
FlashInfer's official AOT packages.

GitHub CPU builds cannot run the GPU correctness suite, CUDA graph replay,
sanitizer, or latency measurements. Validate the downloaded wheel on B300 using
the matching FlashInfer checkout's `tests/experimental/test_nvfp4_sparse_mla_decode.py`
before using it in vLLM. SM100a compilation is not GB200 runtime validation,
and each architecture's artifacts only install on hosts of that architecture.
