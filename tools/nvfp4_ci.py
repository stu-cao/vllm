# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build-time checks for the fork's NVFP4 wheel trial; no GPU execution here."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

KERNEL_PREFIX = "flashinfer/experimental/nvfp4_sparse_mla_decode/csrc/"
KERNEL_FILES = (
    "nvfp4_sparse_mla_decode.cuh",
    "nvfp4_sparse_mla_decode.cu",
    "nvfp4_sparse_mla_decode_jit_binding.cu",
)
VLLM_BACKEND = "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def write_manifest(directory: Path, **values) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "python": sys.version,
        "host_arch": platform.machine(),
        "platform": platform.platform(),
        "gpu_execution_tested": False,
        **values,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def single_wheel(directory: Path) -> Path:
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one wheel in {directory}: {wheels}")
    return wheels[0]


def inspect_flashinfer(directory: Path, source: Path) -> None:
    wheel = single_wheel(directory)
    with zipfile.ZipFile(wheel) as archive:
        for filename in KERNEL_FILES:
            member = KERNEL_PREFIX + filename
            if archive.read(member) != (source / member).read_bytes():
                raise RuntimeError(f"Packaged kernel differs from source: {member}")
        for header in (
            "flashinfer/data/csrc/tvm_ffi_utils.h",
            "flashinfer/data/include/flashinfer/utils.cuh",
        ):
            archive.getinfo(header)
    write_manifest(
        directory,
        flashinfer_sha=command("git", "-C", str(source), "rev-parse", "HEAD"),
        wheel=wheel.name,
        wheel_sha256=sha256(wheel),
        kernel_source_packaging_verified=True,
    )


def compile_cache(arch: str, directory: Path, source: Path) -> None:
    import flashinfer
    import torch
    from flashinfer.experimental.nvfp4_sparse_mla_decode.jit import (
        gen_nvfp4_sparse_mla_decode_module,
    )

    if Path(flashinfer.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Compile must use the installed wheel, not the checkout")
    target = {"10.0a": "sm_100a", "10.3a": "sm_103a"}[arch]
    spec = gen_nvfp4_sparse_mla_decode_module(target)
    spec.build(verbose=True)
    destination = directory / target
    destination.mkdir(parents=True, exist_ok=True)
    library = destination / spec.jit_library_path.name
    shutil.copy2(spec.jit_library_path, library)
    shutil.copy2(spec.ninja_path, destination / "build.ninja.reference")
    write_manifest(
        destination,
        artifact_kind="experimental-jit-compile-output-not-aot-provider",
        flashinfer_sha=command("git", "-C", str(source), "rev-parse", "HEAD"),
        flashinfer_version=flashinfer.__version__,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        tvm_ffi=importlib.metadata.version("apache-tvm-ffi"),
        target=target,
        module=spec.name,
        library=library.name,
        library_sha256=sha256(library),
        compiler=command("nvcc", "--version"),
        cuda_images=command("cuobjdump", "--list-elf", str(library)),
        note="Not a portable installed cache: validate ABI and GPU loading first. "
        "Normal JIT may rebuild after relocation; do not set FLASHINFER_DISABLE_JIT.",
    )


def fetch_vllm(commit: str, destination: Path) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("An exact upstream commit is required")
    index = f"https://wheels.vllm.ai/{commit}/cu130/vllm/"
    with urllib.request.urlopen(index + "metadata.json", timeout=60) as response:
        entries = json.load(response)
    matches = [
        entry
        for entry in entries
        if entry["package_name"] == "vllm"
        and entry["platform_tag"].endswith("_" + platform.machine())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching upstream wheel: {matches}")
    url = urllib.parse.urljoin(index, matches[0]["path"])
    if not url.startswith(f"https://wheels.vllm.ai/{commit}/"):
        raise RuntimeError(f"Unexpected wheel location: {url}")
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        destination.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream)
    metadata = {"commit": commit, "url": url, "sha256": sha256(destination)}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def verify_native_binaries(built: zipfile.ZipFile, original: zipfile.ZipFile) -> int:
    for required in (
        "vllm/_C_stable_libtorch.abi3.so",
        "vllm/_moe_C_stable_libtorch.abi3.so",
        "vllm/vllm-rs",
    ):
        built.getinfo(required)
    native_sets = [
        {
            name
            for name in archive.namelist()
            if name.endswith(".so") or name == "vllm/vllm-rs"
        }
        for archive in (built, original)
    ]
    if native_sets[0] != native_sets[1]:
        raise RuntimeError(
            f"Native binary inventory differs: {native_sets[0] ^ native_sets[1]}"
        )
    for name in sorted(native_sets[0]):
        if built.read(name) != original.read(name):
            raise RuntimeError(f"Native binary differs from pinned upstream: {name}")
    return len(native_sets[0])


def inspect_vllm(directory: Path, upstream: Path) -> None:
    wheel = single_wheel(directory)
    with zipfile.ZipFile(wheel) as built, zipfile.ZipFile(upstream) as original:
        if built.read(VLLM_BACKEND) != Path(VLLM_BACKEND).read_bytes():
            raise RuntimeError("Fork's native NVFP4 backend was not packaged")
        native_count = verify_native_binaries(built, original)
    write_manifest(
        directory,
        vllm_sha=command("git", "rev-parse", "HEAD"),
        wheel=wheel.name,
        wheel_sha256=sha256(wheel),
        reused_native_binary_count=native_count,
        upstream=json.loads(upstream.with_suffix(".json").read_text()),
        note="Python fork wheel with verified reused upstream native binaries; "
        "not a full source rebuild.",
    )


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "inspect-flashinfer":
        inspect_flashinfer(Path(args[0]), Path(args[1]))
    elif action == "compile-cache":
        compile_cache(args[0], Path(args[1]), Path(args[2]))
    elif action == "fetch-vllm":
        fetch_vllm(args[0], Path(args[1]))
    elif action == "inspect-vllm":
        inspect_vllm(Path(args[0]), Path(args[1]))
    else:
        raise ValueError(f"Unknown action: {action}")
