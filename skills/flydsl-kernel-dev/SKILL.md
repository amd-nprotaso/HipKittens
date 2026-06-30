---
name: flydsl-kernel-dev
description: >
  Personal entry point for developing GPU kernels with FlyDSL (Flexible Layout Python DSL,
  an MLIR/ROCm stack for AMD GPUs at /var/home/FlyDSL). Routes a kernel request to the right
  in-repo project skill (write / reference / debug / optimize / profile / build) instead of
  duplicating their content. Use whenever the user wants to write, port, review, debug,
  optimize, profile, or build a FlyDSL kernel, or mentions FlyDSL, Fly dialect, MFMA/WMMA
  kernels, layout algebra, tiled copy/MMA, or preshuffle GEMM on gfx942/gfx950/gfx1250.
allowed-tools: Read Edit Bash Grep Glob Agent Skill
---

# FlyDSL Kernel Development (router)

FlyDSL lives at `/var/home/FlyDSL`. It already ships a mature, git-tracked set of
**project-local skills** under `/var/home/FlyDSL/.claude/skills/` plus authoritative docs
under `/var/home/FlyDSL/docs/`. This skill does not re-document FlyDSL — it routes the task
to the correct project skill and names the canonical source files to read.

## First steps for any FlyDSL kernel task

1. Read `/var/home/FlyDSL/CLAUDE.md` — repository layout, build/test commands, env vars,
   GPU arch table (gfx942/gfx950/gfx11*/gfx120*/gfx1250), and kernel authoring conventions.
2. Identify the task category below and **invoke the matching project skill** via the Skill
   tool (project skills are auto-loaded when working in this repo). Do not reimplement their
   guidance here.
3. Skim a concrete example before writing: `examples/01-vectorAdd.py` (elementwise),
   `examples/02-tiledCopy.py`, `examples/03-tiledMma.py`, `examples/04-preshuffle_gemm.py`,
   and `examples/notebooks/`.

## Routing table

| If the user wants to… | Invoke project skill | Also read |
|---|---|---|
| Write a new kernel / port a Triton kernel (guided procedure) | `flydsl-tile-programming` | `examples/*.py`, `docs/kernel_authoring_guide.md` |
| Look up layout-algebra API, copy/MMA atoms, Vector/Numeric types, autotune, env vars | `flydsl-kernel-authoring` | `docs/layout_system_guide.md`, `docs/cute_layout_algebra_guide.md` |
| Fix NaN / inf / wrong results / compile errors | `debug-flydsl-kernel` | `docs/architecture_guide.md` |
| Optimize a GEMM (LDS ping-pong, swizzle, prefetch, MFMA scheduling, TFLOPS) | `gemm-optimization` | `kernels/preshuffle_gemm.py`, `kernels/mfma_preshuffle_pipeline.py` |
| Fix LDS bank conflicts / lgkmcnt stalls | `lds-optimization` | — |
| Add A/B prefetch / software-pipeline a load | `prefetch-data-load` | `kernels/pipeline_utils.py` |
| Capture an ATT trace | `capture-kernel-trace` | — |
| Analyze a trace / find stall hotspots | `kernel-trace-analysis` | — |
| Bisect a perf regression | `bisect-perf-regression` | — |
| Detect out-of-bounds memory access | `oob-detection` | — |
| Add a new MFMA/WMMA/Copy op to a backend dialect (C++/TableGen) | `add-target-atom-op` | `include/flydsl/`, `lib/Dialect/FlyROCDL/` |
| Build / install the stack | `build-flydsl` | `scripts/build*.sh` |
| Format / pass the CI style gate | `format-code` | `scripts/check_python_style.sh` |

The reference (`flydsl-kernel-authoring`) and the wizard (`flydsl-tile-programming`) are a
pair: use the wizard to produce a kernel step-by-step, the reference to look things up. Both
cross-link to `debug-flydsl-kernel` for failures.

## Conventions that bite (from CLAUDE.md — confirm against current code)

- Prefer the layout API: `fx.rocdl.make_buffer_tensor()` + logical layout ops + `fx.copy_atom_call`.
  Raw `buffer_ops.create_buffer_resource()` / manual byte offsets are legacy.
- `@flyc.kernel` for device kernels, `@flyc.jit` for launch wrappers; import kernels from `kernels.*`.
- `range_constexpr(...)` = compile-time unrolled loop. `range(start, stop, step, init=[...])`
  with **DSL `fx.Index` bounds** = `scf.for` with loop-carried state (Python-int bounds get unrolled
  and silently ignore `init=`).
- Allocate LDS with `SharedAllocator` (`fx.SharedAllocator`) for new kernels; legacy
  `SmemAllocator`/`SmemPtr` remains in un-migrated kernels. Clear `SmemPtr._view_cache = None`
  after a `scf.for` when re-creating shared-memory views (avoids MLIR dominance errors).
- `buffer_load`/`buffer_store` offsets are in **elements**, not bytes.
- Wave size: 64 on CDNA (gfx9xx), 32 on RDNA (gfx11*/gfx120*). gfx1250 hardcodes wave32 itself
  (`is_rdna_arch('gfx1250')` is False). Use `get_warp_size(arch)` from `kernels/kernels_common.py`.
- `python/flydsl/expr/` direct children are target-neutral — no ROCDL/HIP imports there; new
  target-specific code goes in `expr/rocdl/`.

## Verify before relying on these notes

This file is a router; the project skills, docs, and code are the source of truth and evolve.
Before recommending a specific function/flag/path, grep the current tree (`/var/home/FlyDSL`)
to confirm it still exists. If a project skill and this router disagree, trust the project skill.

## Build & test quick reference

```bash
cd /var/home/FlyDSL
bash scripts/build.sh -j64                  # C++ + Python bindings (LLVM via build_llvm.sh first)
pip install -e .
bash scripts/run_tests.sh                   # pytest + examples + MLIR FileCheck
python3 -m pytest tests/kernels/test_pa.py -v
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 <kernel>.py   # dump IR, bypass cache
bash scripts/check_python_style.sh --fix    # black + ruff (CI style gate)
```
