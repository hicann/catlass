---
nav_order: 30
---

# DSL Framework Development Guide

This directory is for developers who need to configure a build environment from source, build CATLASS DSL themselves, and work on it in depth.

> This version does not currently provide a complete product package (install-and-run). The installation and build steps below are the from-source developer workflow.

## Environment Setup

| Document | Scope |
|------|------|
| [Environment Preparation](build_guide/environment.md) | Overall environment requirements, dependency software, and environment checks. |
| [Conda Installation](build_guide/conda.md) | Create a development environment with the repository `environment.yml` (does not include CANN or AscendNPU-IR). |
| [Docker Installation](build_guide/docker.md) | Build a development image with the repository `Dockerfile`, including all development dependencies. |

## Build and Test

| Document | Scope |
|------|------|
| [Building CATLASS DSL](build_guide/build.md) | `./build.sh` build (Development / Release mode) and artifact checks. |
| [Running Test Cases](build_guide/testing.md) | pytest, lit test, and end-to-end verification. |
| [Building AscendNPU-IR](build_guide/ascend_npu_ir.md) | Manually build the AscendNPU-IR that the DSL depends on. |
| [Manual CMake Build](advanced/manual_cmake_build.md) | Advanced usage: configuring `csrc/mlir` directly. |

## In-Depth Development

| Document | Scope |
|------|------|
| [Feature Development](feature_development/index.md) | Extending DSL backend features, including [bc backend integration](feature_development/bc_backend_integration.md) (using `tla.copy` as an example). |
