---
layout: post
title: "GPU-Accelerated American Option Pricing: PSO, LSMC, and a CUDA Port"
date: 2026-04-18 09:00:00
tags: [research, gpu, quantitative-finance]
category: blog
author: Xiaohu Zhang
description: How Particle Swarm Optimization and the Longstaff-Schwartz Monte Carlo method are accelerated on GPU — from OpenCL (Mac) to CUDA (NVIDIA), validated against S&P 500 options during the 2008 financial crisis.
hidden: false
---

# GPU-Accelerated American Option Pricing: PSO, LSMC, and a CUDA Port

---

## Motivation

American options are harder to price than European options because they carry an **early exercise right** — the holder can exercise at any point before expiry, not just at maturity. This transforms pricing from a closed-form problem into a **free-boundary PDE**, which must be solved numerically.

Two modern numerical methods have emerged as strong candidates for GPU acceleration:

1. **Particle Swarm Optimization (PSO)** — searches for the early exercise boundary by treating each "particle" (candidate boundary) as an agent in a swarm
2. **Longstaff-Schwartz Monte Carlo (LSMC)** — simulates stock price paths, then uses regression to estimate continuation values at each time step

Both are embarrassingly parallel at their core, making them ideal for GPU computing. This post summarises the GPU implementations I built and benchmarked — first on an Apple Silicon Mac using **OpenCL**, then ported to **NVIDIA CUDA** for Google Colab.

The original framework is due to Leon Xing Li and Ren-Raw Chen (JFDS 2023; JOD 2024). This work extends it with new GPU kernel variants and cross-platform benchmarking.

---

## The Pricing Methods

### Monte Carlo Simulation

The stock price follows geometric Brownian motion:

$$
S_{t+\Delta t} = S_t \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z_t\right], \quad Z_t \sim \mathcal{N}(0,1)
$$

We simulate $N$ paths of $M$ time steps. On GPU, each thread handles one path — $N = 65{,}536$ paths run simultaneously.

### Particle Swarm Optimization (PSO)

PSO finds the early exercise boundary $\{b_t\}_{t=0}^{T}$ — the stock price below which immediate exercise dominates holding. Each particle $i$ maintains a position vector $\mathbf{x}_i \in \mathbb{R}^M$ representing a candidate boundary.

The velocity update rule is:

$$
\mathbf{v}_i \leftarrow \omega\,\mathbf{v}_i + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i)
$$

where $\mathbf{p}_i$ is the particle's personal best, $\mathbf{g}$ is the global best, and $r_1, r_2 \sim U(0,1)$.

The **fitness function** evaluates each boundary by computing the American option price across all Monte Carlo paths using a backward scan:

$$
\hat{C} = \frac{1}{N} \sum_{j=1}^{N} e^{-r \tau^*_j} \max(K - S_{\tau^*_j},\, 0)
$$

where $\tau^*_j$ is the first time the stock price crosses the candidate boundary on path $j$.

#### Three GPU Variants

| Variant | What runs on GPU | Key idea |
|---|---|---|
| **Hybrid** | Fitness evaluation only | Threads split path-wise |
| **Scalar** | searchGrid + fitness | One thread per particle, backward scan |
| **Fusion** | searchGrid + fitness + pbest update | Single kernel, minimises memory round-trips |

The **fusion kernel** is the most efficient: it eliminates two extra GPU memory writes per iteration by fusing all three operations into one `__global__` function.

### Longstaff-Schwartz LSMC

LSMC avoids explicit boundary computation. At each time step $t$ (traversed backward), it:

1. Identifies in-the-money paths
2. Regresses their discounted future payoffs on basis functions of $S_t$ to estimate continuation value $\hat{C}(S_t)$
3. Exercises wherever immediate payoff $> \hat{C}(S_t)$

The regression at each step requires solving a $3 \times 3$ linear system $\mathbf{A}\boldsymbol{\beta} = \mathbf{b}$. Two GPU matrix inversion strategies were implemented:

- **Gauss-Jordan (GJ):** standard pivoting method
- **Optimized branchless:** analytical $3 \times 3$ inverse using `fmaf()` fused multiply-add, avoiding branching for better GPU warp efficiency

---

## OpenCL → CUDA Port

The original code used **PyOpenCL**, which runs on Apple Metal, AMD, and NVIDIA GPUs but requires a local runtime install. To enable one-click execution on **Google Colab**, I ported the entire GPU backend to **PyCUDA**.

Key translation decisions:

| OpenCL | CUDA |
|---|---|
| `__kernel void f(...)` | `__global__ void f(...)` |
| `get_global_id(0)` | `blockIdx.x * blockDim.x + threadIdx.x` |
| `__local float arr[N]` (parameter) | `extern __shared__ float arr[]` (inside kernel body) |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| `select(a, b, cond)` | `cond ? b : a` |
| `mad(a,b,c)` | `fmaf(a,b,c)` |
| `cl.Buffer` + `cl.enqueue_copy` | `cuda.mem_alloc()` + `cuda.memcpy_htod()` |
| `queue.finish()` | `cuda.Context.synchronize()` |

One subtle issue: in OpenCL, `get_global_size(0)` returns exactly `nFish`. In CUDA, the grid is padded to the next multiple of the block size, so the value would be larger. The fix was to pass `nFish` as an explicit `const int nParticle` parameter to all PSO kernels and guard with `if (gid >= nParticle) return`.

---

## Benchmark Results

**Parameters:** S0=100, K=100, r=3%, σ=30%, T=1yr | nPath=65,536 · nPeriod=50 · nFish=500

| Method | Price | OpenCL Mac (ms) | Mac Speedup | CUDA NVIDIA (ms) | NVIDIA Speedup |
|---|---:|---:|---:|---:|---:|
| MC — CPU (NumPy)   | 10.3587 | 0.58     | —     | 0.46     | —     |
| MC — GPU           | 10.3587 | 78.36    | 0.0x  | 1194.72  | 0.0x  |
| PSO — CPU (NumPy)  | 10.5814 | 14,653   | —     | 29,010   | —     |
| PSO — GPU hybrid   | 10.5814 | 5,758    | 2.5x  | 6,512    | 4.5x  |
| PSO — GPU scalar   | 10.5814 | 777      | 18.9x | 1,048    | 27.7x |
| PSO — GPU fusion   | 10.5814 | 760      | **19.3x** | 1,042 | **27.8x** |
| LSMC — CPU (NumPy) | 10.6055 | 173      | —     | 218      | —     |
| LSMC — GPU opt     | 10.6054 | 166      | 1.0x  | 268      | 0.8x  |

**Key observations:**

- **PSO fusion is the winner** at these parameters: 19x on Mac, 28x on NVIDIA. Fusing three operations into one kernel eliminates memory round-trips that dominate the hybrid variant.
- **LSMC GPU does not beat CPU** at nPeriod=50. With only 50 regression steps, the kernel launch overhead is not amortized. At nPeriod=200 (empirical study below) it achieves 1.2–1.7x speedup.
- **MC GPU first-call overhead**: the CUDA time of ~1,195ms includes PyCUDA's JIT compilation via `nvcc`. The actual kernel runs in under 5ms on subsequent calls.
- Both backends produce **identical prices** to 4 decimal places for PSO. LSMC shows a small float32 rounding gap between Apple Metal and NVIDIA cores (≤0.002), which is normal for float32 arithmetic.

---

## Empirical Study: S&P 500, September 29, 2008

**TARP Rejection Day** — the House voted down the TARP bailout. The S&P 500 fell 8.8% in a single session, closing at 1106.42. Near-expiry American Put options were priced at three moneyness levels using that day's implied volatilities.

**Parameters:** S0=1106.42, r=0.94%, T=30 days | nPath=10,000 · nPeriod=200 · nFish=256

| Case | K | σ | Market Price | PSO GPU | PSO vs Market | LSMC GPU | LSMC vs Market |
|---|---:|---:|---:|---:|---:|---:|---:|
| ITM  (Δ≈−0.75) | 1174.136 | 28.24% | 78.9833 | 80.0182 | +1.31% | 79.4481 | +0.59% |
| ATM  (Δ≈−0.50) | 1106.000 | 27.50% | 34.0100 | 34.4599 | +1.32% | 34.3647 | +1.04% |
| vOTM (Δ≈−0.15) |  987.000 | 41.00% | 10.7400 | 10.8331 | +0.87% | 11.1171 | +3.52% |

**GPU speedup (CPU → PSO GPU fusion):**

| Case | OpenCL Mac | CUDA NVIDIA |
|---|---:|---:|
| ITM  | 7.7x | 8.1x |
| ATM  | 7.8x | 7.7x |
| vOTM | 7.6x | 9.0x |

All methods price within ~1.3% of market for ITM and ATM cases. The vOTM LSMC error (3.5%) is larger — short-dated deep OTM puts have very few in-the-money paths at each regression step, which degrades the regression quality. PSO handles this better (0.87%) because it searches globally for the boundary rather than relying on local regression.

The higher σ=41% for the vOTM case reflects the **volatility smile** — market participants priced in elevated tail risk on that day, which Black-Scholes would underestimate with a flat volatility assumption.

---

## Code and Reproducibility

**OpenCL (Mac/Linux):**
[xiaohuzhang19/Fin_ParallelComputing](https://github.com/xiaohuzhang19/Fin_ParallelComputing)
```bash
pip install numpy scipy pyopencl
cd src && python3 American_option.py
```

**CUDA (Google Colab, NVIDIA GPU):**
[xiaohuzhang19/CUDAPSOAMERICAOPTION](https://github.com/xiaohuzhang19/CUDAPSOAMERICAOPTION)
Open `src/American_option_colab.ipynb` in Colab → set runtime to T4 GPU → Run all.
Section 7 runs the full Sep 2008 empirical study automatically.

Full price and timing comparison across both backends: [`empirical_study_combined_opencl_cuda.md`](https://github.com/xiaohuzhang19/CUDAPSOAMERICAOPTION/blob/main/empirical_study_combined_opencl_cuda.md)

---

## Citations

**Li, L. X., & Chen, R. R. (2023).** Using the Graphics Processing Unit to Evaluate American-Style Derivatives. *Journal of Financial Data Science.*

**Li, L. X., Chen, R. R., & Fabozzi, F. J. (2024).** GPU-Accelerated American Option Pricing: The Case of the Longstaff-Schwartz Monte Carlo Model. *Journal of Derivatives.*
