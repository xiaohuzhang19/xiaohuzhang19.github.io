---
title: "Triangular Rule for Yield-Curve Shock Propagation (Python)"
date: 2026-01-17
tags: [fixed-income, duration, risk, yield-curve, python]
---

# Triangular Rule for Yield-Curve Shock Propagation (Python)

When computing **effective duration** (or **key rate duration**), we often start with a small shock applied to a set of **key-rate nodes** (e.g., 9 market quotes). We then need a consistent way to propagate those discrete shocks to **every maturity** on a fine grid so the **entire curve** can be shifted smoothly.

A common industry approach is the **Triangular Rule**:

- Each maturity is influenced by **at most two adjacent** key-rate nodes.
- Weights change **linearly** between those nodes.
- Influence is **local** (no long-range leakage).

---

## 1) Triangular weights

Let key-rate tenors be:

$$
T_1 < T_2 < \cdots < T_K
$$

and shocks at those tenors be:

$$
\Delta y_1, \Delta y_2, \ldots, \Delta y_K
$$

For a maturity $t \in [T_i, T_{i+1}]$, triangular weights are:

$$
w_i(t) = \frac{T_{i+1}-t}{T_{i+1}-T_i},\quad
w_{i+1}(t) = \frac{t-T_i}{T_{i+1}-T_i}
$$

and the propagated shock is:

$$
\Delta y(t)=w_i(t)\Delta y_i + w_{i+1}(t)\Delta y_{i+1}.
$$

Outside the key-rate range, the most common convention is **flat extrapolation**:
- for $t < T_1$: use $\Delta y(t)=\Delta y_1$
- for $t > T_K$: use $\Delta y(t)=\Delta y_K$

(You can change this if your risk system uses a different boundary rule.)

---

## 2) Python implementation
## reference material https://github.com/lballabio/QuantLib/issues/1882
Below is a clean, production-friendly implementation:

- `triangular_shock()` propagates shocks from key rates to any target tenor grid.
- `make_parallel_shock()` helps create a parallel shock at all nodes.
- `make_keyrate_bump()` creates a single-node bump (useful for KRD).

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union
import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class TriangularRule:
    """
    Triangular shock propagation for yield-curve tenors.

    Boundary rule (default):
      - Left of first key tenor: flat extrapolation using first shock
      - Right of last key tenor: flat extrapolation using last shock
    """

    key_tenors: np.ndarray  # shape (K,)
    key_shocks: np.ndarray  # shape (K,)

    def __post_init__(self):
        kt = np.asarray(self.key_tenors, dtype=float)
        ks = np.asarray(self.key_shocks, dtype=float)

        if kt.ndim != 1 or ks.ndim != 1:
            raise ValueError("key_tenors and key_shocks must be 1D arrays.")
        if kt.size != ks.size:
            raise ValueError("key_tenors and key_shocks must have the same length.")
        if kt.size < 2:
            raise ValueError("Need at least two key tenors for triangular interpolation.")
        if not np.all(np.isfinite(kt)) or not np.all(np.isfinite(ks)):
            raise ValueError("key_tenors and key_shocks must be finite numbers.")
        if np.any(np.diff(kt) <= 0):
            raise ValueError("key_tenors must be strictly increasing (no duplicates).")

        object.__setattr__(self, "key_tenors", kt)
        object.__setattr__(self, "key_shocks", ks)

    def propagate(self, target_tenors: ArrayLike) -> np.ndarray:
        """
        Propagate key shocks to the target tenors using triangular weights.

        Parameters
        ----------
        target_tenors : array-like
            Tenor grid (in the same unit as key_tenors, e.g. years).

        Returns
        -------
        np.ndarray
            Propagated shocks at each target tenor.
        """
        t = np.asarray(target_tenors, dtype=float)
        if t.ndim != 1:
            raise ValueError("target_tenors must be a 1D array.")
        if not np.all(np.isfinite(t)):
            raise ValueError("target_tenors must be finite numbers.")

        kt = self.key_tenors
        ks = self.key_shocks

        out = np.empty_like(t)

        # Left / right extrapolation (flat)
        left_mask = t <= kt[0]
        right_mask = t >= kt[-1]
        mid_mask = ~(left_mask | right_mask)

        out[left_mask] = ks[0]
        out[right_mask] = ks[-1]

        # For mid tenors: find the right interval index i such that kt[i] <= t < kt[i+1]
        # np.searchsorted returns insertion index; subtract 1 gives left index.
        idx = np.searchsorted(kt, t[mid_mask], side="right") - 1
        # idx is in [0, K-2] for mid points
        t_mid = t[mid_mask]
        T_i = kt[idx]
        T_ip1 = kt[idx + 1]
        dy_i = ks[idx]
        dy_ip1 = ks[idx + 1]

        # Triangular weights
        denom = (T_ip1 - T_i)
        w_i = (T_ip1 - t_mid) / denom
        w_ip1 = (t_mid - T_i) / denom

        out[mid_mask] = w_i * dy_i + w_ip1 * dy_ip1
        return out


def make_parallel_shock(key_tenors: ArrayLike, bump_bps: float) -> TriangularRule:
    """
    Create a parallel bump (same shock at all key tenors).

    bump_bps: e.g. +10 means +10bp = +0.001 in rate terms.
    """
    kt = np.asarray(key_tenors, dtype=float)
    bump = bump_bps / 1e4
    ks = np.full_like(kt, fill_value=bump)
    return TriangularRule(key_tenors=kt, key_shocks=ks)


def make_keyrate_bump(key_tenors: ArrayLike, node_index: int, bump_bps: float) -> TriangularRule:
    """
    Create a single key-rate bump for KRD:
      - only one node is bumped, others are 0.
    """
    kt = np.asarray(key_tenors, dtype=float)
    if node_index < 0 or node_index >= kt.size:
        raise IndexError("node_index out of range.")
    ks = np.zeros_like(kt)
    ks[node_index] = bump_bps / 1e4
    return TriangularRule(key_tenors=kt, key_shocks=ks)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example 9 key rates (years)
    key_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0]

    # Target grid: monthly up to 30Y (in years)
    target_tenors = np.arange(1, 30 * 12 + 1) / 12.0  # 1M..360M

    # (A) Parallel +10bp shock propagated to the full grid
    rule_parallel = make_parallel_shock(key_tenors, bump_bps=10.0)
    shock_full = rule_parallel.propagate(target_tenors)

    print("Parallel +10bp shock at first 10 target points:")
    print(np.round(shock_full[:10] * 1e4, 4), "bps")  # back to bps for display

    # (B) Key-rate bump at the 5Y node (index 5) for KRD
    rule_krd_5y = make_keyrate_bump(key_tenors, node_index=5, bump_bps=10.0)
    shock_5y = rule_krd_5y.propagate(target_tenors)

    # Show where the bump has influence (non-zero region)
    nz = np.where(np.abs(shock_5y) > 1e-12)[0]
    print("\n5Y-node bump influences target tenors roughly from:")
    print(f"{target_tenors[nz[0]]:.3f}Y to {target_tenors[nz[-1]]:.3f}Y")
