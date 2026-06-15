"""Density and vertical-stability analysis helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

import numpy as np

from nespreso.physics_metrics import (
    density_smoothness_metrics,
    eos_from_SP_T,
    static_stability_metrics,
)


def compute_density_profiles(T_profiles, S_profiles, lat_arr, lon_arr, depth_arr, model_name, pres_data=None):
    """Compute density profiles from T/S using EOS."""
    n_prof, n_depth = T_profiles.shape
    rho_profiles = np.full((n_prof, n_depth), np.nan)

    # Use PRES data if available, otherwise use depth as pressure
    if pres_data is not None:
        p = pres_data.T  # (n_profiles, depth)
    else:
        # Pressure from depth (approximate: 1 dbar ≈ 1 m)
        p = np.broadcast_to(depth_arr, (n_prof, n_depth))

    # Process profile by profile to handle lat/lon
    for i in range(n_prof):
        T_prof = T_profiles[i, :]
        S_prof = S_profiles[i, :]
        lat_prof = lat_arr[i]
        lon_prof = lon_arr[i]
        p_prof = p[i, :]

        # Only process if we have sufficient valid data
        valid_mask = np.isfinite(T_prof) & np.isfinite(S_prof) & np.isfinite(p_prof)
        if np.sum(valid_mask) >= 2:
            try:
                _, _, rho_prof = eos_from_SP_T(S_prof, T_prof, p_prof, lon=lon_prof, lat=lat_prof)
                rho_profiles[i, :] = rho_prof
            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"  Warning: EOS failed for {model_name} profile {i}: {e}")

    return rho_profiles


def compute_stability_metrics(rho_profiles, depth_arr, model_name):
    """Compute static stability metrics for a set of profiles."""
    # Reshape to (n_profiles, depth) - already in correct format
    # metrics functions expect depth as last axis
    stability = static_stability_metrics(rho_profiles, depth_arr, axis=-1, g=9.81)
    return stability


def compute_smoothness_metrics(rho_profiles, depth_arr, model_name, zmin=50.0, zmax=300.0):
    """Compute density smoothness metrics for a set of profiles."""
    smoothness = density_smoothness_metrics(rho_profiles, depth_arr, zmin=zmin, zmax=zmax, axis=-1)
    return smoothness
