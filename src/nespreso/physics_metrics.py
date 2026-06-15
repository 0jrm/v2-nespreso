import numpy as np
import math
import importlib
import warnings

# -----------------------------#
# Utilities
# -----------------------------#


def _ensure_array(x):
    return np.asarray(x, dtype=float)


def centered_diff(a, dx, axis=-1):
    """
    Centered differences with one-sided at boundaries.
    If the axis has length 1, returns zeros.
    NaN-safe: preserves NaN values and handles invalid differences.
    """
    a = _ensure_array(a)
    n = a.shape[axis]
    out = np.full_like(a, np.nan, dtype=float)
    if n < 2:
        return out

    slc = [slice(None)] * a.ndim

    # interior
    slc_center = list(slc)
    slc_center[axis] = slice(1, -1)
    slc_minus = list(slc)
    slc_minus[axis] = slice(0, -2)
    slc_plus = list(slc)
    slc_plus[axis] = slice(2, None)
    a_plus = a[tuple(slc_plus)]
    a_minus = a[tuple(slc_minus)]
    # Only compute where both values are finite
    valid = np.isfinite(a_plus) & np.isfinite(a_minus)
    out[tuple(slc_center)] = np.where(valid, (a_plus - a_minus) / (2.0 * dx), np.nan)

    # left boundary (forward)
    slc0 = list(slc)
    slc0[axis] = 0
    slc1 = list(slc)
    slc1[axis] = 1
    a0 = a[tuple(slc0)]
    a1 = a[tuple(slc1)]
    valid = np.isfinite(a0) & np.isfinite(a1)
    out[tuple(slc0)] = np.where(valid, (a1 - a0) / dx, np.nan)

    # right boundary (backward)
    slc_last = list(slc)
    slc_last[axis] = -1
    slc_lastm1 = list(slc)
    slc_lastm1[axis] = -2
    a_last = a[tuple(slc_last)]
    a_lastm1 = a[tuple(slc_lastm1)]
    valid = np.isfinite(a_last) & np.isfinite(a_lastm1)
    out[tuple(slc_last)] = np.where(valid, (a_last - a_lastm1) / dx, np.nan)

    return out


def second_derivative(a, dx, axis=-1):
    """
    Second derivative via centered differences.
    If axis length < 3, returns zeros.
    NaN-safe: preserves NaN values and handles invalid differences.
    """
    a = _ensure_array(a)
    n = a.shape[axis]
    out = np.full_like(a, np.nan, dtype=float)
    if n < 3:
        return out

    slc = [slice(None)] * a.ndim

    # interior
    slc_center = list(slc)
    slc_center[axis] = slice(1, -1)
    slc_minus = list(slc)
    slc_minus[axis] = slice(0, -2)
    slc_plus = list(slc)
    slc_plus[axis] = slice(2, None)
    a_center = a[tuple(slc_center)]
    a_plus = a[tuple(slc_plus)]
    a_minus = a[tuple(slc_minus)]
    # Only compute where all three values are finite
    valid = np.isfinite(a_center) & np.isfinite(a_plus) & np.isfinite(a_minus)
    out[tuple(slc_center)] = np.where(valid, (a_plus - 2.0 * a_center + a_minus) / (dx**2), np.nan)

    # boundaries: copy nearest interior value (if finite)
    slc0 = list(slc)
    slc0[axis] = 0
    slc1 = list(slc)
    slc1[axis] = 1
    out[tuple(slc0)] = out[tuple(slc1)]

    slc_last = list(slc)
    slc_last[axis] = -1
    slc_lastm1 = list(slc)
    slc_lastm1[axis] = -2
    out[tuple(slc_last)] = out[tuple(slc_lastm1)]

    return out


def spearmanr(a, b):
    """
    Spearman rank correlation for flattened arrays (no SciPy).
    """
    a = _ensure_array(a).ravel()
    b = _ensure_array(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a = a[mask]
    b = b[mask]

    def rankdata(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        # average ties
        vals, idx_start, counts = np.unique(x[order], return_index=True, return_counts=True)
        for s, c in zip(idx_start, counts):
            if c > 1:
                avg = (2 * s + c - 1) / 2.0
                ranks[order[s : s + c]] = avg
        return ranks

    ra = rankdata(a)
    rb = rankdata(b)
    ra_m = ra - ra.mean()
    rb_m = rb - rb.mean()
    num = np.sum(ra_m * rb_m)
    den = math.sqrt(np.sum(ra_m**2) * np.sum(rb_m**2))
    return num / den if den > 0 else np.nan


def angle_between(u1, v1, u2, v2):
    """
    Angle between vectors (u1,v1) and (u2,v2) in degrees [0, 180].
    """
    u1 = _ensure_array(u1)
    v1 = _ensure_array(v1)
    u2 = _ensure_array(u2)
    v2 = _ensure_array(v2)
    dot = u1 * u2 + v1 * v2
    n1 = np.hypot(u1, v1)
    n2 = np.hypot(u2, v2)
    denom = n1 * n2

    cosang = np.full_like(dot, np.nan, dtype=float)
    mask = denom > 0
    cosang[mask] = np.clip(dot[mask] / denom[mask], -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return ang


def smooth_vertical_profile(data, z, window_m=10.0, axis=-1):
    """
    Smooth vertical profile using Hanning window convolution.

    Applies padding: edge mode at surface (top), reflect mode at bottom.

    Parameters
    ----------
    data : array_like
        1D or multi-dimensional array with depth along specified axis.
    z : array_like
        1D depth array [m], positive downward.
    window_m : float, default=10.0
        Window size in meters.
    axis : int, default=-1
        Axis along which to smooth (depth axis).

    Returns
    -------
    smoothed : ndarray
        Smoothed data with same shape as input.
    """
    if window_m <= 0:
        print(f"Window size is less than or equal to 0, returning original data")
        return data

    data = _ensure_array(data)
    z = _ensure_array(z)

    if z.ndim != 1:
        raise ValueError("Depth array z must be 1D")

    # Move depth axis to last position for easier manipulation
    data_moved = np.moveaxis(data, axis, -1)
    original_shape = data_moved.shape
    n_depth = data_moved.shape[-1]

    if n_depth != z.size:
        raise ValueError(f"Depth axis size ({n_depth}) must match z.size ({z.size})")

    # Calculate window size in grid points
    dz_mean = float(np.nanmean(np.diff(z)))
    if dz_mean <= 0 or not np.isfinite(dz_mean):
        # Fallback: use minimum positive difference
        dz_vals = np.diff(z)
        dz_mean = float(np.nanmin(dz_vals[dz_vals > 0]))
        if dz_mean <= 0 or not np.isfinite(dz_mean):
            # No valid spacing, return original data
            return data

    window_points = max(3, int(np.round(window_m / dz_mean)))
    if window_points % 2 == 0:
        window_points += 1  # Ensure odd for symmetric window

    if window_points >= n_depth:
        # Window too large, use all points
        window_points = n_depth if n_depth % 2 == 1 else n_depth - 1

    # Create Hanning window
    window = np.hanning(window_points)
    window = window / np.sum(window)  # Normalize

    # Reshape to 2D: (all_other_dims, depth)
    data_2d = data_moved.reshape(-1, n_depth)
    smoothed_2d = np.full_like(data_2d, np.nan)

    # Pad each profile: edge at top (surface), reflect at bottom
    pad_size = window_points // 2
    for i in range(data_2d.shape[0]):
        prof = data_2d[i, :]

        # Pad: edge mode at beginning (surface), reflect at end (bottom)
        if pad_size > 0:
            # Top padding: replicate first (surface) value
            top_pad = np.full(pad_size, prof[0], dtype=prof.dtype)

            # Bottom padding: reflect last values
            if n_depth > 1:
                # Take last few values and create reflection pattern
                bottom_vals = prof[-min(pad_size + 1, n_depth) :]
                # Create reflection: reverse and append original
                if len(bottom_vals) > 1:
                    bottom_pad = np.concatenate([bottom_vals[-2::-1], bottom_vals])
                    # Take only what we need
                    bottom_pad = bottom_pad[:pad_size]
                else:
                    # Single value, just replicate
                    bottom_pad = np.full(pad_size, prof[-1], dtype=prof.dtype)
            else:
                # Single depth level, replicate
                bottom_pad = np.full(pad_size, prof[-1], dtype=prof.dtype)

            # Concatenate: top_pad + prof + bottom_pad
            prof_padded = np.concatenate([top_pad, prof, bottom_pad])
        else:
            # No padding needed
            prof_padded = prof

        # Convolve
        smoothed_prof = np.convolve(prof_padded, window, mode="valid")

        # Handle NaN: only smooth where original data is finite
        finite_mask = np.isfinite(prof)
        if np.any(finite_mask):
            smoothed_2d[i, :] = np.where(finite_mask, smoothed_prof, np.nan)
        else:
            smoothed_2d[i, :] = prof

    # Reshape back and move axis back
    smoothed_moved = smoothed_2d.reshape(original_shape)
    smoothed = np.moveaxis(smoothed_moved, -1, axis)

    return smoothed


# -----------------------------#
# TEOS-10 / EOS wrapper
# -----------------------------#

_gsw_spec = importlib.util.find_spec("gsw")
_gsw = importlib.import_module("gsw") if _gsw_spec is not None else None


def eos_from_SP_T(SP, T, p, lon=None, lat=None):
    """
    Convert Practical Salinity SP and in-situ T to
    Absolute Salinity SA, Conservative Temperature CT, and density rho.

    Uses TEOS-10 via gsw if available.
    Otherwise uses a simple differentiable linearized EOS as fallback.

    Parameters
    ----------
    SP : array_like
        Practical Salinity [psu].
    T : array_like
        In-situ temperature [degC].
    p : array_like or float
        Sea pressure [dbar].
    lon, lat : array_like or float, optional
        Required for full SA_from_SP; if missing, uses (0,0).

    Returns
    -------
    SA, CT, rho : ndarrays
        SA [g/kg], CT [degC], rho [kg/m^3]
    """
    SP = _ensure_array(SP)
    T = _ensure_array(T)
    p = _ensure_array(p)

    if _gsw is not None:
        if lon is None or lat is None:
            SA = _gsw.SA_from_SP(SP, p, 0.0, 0.0)
        else:
            SA = _gsw.SA_from_SP(SP, p, lon, lat)
        CT = _gsw.CT_from_t(SA, T, p)
        rho = _gsw.rho(SA, CT, p)
    else:
        # Differentiable toy EOS: for NN regularization / tests only.
        SA = SP.astype(float)
        CT = T.astype(float)
        rho0 = 1027.0
        alpha = 0.25  # thermal expansion [kg/m^3/K]
        beta = 0.75  # haline contraction [kg/m^3/(g/kg)]
        gamma = 4.5e-3  # compressibility [kg/m^3/dbar]
        rho = rho0 - alpha * (CT - 10.0) + beta * (SA - 35.0) + gamma * (p - 0.0)

    return SA, CT, rho


# -----------------------------#
# Vertical physics metrics
# -----------------------------#


def brunt_vaisala_N2_from_rho(rho, z, axis=-1, g=9.81, use_rho_local=False):
    """
    N^2 from density.

    Assumes:
      - z is depth [m], positive downward.
      - Stable stratification => rho increasing with depth => N^2 > 0.

    N^2 = (g / rho_ref) * d rho / d z  (rho_ref = mean or local).
    """
    rho = _ensure_array(rho)
    z = _ensure_array(z)

    # broadcast z
    if z.ndim == 1:
        shape = [1] * rho.ndim
        shape[axis] = z.size
        z = z.reshape(shape)

    dz = np.gradient(z, axis=axis)
    drho = np.gradient(rho, axis=axis)

    # NaN-safe division: set to NaN where dz is zero or invalid
    dz_safe = np.where(np.abs(dz) > 1e-10, dz, np.nan)
    drho_dz = np.where(np.isfinite(dz_safe), drho / dz_safe, np.nan)

    if use_rho_local:
        # NaN-safe: set to NaN where rho is invalid or zero
        rho_safe = np.where((np.abs(rho) > 1e-10) & np.isfinite(rho), rho, np.nan)
        N2 = np.where(np.isfinite(rho_safe) & np.isfinite(drho_dz), (g / rho_safe) * drho_dz, np.nan)
    else:
        # Suppress "Mean of empty slice" warning when all values are NaN
        # This is expected behavior for profiles with no valid data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            rho_ref = np.nanmean(rho, axis=axis, keepdims=True)
        # NaN-safe: set to NaN where rho_ref is invalid or zero
        rho_ref_safe = np.where((np.abs(rho_ref) > 1e-10) & np.isfinite(rho_ref), rho_ref, np.nan)
        N2 = np.where(np.isfinite(rho_ref_safe) & np.isfinite(drho_dz), (g / rho_ref_safe) * drho_dz, np.nan)

    return N2


def static_stability_metrics(rho, z, axis=-1, g=9.81, hard_fail_threshold=-1e-5):
    """
    Static stability diagnostics.

    Parameters
    ----------
    hard_fail_threshold : float, default=-1e-5
        Threshold for hard fail check: N2 < threshold indicates severe instability.

    Returns
    -------
    dict with:
      N2 : array, N^2 [s^-2]
      hard_instability_fail : bool
          True if any N2 < hard_fail_threshold (severe instability).
      frac_unstable : float
          Fraction of profiles with any N2 < 0 (soft fail).
      frac_soft_unstable : float
          Fraction of points with N2 < 0 & N2 > hard_fail_threshold (soft fail only).
      min_N2 : float
          Global minimum N2.
      int_neg_N2 : array
          Integrated negative N2 over depth per profile [m/s^2],
          same shape as rho with depth axis removed.
    """
    rho = _ensure_array(rho)
    z = _ensure_array(z)

    N2 = brunt_vaisala_N2_from_rho(rho, z, axis=axis, g=g)
    N2_last = np.moveaxis(N2, axis, -1)

    if z.ndim == 1:
        depth = z
    else:
        depth = np.moveaxis(z, axis, -1)[0, ...]

    dz = np.gradient(depth)
    n_depth = N2_last.shape[-1]
    profs = N2_last.reshape(-1, n_depth)

    if profs.size == 0:
        return dict(
            N2=N2,
            hard_instability_fail=False,
            frac_unstable=np.nan,
            frac_soft_unstable=np.nan,
            min_N2=np.nan,
            int_neg_N2=np.full(rho.shape[:-1], np.nan),
        )

    # Hard fail check: any N2 < hard_fail_threshold
    profs_finite = np.isfinite(profs)
    hard_fail = False
    if np.any(profs_finite):
        hard_fail = np.any(profs[profs_finite] < hard_fail_threshold)

    # NaN-safe: only consider finite values
    # For each profile, check if any finite value is < 0
    unstable = []
    for i in range(profs.shape[0]):
        finite_mask = profs_finite[i, :]
        if np.any(finite_mask):
            prof_finite = profs[i, :][finite_mask]
            unstable.append(np.any(prof_finite < 0.0))
        else:
            unstable.append(False)
    unstable = np.array(unstable)
    frac_unstable = np.nanmean(unstable.astype(float)) if unstable.size > 0 else np.nan

    # Soft unstable: N2 < 0 but > hard_fail_threshold
    soft_unstable_mask = (profs < 0.0) & (profs >= hard_fail_threshold) & profs_finite
    total_finite = np.sum(profs_finite)
    frac_soft_unstable = float(np.sum(soft_unstable_mask) / total_finite) if total_finite > 0 else np.nan

    # Global minimum of finite values
    min_N2 = np.nanmin(profs)

    # NaN-safe integration: only integrate finite negative values
    # For each profile, only integrate if there are finite values
    neg = np.minimum(profs, 0.0)
    int_neg = []
    for i in range(profs.shape[0]):
        prof_neg = neg[i, :]
        prof_finite = np.isfinite(profs[i, :])
        if np.any(prof_finite):
            # Only integrate finite values
            int_val = np.nansum(prof_neg[prof_finite] * dz[prof_finite])
            int_neg.append(int_val)
        else:
            # No valid data for this profile -> NaN
            int_neg.append(np.nan)
    int_neg = np.array(int_neg)
    int_neg_N2 = int_neg.reshape(rho.shape[:-1])

    return dict(
        N2=N2,
        hard_instability_fail=hard_fail,
        frac_unstable=frac_unstable,
        frac_soft_unstable=frac_soft_unstable,
        min_N2=min_N2,
        int_neg_N2=int_neg_N2,
    )


def density_smoothness_metrics(rho, z, zmin=50.0, zmax=300.0, axis=-1):
    """
    Smoothness diagnostics in a depth window [zmin, zmax].

    Returns
    -------
    dict with:
      var_d2rho_dz2 : float
          Mean (over profiles) of var(d²ρ/dz²) within window.
      mean_inflections : float
          Mean number of inflection points per profile in window.
    """
    rho = _ensure_array(rho)
    z = _ensure_array(z)
    rho_last = np.moveaxis(rho, axis, -1)

    if z.ndim == 1:
        depth = z
    else:
        depth = np.moveaxis(z, axis, -1)[0, ...]

    mask = (depth >= zmin) & (depth <= zmax)
    idx = np.where(mask)[0]
    if idx.size < 3:
        return dict(var_d2rho_dz2=np.nan, mean_inflections=np.nan)

    dz = float(np.nanmean(np.diff(depth[idx])))
    d2 = second_derivative(rho_last, dz, axis=-1)[..., idx]

    n_depth = d2.shape[-1]
    profs = d2.reshape(-1, n_depth)

    # NaN-safe variance of curvature
    var_per_prof = np.array(
        [np.nanvar(profs[i, :]) if np.any(np.isfinite(profs[i, :])) else np.nan for i in range(profs.shape[0])]
    )
    var_d2_mean = np.nanmean(var_per_prof)

    # inflection points = sign changes in curvature (NaN-safe)
    # Only consider finite values for sign changes
    sign = np.sign(profs)
    # Set NaN to 0, then propagate zeros to previous sign for robustness
    sign = np.where(np.isfinite(profs), sign, 0.0)
    for i in range(sign.shape[0]):
        for k in range(1, n_depth):
            if sign[i, k] == 0.0:
                sign[i, k] = sign[i, k - 1]
    sign_changes = (sign[:, 1:] * sign[:, :-1]) < 0.0
    # Only count sign changes where both values are finite
    finite_mask = np.isfinite(profs[:, 1:]) & np.isfinite(profs[:, :-1])
    n_inflections = np.where(finite_mask, sign_changes, False).sum(axis=1)
    mean_inflections = np.nanmean(n_inflections.astype(float))

    return dict(
        var_d2rho_dz2=var_d2_mean,
        mean_inflections=mean_inflections,
    )


# -----------------------------#
# Horizontal physics diagnostics
# -----------------------------#


def surface_geostrophic_velocity(eta, dx, dy, f, g=9.81):
    """
    Surface geostrophic velocity from SSH.

    u_g = -(g/f) * dη/dy
    v_g =  (g/f) * dη/dx

    Inputs
    ------
    eta : 2D array
        SSH [m].
    dx, dy : float
        Grid spacing [m].
    f : float or 2D array
        Coriolis parameter [1/s].

    Returns
    -------
    u_g, v_g : 2D arrays [m/s]
    """
    eta = _ensure_array(eta)
    f = _ensure_array(f)

    d_eta_dx = centered_diff(eta, dx, axis=-1)
    d_eta_dy = centered_diff(eta, dy, axis=-2)

    u_g = -(g / f) * d_eta_dy
    v_g = (g / f) * d_eta_dx

    return u_g, v_g


def thermal_wind_shear_from_rho(rho, z, dx, dy, f, rho0=1025.0, g=9.81, axis_z=-1):
    """
    Thermal-wind shear from density.

    ∂u_g/∂z = -(g / (f ρ0)) ∂ρ/∂y
    ∂v_g/∂z =  (g / (f ρ0)) ∂ρ/∂x

    rho : 3D array [y, x, z] (axis_z can be moved)
    z   : 1D depth [m], positive downward
    """
    rho = _ensure_array(rho)
    z = _ensure_array(z)

    assert z.ndim == 1

    # move z to last axis
    rho_yxz = np.moveaxis(rho, axis_z, -1)

    dρ_dx = centered_diff(rho_yxz, dx, axis=-2)
    dρ_dy = centered_diff(rho_yxz, dy, axis=-3)

    f_arr = _ensure_array(f)
    while f_arr.ndim < dρ_dx.ndim:
        f_arr = f_arr[..., None]

    du_dz = -(g / (f_arr * rho0)) * dρ_dy
    dv_dz = (g / (f_arr * rho0)) * dρ_dx

    # move derivative axis back
    du_dz = np.moveaxis(du_dz, -1, axis_z)
    dv_dz = np.moveaxis(dv_dz, -1, axis_z)

    return du_dz, dv_dz


def integrate_shear_to_surface(du_dz, dv_dz, z, z_ref, axis_z=-1):
    """
    Integrate shear from reference depth z_ref to surface (z=0):

      delta_u_g = ∫_{z_ref}^{0} (∂u_g/∂z) dz
           = - ∫_{0}^{z_ref} (∂u_g/∂z) dz

    z : 1D depth [m], positive downward, increasing.
    """
    du_dz = _ensure_array(du_dz)
    dv_dz = _ensure_array(dv_dz)
    z = _ensure_array(z)

    assert z.ndim == 1
    if not (z.min() <= z_ref <= z.max()):
        raise ValueError("z_ref outside depth range.")

    k_ref = int(np.searchsorted(z, z_ref))

    du = np.moveaxis(du_dz, axis_z, -1)
    dv = np.moveaxis(dv_dz, axis_z, -1)

    z_sub = z[: k_ref + 1]
    du_sub = du[..., : k_ref + 1]
    dv_sub = dv[..., : k_ref + 1]

    du_int = np.trapz(du_sub, z_sub, axis=-1)
    dv_int = np.trapz(dv_sub, z_sub, axis=-1)

    delta_u = -du_int
    delta_v = -dv_int

    return delta_u, delta_v


def thermal_wind_metrics(rho, z, eta, dx, dy, f, z_ref, rho0=1025.0, g=9.81, axis_z=-1, shear_depth_limit=200.0):
    """
    Thermal-wind based horizontal diagnostics.

    Uses:
      - 3D density rho(x,y,z)
      - SSH eta(x,y)

    Parameters
    ----------
    shear_depth_limit : float, default=200.0
        Depth limit (m) for calculating fraction of shear in top layer.

    Returns
    -------
    dict with:
      median_angle_deg :
          Median angle between delta_u_g (thermal-wind integrated) and u_g^sfc.
      spearman_mag_corr :
          Spearman corr between |delta_u_g| and |∇_h η|.
      frac_shear_top{shear_depth_limit} :
          Fraction of depth-integrated |∂z u_g| in top {shear_depth_limit} m.
    """
    # surface geostrophic
    u_sfc, v_sfc = surface_geostrophic_velocity(eta, dx, dy, f, g=g)

    # shear from density
    du_dz, dv_dz = thermal_wind_shear_from_rho(rho, z, dx, dy, f, rho0=rho0, g=g, axis_z=axis_z)

    # vertically integrated geostrophic shear
    delta_u, delta_v = integrate_shear_to_surface(du_dz, dv_dz, z, z_ref, axis_z=axis_z)

    # angle between delta_u_g and surface geostrophic velocity
    ang = angle_between(delta_u, delta_v, u_sfc, v_sfc)
    median_angle = float(np.nanmedian(ang))

    # magnitude consistency with |∇h η|
    mag_delta_u = np.hypot(delta_u, delta_v)
    d_eta_dx = centered_diff(eta, dx, axis=-1)
    d_eta_dy = centered_diff(eta, dy, axis=-2)
    mag_grad_eta = np.hypot(d_eta_dx, d_eta_dy)
    spearman_mag_corr = float(spearmanr(mag_delta_u, mag_grad_eta))

    # vertical distribution of |∂z u_g|
    du = np.moveaxis(du_dz, axis_z, -1)
    dv = np.moveaxis(dv_dz, axis_z, -1)
    abs_shear = np.hypot(du, dv)

    total = np.trapz(abs_shear, z, axis=-1)
    mask_top = (z >= 0) & (z <= shear_depth_limit)
    if mask_top.any():
        top = np.trapz(abs_shear[..., mask_top], z[mask_top], axis=-1)
        frac_top = float(np.nanmean(top / (total + 1e-12)))
    else:
        frac_top = np.nan

    return dict(
        median_angle_deg=median_angle,
        spearman_mag_corr=spearman_mag_corr,
        **{f"frac_shear_top{int(shear_depth_limit)}": frac_top},
    )


def ekman_pumping(tau_x, tau_y, dx, dy, f, rho0=1025.0):
    """
    Ekman pumping:
      w_E = (1 / (ρ0 f)) (∂τ_y/∂x - ∂τ_x/∂y)
    """
    tau_x = _ensure_array(tau_x)
    tau_y = _ensure_array(tau_y)

    curl_tau = centered_diff(tau_y, dx, axis=-1) - centered_diff(tau_x, dy, axis=-2)

    f_arr = _ensure_array(f)
    w_E = curl_tau / (rho0 * f_arr)
    return w_E


def isopycnal_depth(rho, z, sigma, axis_z=-1):
    """
    Depth of a given isopycnal (e.g. sigma_theta = 25).

    Finds the shallowest depth where rho crosses sigma (linear in-between levels).

    Parameters
    ----------
    rho : array_like
        Density field [..., z].
    z : 1D array
        Depth [m], positive downward.
    sigma : float
        Target density [kg/m^3].

    Returns
    -------
    z_sigma : array [...], depth [m] (nan where no crossing).
    """
    rho = _ensure_array(rho)
    z = _ensure_array(z)

    rho_last = np.moveaxis(rho, axis_z, -1)
    Nz = z.size
    out = np.full(rho_last.shape[:-1], np.nan, dtype=float)

    diff = rho_last - sigma

    for k in range(Nz - 1):
        d0 = diff[..., k]
        d1 = diff[..., k + 1]

        # crossing sigma between k and k+1
        mask = (d0 <= 0.0) & (d1 >= 0.0)
        if not np.any(mask):
            continue

        z0 = z[k]
        z1 = z[k + 1]
        denom = d1 - d0
        frac = np.zeros_like(d0, dtype=float)

        valid = mask & (denom != 0.0)
        frac[valid] = -d0[valid] / denom[valid]

        z_iso = z0 + frac * (z1 - z0)

        # keep shallowest crossing
        replace = mask & (np.isnan(out) | (z_iso < out))
        out[replace] = z_iso[replace]

    return out


def ekman_tilt_metrics(rho, z, tau_x, tau_y, dx, dy, f, sigma, rho0=1025.0, z_sigma_lagged=None):
    """
    Ekman tilt diagnostics.

    Steps:
      - compute w_E from wind stress curl
      - get isopycnal depth z_sigma(x,y) from density (or use provided lagged depth)
      - compare -z_sigma vs w_E

    Parameters
    ----------
    z_sigma_lagged : array, optional
        Pre-computed isopycnal depth (e.g., from day t+1 when wind is from day t).
        If provided, uses this instead of computing from rho.

    Returns
    -------
    dict with:
      corr_negz_we :
          Pearson correlation between -z_sigma and w_E.
      slope :
          Regression slope of (-z_sigma) on w_E.
    """
    w_E = ekman_pumping(tau_x, tau_y, dx, dy, f, rho0=rho0)

    if z_sigma_lagged is not None:
        z_sigma = z_sigma_lagged
    else:
        z_sigma = isopycnal_depth(rho, z, sigma)

    zs = (-z_sigma).ravel()
    we = w_E.ravel()
    mask = np.isfinite(zs) & np.isfinite(we)

    if mask.sum() < 3:
        return dict(corr_negz_we=np.nan, slope=np.nan)

    zs = zs[mask]
    we = we[mask]

    zs_m = zs - zs.mean()
    we_m = we - we.mean()

    num = float(np.sum(zs_m * we_m))
    den = math.sqrt(float(np.sum(zs_m**2) * np.sum(we_m**2)))
    corr = num / den if den > 0.0 else np.nan

    var_we = float(np.mean(we_m**2))
    slope = num / (len(we) * var_we + 1e-12)

    return dict(
        corr_negz_we=corr,
        slope=slope,
    )
