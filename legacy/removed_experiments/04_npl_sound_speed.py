"""ARCHIVED — monolith __main__ block (Phase 9 removal, commit 2cbb2b1).

Source: legacy/monolith/singleFileModel_SAT_stats4verticalProj_meeting20260203.py
lines ~295-327. NPL sound-speed / SLD / BLG acoustics.

NOT wired. Code below is preserved verbatim (still commented).
"""


    # %%
    # def calculate_sound_speed_NPL(T, S, Z, Phi=45):
    # """
    # Calculate sound speed (in m/s) using the NPL equation.
    # T: Temperature in degrees Celsius
    # S: Salinity in PSU
    # Z: Depth in meters
    # Phi: Latitude in degrees (default 45)
    # """
    # c = (1402.5 + 5 * T - 5.44e-2 * T**2 + 2.1e-4 * T**3
    #      + 1.33 * S - 1.23e-2 * S * T + 8.7e-5 * S * T**2
    #      + 1.56e-2 * Z + 2.55e-7 * Z**2 - 7.3e-12 * Z**3
    #      + 1.2e-6 * Z * (Phi - 45) - 9.5e-13 * T * Z**3
    #      + 3e-7 * T**2 * Z + 1.43e-5 * S * Z)
    # return c

    # # Recalculate sound speed at each depth using the NPL equation
    # sound_speed_profile_NPL = np.array([calculate_sound_speed_NPL(T, S, z) for T, S, z in zip(temperature_profile, salinity_profile, depths)])

    # # Finding the Sonic Layer Depth (SLD) using the NPL equation
    # max_sound_speed_index_NPL = np.argmax(sound_speed_profile_NPL)
    # SLD_NPL = depths[max_sound_speed_index_NPL]
    # # Conversion factor from meters to feet
    # meters_to_feet = 3.28084

    # # Conversion factor for the gradient from per feet to per 100 meters
    # conversion_factor = meters_to_feet / 100

    # # Calculating the Below Layer Gradient (BLG) using the NPL equation
    # gradient_NPL = np.gradient(sound_speed_profile_NPL, depths_feet)
    # # Average gradient below MLD in m/s per 100 feet using the NPL equation
    # BLG_NPL = np.mean(gradient_NPL[MLD_index:]) * conversion_factor
