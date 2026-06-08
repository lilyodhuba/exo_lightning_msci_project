"""
Functions and parameters for droplet properties.

Unless otherwise stated, the reference is:
Loftus, K., & Wordsworth, R. D. (2021). The physics of falling raindrops in diverse planetary atmospheres.
Journal of Geophysical Research: Planets, 126, e2020JE006653. https://doi.org/10.1029/2020JE006653
"""

import numpy as np
from scipy import optimize


def _f_droplet_terminal_velocity(
    v_term0: float,
    r_eq: float,
    air_dens: float,
    liq_dens: float,
    surf_tens: float,
    gravity: float,
    method_draf_coef: str = "loth08",
) -> float:
    """
    Function for a numerical solver to calculate terminal velocity.

    Parameters
    ----------
    v_term0: float
        First guess terminal velocity [m s-1]
    r_eq: float
        Equivalent volume spherical radius of the droplet [m]
    fall_speed: float
        Droplet fall speed [m s-1]
    air_dens: float
        Air density [kg m-3]
    liq_dens: float
        Condensible liquid density [kg m-3]
    surf_tens: float
        Condensible liquid surface tension [N m-1]
    gravity: float
        Acceleration due to gravity [m s-2]
    method_draf_coef: str
        Parameterization. Default: Loth+ (2008).
    Returns
    -------
    float
        Difference between v predicted by given v and given v [m s-1]
    """
    try:
        shape_ratio = calc_droplet_shape_ratio(
            r_eq, liq_dens, air_dens, surf_tens, gravity
        )
    except Exception:
        # TODO: implement a better catch
        shape_ratio = 1.0
    xsec_area = r_eq**2 * shape_ratio ** (
        -2.0 / 3.0
    )  # [m2] cross sectional area of oblate spheroid
    drag_coef = droplet_drag_coef(
        r_eq,
        v_term0,
        air_dens,
        liq_dens,
        surf_tens,
        gravity,
        method=method_draf_coef,
    )
    if r_eq < 1e-9:  # avoid errors from ode solver trying unphysical values
        return 0.0
    else:
        return (
            v_term0
            - (
                8.0
                / 3.0
                * r_eq**3
                / xsec_area
                * gravity
                * (liq_dens - air_dens)
                / air_dens
                / drag_coef
                # TODO: add correction factor?
            )
            ** 0.5
        )


def droplet_terminal_velocity(
    r_eq: float,
    air_dens: float,
    liq_dens: float,
    surf_tens: float,
    gravity: float,
    method_draf_coef: str = "loth08",
    v_min: float = 1e-10,
    v_max: float = 300,
) -> float:
    """
    Calculate raindrop terminal velocity numerically.

    Parameters
    ----------
    v_term0: float
        First guess terminal velocity [m s-1]
    r_eq: float
        Equivalent volume spherical radius of the droplet [m]
    fall_speed: float
        Droplet fall speed [m s-1]
    air_dens: float
        Air density [kg m-3]
    liq_dens: float
        Condensible liquid density [kg m-3]
    surf_tens: float
        Condensible liquid surface tension [N m-1]
    gravity: float
        Acceleration due to gravity [m s-2]
    method_draf_coef: str
        Parameterization. Default: Loth+ (2008).
    Returns
    -------
    float
        Terminal velocity [m s-1]
    """
    try:
        # first try using Newton root solver
        v = optimize.newton(
            _f_droplet_terminal_velocity,
            1.5,
            args=(r_eq, air_dens, liq_dens, surf_tens, gravity, method_draf_coef),
            maxiter=100,
        )
        v = float(v)
    except Exception as e:
        print(f"In droplet_terminal_velocity():\n\n{e}")
        # then try a bounded root solver if it fails
        v = optimize.brentq(
            _f_droplet_terminal_velocity,
            v_min,
            v_max,
            args=(r_eq, air_dens, liq_dens, surf_tens, gravity, method_draf_coef),
        )
    return v


def droplet_reynolds_number(
    r_eq: float, fall_speed: float, air_dens: float, dyn_visc_air: float = 1.789e-5
) -> float:
    # TODO: Calculate viscosity from local conditions
    """
    Calculate the Reynolds number.

    Parameters
    ----------
    r_eq: float
        Equivalent volume spherical radius of the droplet [m]
    fall_speed: float
        Droplet fall speed [m s-1]
    air_dens: float
        Air density [kg m-3]
    dyn_visc_air: float
        Dynamic viscosity of air [Pa s]

    Returns
    -------
    float:
        Reynolds number []
    """
    return 2 * air_dens * r_eq * abs(fall_speed) / dyn_visc_air


def droplet_drag_coef(
    r_eq: float,
    fall_speed: float,
    air_dens: float,
    liq_dens: float,
    surf_tens: float,
    gravity: float,
    method: str = "loth08",
):
    """
    Calculate the drag coefficient of an oblate spheroid.

    Parameters
    ----------
    r_eq: float
        Equivalent volume spherical radius of the droplet [m]
    fall_speed: float
        Droplet fall speed [m s-1]
    air_dens: float
        Air density [kg m-3]
    liq_dens: float
        Condensible liquid density [kg m-3]
    surf_tens: float
        Condensible liquid surface tension [N m-1]
    gravity: float
        Acceleration due to gravity [m s-2]
    method: str
        Parameterization. Default: Loth+ (2008).

    Returns
    -------
    float:
        Drag coefficient []

    References
    ----------
    * Loth+ (2008)
    * LW21 eq.6-7
    """
    if method != "loth08":
        raise NotImplementedError(f'Invalid method "{method}".')
    # TODO: viscosity
    re_number = droplet_reynolds_number(r_eq, fall_speed, air_dens)
    try:
        shape_ratio = calc_droplet_shape_ratio(
            r_eq, liq_dens, air_dens, surf_tens, gravity
        )
    except Exception:
        # TODO: implement a better catch
        shape_ratio = 1.0
    if shape_ratio < 0.999:
        try:
            eps = (1 - shape_ratio**2) ** 0.5
            surf_area_ratio = shape_ratio ** (-2.0 / 3.0) / 2.0 + shape_ratio ** (
                4.0 / 3.0
            ) / 4.0 / eps * np.log((1 + eps) / (1 - eps))  # Loth et al. (2008) eq (21a)
            non_sphere_correction = (
                1 + 1.5 * (surf_area_ratio - 1) ** 0.5 + 6.7 * (surf_area_ratio - 1)
            )  # Loth et al. (2008) eq (25)
        except Exception as e:
            print(f"In droplet_drag_coef():\n\n{e}")
            non_sphere_correction = 1.0
    else:
        non_sphere_correction = 1.0
    drag_coef = 24.0 / re_number * (1 + 0.15 * re_number ** (229.0 / 333.0)) + 0.42 / (
        1.0 + 4.25e4 * re_number ** (-1.16)
    )  # Clift & Gauvin (1970)
    # 229.0/333.0 = 0.687 repeating (shown by an overbar)
    # from Clift+ (2005) Table 5.1 or Loth+ (2008) eq (8) (latter doesn't show repeating)
    drag_coef *= non_sphere_correction  # correct C_D with shape factor
    return drag_coef


def liquid_surface_tension(temperature):
    """
    Calculate condensible liquid surface tension as a function of temperature

    Parameters
    ----------
    temperature: float
        Air temperature [K]

    Returns
    -------
    float:
        Surface tension [N m-1]

    References
    ----------
    * LW21
    * Vargaftik+ (1983)
    """
    B = 235e-3  # [N/m]
    b = -0.625  # []
    mu = 1.256  # []
    temp_crit = 647.15  # [K]
    surf_tens = (
        B
        * ((temp_crit - temperature) / temp_crit) ** mu
        * (1 + b * ((temp_crit - temperature) / temp_crit))
    )
    return surf_tens


def _f_droplet_shape_ratio(
    b_to_a: float,
    r_eq: float,
    liq_dens: float,
    air_dens: float,
    surf_tens: float,
    gravity: float,
) -> float:
    """
    Function for a numerical solver to calculate the shape of a droplet as an oblate spheroid.

    Parameters
    ----------
    b_to_a: float
        b/a, ratio of the droplet []
    r_eq: float
        Equivalent volume spherical radius of the droplet [m]
    liq_dens: float
        Condensible liquid density [kg m-3]
    air_dens: float
        Air density [kg m-3]
    surf_tens: float
        Condensible liquid surface tension [N m-1]
    gravity: float
        Acceleration due to gravity [m s-2]

    Returns
    -------
    float
        Difference between droplet size predicted by given input ratio and actual drop size [m]
    """
    return (
        r_eq
        - (surf_tens / gravity / (liq_dens - air_dens)) ** 0.5
        * b_to_a ** (-1.0 / 6.0)
        * (b_to_a ** (-2.0) - 2.0 * b_to_a ** (-1.0 / 3.0) + 1.0) ** 0.5
    )


def calc_droplet_shape_ratio(
    r_eq: float,
    liq_dens: float,
    air_dens: float,
    surf_tens: float,
    gravity: float,
) -> float:
    """
    Calculate (equilibrium) shape of drop as an oblate spheroid in a form of b/a ratio.

    Parameters
    ----------
    r_eq: float
        Equivalent volume spherical radius of the droplet [m]
    liq_dens: float
        Condensible liquid density [kg m-3]
    air_dens: float
        Air density [kg m-3]
    surf_tens: float
        Condensible liquid surface tension [N m-1]
    gravity: float
        Acceleration due to gravity [m s-2]

    Returns
    -------
    float:
        Shape ratio b/a []

    References
    ----------
    * Green (1975) eq.6 [note a small typo in G75, σ should be raised to 0.5 rather than 1]
    * LW21 eq.2
    """

    return optimize.brentq(
        _f_droplet_shape_ratio,
        1e-2,
        1,
        args=(r_eq, liq_dens, air_dens, surf_tens, gravity),
    )
