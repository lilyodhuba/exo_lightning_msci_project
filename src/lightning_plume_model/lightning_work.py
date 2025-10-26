"""
Earth 1-D Lightning Simulation - Minimally Modified from Original

Only changes from original:
1. Made n_bins a parameter (default 31)
2. Created plot_comparison() function for organized plotting
3. Slightly cleaner main() function
4. Moved physical constants into a separate dataclass
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used in lightning plume model calculations."""

    g: float = 9.81  # Gravitational acceleration [m/s2]
    R: float = 8.31446  # Universal gas constant [J/mol/K]
    mu: float = 0.02896  # Molar mass of dry air [kg/mol]
    epsilon: float = 0.6222  # Ratio of molecular weights [dimensionless]
    c_p: float = 14500.0  # Specific heat capacity at constant pressure [J/kg/K]
    L: float = 2257000.0  # Latent heat of vaporization of water [J/kg]
    eps0: float = 8.854e-12  # Vacuum permittivity [F/m]
    e_charge: float = 1.602e-19  # Elementary charge [C]
    rho_water: float = 1000.0  # Density of liquid water [kg/m3]
    rhoro: float = 2.5  # Ratio of ice to liquid water density [dimensionless]
    Cdrag: float = 0.5  # Drag coefficient [dimensionless]
    Eflash: float = 1.5e9  # Energy per lightning flash [J]
    mfptime: float = 4.0e-11  # Mean free path time for ion collisions [s]
    temp_freeze: float = 273.15  # Freezing point of water [K]
    bar_to_pa: float = 1e5  # Conversion factor from bar to Pascal [Pa/bar]


# Initialize physical constants
CONST = PhysicalConstants()


@dataclass(frozen=True)
class SimulationParameters:
    """Configurable parameters for a single simulation run."""

    plume_base_temp: float
    base_humidity_fraction: float
    plume_base_radius: float
    temp_supercool: float
    water_collision_efficiency: float
    ice_collision_efficiency: float
    start_pressure: float = 100_000.0
    start_upward_velocity: float = 0.001
    pressure_step: float = 10.0
    growth_time_step: float = 0.01
    n_bins: int = 31
    min_radius: float = 1e-5
    max_radius: float = 0.46340950011842
    flash_rate_step: int = 10


def saturation_vapour_pressure(temp: float) -> float:
    """Saturation vapour pressure of water (Lowe 1977)."""
    a0 = 6984.505294
    a1 = -188.9039310
    a2 = 2.133357675
    a3 = -0.01288580973
    a4 = 4.393587233e-5
    a5 = -8.023923082e-8
    a6 = 6.136820929e-11
    mb = a0 + temp * (
        a1 + temp * (a2 + temp * (a3 + temp * (a4 + temp * (a5 + temp * a6))))
    )
    return max(mb * 100.0, 0.0)


def temp_strat(pressure: float) -> float:
    """International Standard Atmosphere stratospheric temperature."""
    if pressure > 5474.9:
        return 216.6
    elif pressure > 868.02:
        return 216.6 + 12.0 * np.log(pressure / 5474.9) / np.log(868.02 / 5474.9)
    elif pressure > 110.91:
        return 228.6 + 42.0 * np.log(pressure / 868.02) / np.log(110.91 / 868.02)
    elif pressure > 66.939:
        return 270.6
    else:
        return 270.6 - 56.0 * np.log(pressure / 66.939) / np.log(3.9564 / 66.939)


def dry_adiabat(
    P: float,
    T: float,
    f: float,
    c_p: float = CONST.c_p,
    const: PhysicalConstants = CONST,
) -> float:
    r"""
    Calculate the dry adiabatic temperature gradient.

    This function computes the dry adiabatic temperature gradient, which describes
    how temperature changes with pressure in a dry air parcel under adiabatic conditions.

    The formula used is:
    .. math::
        \frac{dT}{dP} = \frac{RT}{\mu P} \cdot \frac{1 + f/\epsilon}{1 + f} \cdot \frac{1}{c_p}

    where:
    - :math:`R` is the universal gas constant [J/mol/K]
    - :math:`T` is temperature [K]
    - :math:`\mu` is the molar mass of dry air [kg/mol]
    - :math:`P` is pressure [Pa]
    - :math:`f` is the mixing ratio [kg/kg]
    - :math:`\epsilon` is the ratio of molecular weights of water vapor to dry air [dimensionless]
    - :math:`c_p` is the specific heat capacity at constant pressure [J/kg/K]

    Parameters
    ----------
    P : float
        Pressure [Pa]
    T : float
        Temperature [K]
    f : float
        Mixing ratio [kg/kg]
    c_p : float, optional
        Specific heat capacity at constant pressure [J/kg/K]
    const : PhysicalConstants, optional
        Physical constants object

    Returns
    -------
    float
        Temperature gradient dT/dP [K/Pa]
    """
    return const.R * T * ((1 + f / const.epsilon) / (1 + f)) / (const.mu * P * c_p)


def entrainment(
    T: float,
    P: float,
    plume_radius: float = 5000.0,
    const: PhysicalConstants = CONST,
) -> float:
    r"""
    Calculate the entrainment parameter phi.

    The entrainment parameter is given by:
    ..math::
        phi = -0.2 * R * T / (r * mu * P * g)

    where:
    - :math:`R` is the universal gas constant [J/mol/K]
    - :math:`T` is temperature [K]
    - :math:`\mu` is the molar mass of dry air [kg/mol]
    - :math:`P` is pressure [Pa]
    - :math:`g` is gravity [m/s2]

    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [Pa]
    plume_radius : float, optional
        Radius of updraft [m], defaults to 5000.0
    const : PhysicalConstants, optional
        Physical constants object

    Returns
    -------
    float
        Entrainment parameter (phi) [1/Pa]
    """
    return -0.2 * const.R * T / (plume_radius * const.mu * P * const.g)


def moist_adiabat(
    P: float,
    Trise: float,
    Tfall: float,
    lcondensate: float,
    frise: float,
    ffall: float,
    satvappre: float,
    entrain_param: float,
    c_p: float = CONST.c_p,
    const: PhysicalConstants = CONST,
) -> float:
    r"""
    Calculate the moist adiabatic temperature gradient.

    This function computes the moist adiabatic lapse rate taking into account condensation
    and entrainment effects. The calculation differs depending on whether the parcel is
    saturated (fS <= frise) or unsaturated.

    For saturated conditions:
    .. math::
        \Gamma_{m} = \Gamma_d \frac{1 + \frac{L f_s \mu}{RT_v}
            - \frac{(T_r - T_f)\phi}{\Gamma_d} - \frac{L(f_s - f_f)\phi}{\Gamma_d c_p}}{1 + \frac{L^2 f_s \epsilon \mu}{c_p RT^2}}

    For unsaturated conditions:
    .. math::
        \Gamma_{m} = \Gamma_d - (T_r - T_f)\phi

    where:
    - :math:`\Gamma_d` is the dry adiabatic lapse rate
    - :math:`\phi` is the entrainment parameter
    - :math:`f_s` is the saturation mixing ratio

    Parameters
    ----------
    P : float
        Pressure [Pa]
    Trise : float
        Temperature of rising air [K]
    Tfall : float
        Temperature of falling air [K]
    lcondensate : float
        Latent heat of condensation [J/kg]
    frise : float
        Mixing ratio of rising air [kg/kg]
    ffall : float
        Mixing ratio of falling air [kg/kg]
    satvappre : float
        Saturation vapor pressure [Pa]
    entrain_param : float
        Entrainment parameter [1/Pa]
    c_p : float, optional
        Specific heat capacity at constant pressure [J/kg/K]
    const : PhysicalConstants, optional
        Physical constants object

    Returns
    -------
    float
        Moist adiabatic temperature gradient [K/m]
    """
    Gamma = dry_adiabat(P, Trise, frise, c_p=c_p, const=const)
    fS = const.epsilon * satvappre / P

    if fS <= frise:
        Tv = Trise * ((1 + frise / const.epsilon) / (1 + frise))
        numer = (
            1
            + const.L * fS * const.mu / (const.R * Tv)
            - ((Trise - Tfall) * entrain_param / Gamma)
            - const.L * (fS - ffall) * entrain_param / (Gamma * c_p)
        )
        denom = 1 + (const.L * const.L * fS * const.epsilon * const.mu) / (
            c_p * const.R * Trise * Trise
        )
        return Gamma * numer / denom
    else:
        return Gamma - ((Trise - Tfall) * entrain_param)


def upward_wind_gradient(
    P: float,
    Trise: float,
    Tfall: float,
    lcondensate: float,
    frise: float,
    ffall: float,
    w: float,
    entrain_param: float,
    const: PhysicalConstants = CONST,
) -> float:
    r"""
    Calculate the vertical velocity gradient in a convective plume.

    This function computes the vertical velocity gradient based on thermodynamic
    properties and physical constants. The gradient is derived from the equation:

    .. math::
        \frac{dw}{dP} = -\frac{R}{P\mu w}\left[T_{rise}(1-l)\frac{1+f_{rise}/\epsilon}{1+f_{rise}}
        - T_{fall}\frac{1+f_{fall}/\epsilon}{1+f_{fall}}\right] - w\phi

    Parameters
    ----------
    P : float
        Pressure [Pa]
    Trise : float
        Temperature of rising air [K]
    Tfall : float
        Temperature of falling air [K]
    lcondensate : float
        Liquid water content ratio [dimensionless]
    frise : float
        Water vapor mixing ratio in rising air [kg/kg]
    ffall : float
        Water vapor mixing ratio in falling air [kg/kg]
    w : float
        Vertical velocity [m/s]
    entrain_param : float,
        Entrainment parameter [1/Pa]
    const : PhysicalConstants, optional
        Object containing physical constants, defaults to CONST

    Returns
    -------
    float
        Vertical velocity gradient [m/s/Pa]

    Notes
    -----
    The equation represents the change in vertical velocity with respect to pressure,
    accounting for temperature differences between rising and falling air, water vapor
    content, and entrainment effects.
    """
    dwdPn = (
        -const.R
        * (
            Trise * (1 - lcondensate) * ((1 + frise / const.epsilon) / (1 + frise))
            - Tfall * ((1 + ffall / const.epsilon) / (1 + ffall))
        )
        / (P * const.mu * w)
        - w * entrain_param
    )
    return dwdPn


def stepgrow(
    n0s: np.ndarray,
    slopes: np.ndarray,
    binbounds: np.ndarray,
    upbooms: np.ndarray,
    rho: float,
    Eij: np.ndarray,
    vrel: np.ndarray,
    delt: float,
    showdetails: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Particle growth through collisions - COPIED EXACTLY FROM ORIGINAL."""
    r0s = np.zeros(len(n0s))
    Ns = np.zeros(len(n0s))
    Ms = np.zeros(len(n0s))
    n0snew = np.zeros(len(n0s))
    slopesnew = np.zeros(len(n0s))
    Nsnew = np.zeros(len(n0s))
    Msnew = np.zeros(len(n0s))
    mmean = np.zeros(len(n0s))
    upbos = np.zeros(len(n0s))

    if showdetails == 1:
        print("n0s", n0s)
        print("slopes", slopes)

    for ssa in range(len(n0s)):
        if n0s[ssa] + 0.5 * (binbounds[ssa] - binbounds[ssa + 1]) * slopes[ssa] <= 0:
            upbos[ssa] = binbounds[ssa]
        elif n0s[ssa] + 0.5 * (binbounds[ssa + 1] - binbounds[ssa]) * slopes[ssa] >= 0:
            upbos[ssa] = binbounds[ssa + 1]
        else:
            upbos[ssa] = (
                0.5 * binbounds[ssa] + 0.5 * binbounds[ssa + 1] - n0s[ssa] / slopes[ssa]
            )

    for s in range(len(n0s)):
        r0s[s] = (binbounds[s + 1] + binbounds[s]) / 2.0

    for s in range(len(n0s)):
        if upbos[s] >= binbounds[s + 1]:
            Ns[s] = n0s[s] * (binbounds[s + 1] - binbounds[s])
            Ms[s] = (4 * np.pi * rho / 3.0) * (
                (0.2 * (binbounds[s + 1] ** 5 - binbounds[s] ** 5) * slopes[s])
                + (
                    0.25
                    * (binbounds[s + 1] ** 4 - binbounds[s] ** 4)
                    * (n0s[s] - slopes[s] * r0s[s])
                )
            )
            mmean[s] = Ms[s] / Ns[s]
        elif upbos[s] > binbounds[s]:
            Ns[s] = n0s[s] * (upbos[s] - binbounds[s]) + slopes[s] * (
                upbos[s] - binbounds[s]
            ) * (upbos[s] / 2.0 + binbounds[s] / 2.0 - r0s[s])
            Ms[s] = (4 * np.pi * rho / 3.0) * (
                (0.2 * (upbos[s] ** 5 - binbounds[s] ** 5) * slopes[s])
                + (
                    0.25
                    * (upbos[s] ** 4 - binbounds[s] ** 4)
                    * (n0s[s] - slopes[s] * r0s[s])
                )
            )
            mmean[s] = Ms[s] / Ns[s]
        else:
            Ns[s] = 0.0
            Ms[s] = 0.0
            mmean[s] = (
                (1 / 3.0) * np.pi * rho * (binbounds[s + 1] ** 4 - binbounds[s] ** 4)
            )

    if showdetails == 1:
        print("Ns", Ns)
        print("Ms", Ms)

    for s in range(len(n0s)):
        Nsnew[s] = Ns[s]
        Msnew[s] = Ms[s]

    for i in range(len(n0s)):
        for j in range(len(n0s)):
            if i >= j:
                lambdaij = min(
                    Eij[i, j]
                    * np.pi
                    * (r0s[i] ** 2 + r0s[j] ** 2)
                    * vrel[i, j]
                    * Ns[j]
                    * delt,
                    1.0,
                )
                Nsnew[j] = Nsnew[j] - lambdaij * Ns[i]
                Msnew[j] = Msnew[j] - lambdaij * Ns[i] * mmean[j]
                Msnew[i] = Msnew[i] + lambdaij * Ns[i] * mmean[j]
                rx = (binbounds[i + 1] ** 3 - r0s[j] ** 3) ** (1 / 3.0)

                if upbos[i] >= binbounds[i + 1]:
                    rxx = binbounds[i + 1]
                    Nxx = (
                        n0s[i] * (rxx - rx)
                        - slopes[i] * r0s[i] * (rxx - rx)
                        + slopes[i] * (rxx - rx) * (rxx + rx) / 2.0
                    )
                    Mxx = (np.pi * rho / 3.0) * (rxx**4 - rx**4) * (
                        n0s[i] - slopes[i] * r0s[i]
                    ) + (4 * np.pi * rho / 15.0) * slopes[i] * (rxx**5 - rx**5)
                elif upbos[i] > rx:
                    rxx = upbos[i] + 0.0
                    Nxx = (
                        n0s[i] * (rxx - rx)
                        - slopes[i] * r0s[i] * (rxx - rx)
                        + slopes[i] * (rxx - rx) * (rxx + rx) / 2.0
                    )
                    Mxx = (np.pi * rho / 3.0) * (rxx**4 - rx**4) * (
                        n0s[i] - slopes[i] * r0s[i]
                    ) + (4 * np.pi * rho / 15.0) * slopes[i] * (rxx**5 - rx**5)
                else:
                    Nxx = 0.0
                    Mxx = 0.0

                if (i + 1) < len(n0s):
                    Msnew[i + 1] = (
                        Msnew[i + 1] + Mxx * lambdaij + mmean[j] * Nxx * lambdaij
                    )
                    Msnew[i] = Msnew[i] - Mxx * lambdaij - mmean[j] * Nxx * lambdaij
                    Nsnew[i + 1] = Nsnew[i + 1] + Nxx * lambdaij
                    Nsnew[i] = Nsnew[i] - Nxx * lambdaij

    if showdetails == 1:
        print("Nsnew", Nsnew)
        print("Msnew", Msnew)

    for s in range(len(n0s)):
        R2 = 0.5 * (binbounds[s + 1] ** 2 - binbounds[s] ** 2)
        R4 = 0.25 * (binbounds[s + 1] ** 4 - binbounds[s] ** 4)
        R5 = 0.2 * (binbounds[s + 1] ** 5 - binbounds[s] ** 5)
        n0snewtest = Nsnew[s] / (binbounds[s + 1] - binbounds[s])
        slopesnewtest = (
            3 * (binbounds[s + 1] - binbounds[s]) * Msnew[s] / (4 * rho * np.pi)
            - Nsnew[s] * R4
        ) / ((binbounds[s + 1] - binbounds[s]) * R5 - R2 * R4)
        ncrit = n0snewtest + slopesnewtest * (binbounds[s + 1] - r0s[s])

        if (Nsnew[s] <= 0) or (Msnew[s] <= 0):
            upbos[s] = binbounds[s]
            slopesnew[s] = 0.0
            n0snew[s] = 0.0
        elif ncrit >= 0:
            n0snew[s] = n0snewtest
            slopesnew[s] = slopesnewtest
            upbos[s] = binbounds[s + 1]
        else:
            bquar = binbounds[s]
            Nquar = Nsnew[s]
            Mquar = 3.0 * Msnew[s] / (4.0 * rho * np.pi)
            uarray = np.roots(
                [
                    1.0,
                    2.0 * bquar,
                    3.0 * (bquar**2),
                    4.0 * (bquar**3) - 10.0 * Mquar / Nquar,
                ]
            )
            uquar = binbounds[s]
            for tt in range(len(uarray)):
                uposs = uarray[tt]
                if uposs.real >= binbounds[s] and uposs.real < binbounds[s + 1]:
                    if abs(uposs.imag) < 10**-10:
                        if showdetails == 1:
                            print(s, uposs)
                        uquar = uposs.real + 0.0
            upbos[s] = uquar
            if showdetails == 1:
                print(s, upbos[s])
            if upbos[s] > binbounds[s]:
                R2 = 0.5 * (upbos[s] ** 2 - binbounds[s] ** 2)
                R4 = 0.25 * (upbos[s] ** 4 - binbounds[s] ** 4)
                R5 = 0.2 * (upbos[s] ** 5 - binbounds[s] ** 5)
                n0snewerm = Nsnew[s] / (upbos[s] - binbounds[s])
                if showdetails == 1:
                    print(s, R2, R4, R5, n0snewerm)
                    print(
                        s,
                        (
                            (3 * (upbos[s] - binbounds[s]) * Msnew[s])
                            / (4 * rho * np.pi)
                        ),
                        Nsnew[s] * R4,
                        ((upbos[s] - binbounds[s]) * R5),
                        (R2 * R4),
                    )
                slopesnew[s] = (
                    (
                        ((3 * (upbos[s] - binbounds[s]) * Msnew[s]) / (4 * rho * np.pi))
                        - Nsnew[s] * R4
                    )
                    * 40.0
                    / (
                        (
                            3.0 * (upbos[s] * upbos[s])
                            + 4.0 * (upbos[s] * binbounds[s])
                            + 3.0 * (binbounds[s] * binbounds[s])
                        )
                        * (upbos[s] - binbounds[s]) ** 4
                    )
                )
                n0snew[s] = (
                    n0snewerm
                    + (r0s[s] - (binbounds[s] + upbos[s]) / 2.0) * slopesnew[s]
                )
                if showdetails == 1:
                    print(s, n0snew[s], slopesnew[s])
            else:
                slopesnew[s] = 0.0
                n0snew[s] = 0.0

    if showdetails == 1:
        print("n0snew", n0snew)
        print("slopesnew", slopesnew)

    return n0snew, slopesnew, upbos


def dQdt(
    n0s: np.ndarray,
    slopes: np.ndarray,
    binbounds: np.ndarray,
    upperbound: np.ndarray,
    velocities: np.ndarray,
    Qcoefff: float = 1.0,
    ioncharges: List[float] = [],
    ionnumbers: List[float] = [],
    ionvelocities: List[float] = [],
    radju: float = 1,
) -> np.ndarray:
    """Calculate charging rate - COPIED FROM ORIGINAL."""
    r0s = np.zeros(len(n0s))
    ns = np.zeros(len(n0s))
    dQidt = np.zeros(len(n0s))

    for s in range(len(n0s)):
        r0s[s] = (binbounds[s + 1] + binbounds[s]) / 2.0
        if n0s[s] + 0.5 * (binbounds[s] - binbounds[s + 1]) * slopes[s] <= 0:
            rpl = binbounds[s]
        elif n0s[s] + 0.5 * (binbounds[s + 1] - binbounds[s]) * slopes[s] >= 0:
            rpl = binbounds[s + 1]
        else:
            rpl = 0.5 * binbounds[s] + 0.5 * binbounds[s + 1] - n0s[s] / slopes[s]
        rmi = binbounds[s]
        ns[s] = (rpl - rmi) * (n0s[s] - 0.5 * (binbounds[s + 1] + rmi) * slopes[s]) + (
            0.5 * (rpl**2 - rmi**2)
        ) * slopes[s]

    for iu in range(len(n0s)):
        dQitot = 0.0
        for ju in range(len(n0s)):
            rG = radju * min(r0s[iu], r0s[ju])
            if rG <= 0.000111:
                Gr = 0.0271 * ((1000000.0 * rG) ** 2.7)
            else:
                Gr = 0.0988 * ((1000000.0 * rG) ** 0.98)
            velocity = abs(velocities[iu] - velocities[ju])
            delQ = ((velocity / 3.0) ** 2.5) * Gr * (10**-15)
            dQi = delQ * ns[ju] * np.pi * (r0s[iu] ** 2 + r0s[ju] ** 2) * (radju**2)
            if iu < ju:
                dQitot = dQitot + dQi
            else:
                dQitot = dQitot - dQi
        for k in range(len(ioncharges)):
            dQik = (
                ioncharges[k]
                * ionnumbers[k]
                * ionvelocities[k]
                * (np.pi * r0s[iu] ** 2)
            )
            dQitot = dQitot + dQik
        dQidt[iu] = dQitot * Qcoefff

    return dQidt


def dEdt(
    n0s: np.ndarray,
    slopes: np.ndarray,
    binbounds: np.ndarray,
    upperbound: np.ndarray,
    velocities: np.ndarray,
    charges: np.ndarray,
    Efield: float,
    mfptime: float = None,
    ioncharges: List[float] = [],
    ionnumbers: List[float] = [],
    ionvelocities: List[float] = [],
    ionmasses: List[float] = [],
    const: PhysicalConstants = CONST,
) -> float:
    """Calculate electric field rate - COPIED FROM ORIGINAL."""
    if mfptime is None:
        mfptime = const.mfptime

    r0s = np.zeros(len(n0s))
    ns = np.zeros(len(n0s))

    for s in range(len(n0s)):
        r0s[s] = (binbounds[s + 1] + binbounds[s]) / 2.0
        if n0s[s] + 0.5 * (binbounds[s] - binbounds[s + 1]) * slopes[s] <= 0:
            rpl = binbounds[s]
        elif n0s[s] + 0.5 * (binbounds[s + 1] - binbounds[s]) * slopes[s] >= 0:
            rpl = binbounds[s + 1]
        else:
            rpl = 0.5 * binbounds[s] + 0.5 * binbounds[s + 1] - n0s[s] / slopes[s]
        rmi = binbounds[s]
        ns[s] = (rpl - rmi) * (n0s[s] - 0.5 * (binbounds[s + 1] + rmi) * slopes[s]) + (
            0.5 * (rpl**2 - rmi**2)
        ) * slopes[s]

    Jc = 0.0
    Jd = 0.0
    for g in range(len(n0s)):
        Jcg = -ns[g] * velocities[g] * charges[g]
        Jc = Jc + Jcg
    for k in range(len(ionnumbers)):
        Jdk = (Efield * ionnumbers[k] * mfptime * (const.e_charge) ** 2) / ionmasses[k]
        Jd = Jd + Jdk

    return -(Jc + Jd) / (const.eps0)


def kayer(
    sim_params: SimulationParameters,
    const: PhysicalConstants = CONST,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    np.ndarray,
    np.ndarray,
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """
    Run simulation
    """
    anlT = 10.0 + 3.0 * (sim_params.plume_base_temp - 295.0) / 10.0
    fprea = (
        sim_params.base_humidity_fraction
        * const.epsilon
        * saturation_vapour_pressure(sim_params.plume_base_temp)
        / (
            sim_params.start_pressure
            - saturation_vapour_pressure(sim_params.plume_base_temp)
        )
    )
    frise = fprea / (1.0 + fprea)

    P = sim_params.start_pressure
    w = sim_params.start_upward_velocity
    Rplume = sim_params.plume_base_radius
    condensate = 0.0
    Trise = sim_params.plume_base_temp
    Tfall = sim_params.plume_base_temp + anlT
    stepmax = int(
        sim_params.start_pressure / sim_params.pressure_step - sim_params.pressure_step
    )

    Pressures: List[float] = []
    Tempsrise: List[float] = []
    Tempsfall: List[float] = []
    Tempsdiff: List[float] = []
    Velocities: List[float] = []
    fsrise: List[float] = []
    lsrise: List[float] = []
    Radii: List[float] = []

    binbounds = np.geomspace(
        sim_params.min_radius, sim_params.max_radius, sim_params.n_bins + 1
    )
    rhrro = const.rhoro ** (1.0 / 3.0)
    vrQQ = np.sqrt(
        (8.0 / (3.0 * const.Cdrag)) * const.rho_water * const.g * const.R / const.mu
    )
    vrel = np.ones([sim_params.n_bins, sim_params.n_bins]) * 10.0
    delt = 0.01
    upbsin = np.zeros(sim_params.n_bins)
    for s in range(len(binbounds) - 1):
        upbsin[s] = binbounds[s]
    upbsin[0] = binbounds[1]

    togglecondens = 0
    n0slist: List[np.ndarray] = []
    slopeslist: List[np.ndarray] = []
    Upbos: List[np.ndarray] = []
    NPrecipitations: List[np.ndarray] = []
    MPrecipitations: List[np.ndarray] = []
    n0s = np.zeros(sim_params.n_bins)
    slopes = np.zeros(sim_params.n_bins)
    upbs = upbsin + np.zeros(sim_params.n_bins)
    precipN = np.zeros(sim_params.n_bins)
    precipM = np.zeros(sim_params.n_bins)

    for i in range(stepmax):
        Pressures.append(P)
        Tempsrise.append(Trise)
        Tempsfall.append(Tfall)
        fsrise.append(frise)
        lsrise.append(condensate)
        Velocities.append(w)
        Tempsdiff.append(Trise - Tfall)
        Radii.append(Rplume)

        Pnew = P - sim_params.pressure_step
        fJrise = frise / (const.epsilon + frise * (1.0 - const.epsilon))
        muecurr = (1.0 - fJrise * (1.0 - const.epsilon)) * const.mu
        Cpcurr = 3.5 * const.R / muecurr
        Tfallnew = Tfall - sim_params.pressure_step * dry_adiabat(
            P, Tfall, 0.0, c_p=Cpcurr, const=const
        )

        if (P < 22632) or (Tfallnew < 216.6):
            fadjTf = sim_params.pressure_step / 100.0
            Tfallnew = (1 - fadjTf) * Tfallnew + fadjTf * temp_strat(Pnew)

        entrain_param = entrainment(Trise, P, Rplume, const)

        Trisenew = Trise - sim_params.pressure_step * moist_adiabat(
            P,
            Trise,
            Tfall,
            condensate,
            frise / (1.0 - frise),
            0.0,
            saturation_vapour_pressure(Trise),
            entrain_param,
            c_p=Cpcurr,
            const=const,
        )

        wnew = w - sim_params.pressure_step * upward_wind_gradient(
            P,
            Trise,
            Tfall,
            condensate,
            frise / (1.0 - frise),
            0.0,
            w,
            entrain_param,
            const=const,
        )

        frsnew = 0.6222 * (
            saturation_vapour_pressure(Trisenew)
            / (Pnew - saturation_vapour_pressure(Trisenew))
        )
        fsatnew = frsnew / (1.0 + frsnew)

        if w > 0.001:
            phinew = -0.2 * const.R * Trisenew / (Rplume * const.mu * Pnew * const.g)
            fracdelm = -phinew * sim_params.pressure_step
            frise = frise * (1.0 - fracdelm)
            condensate = condensate * (1.0 - fracdelm)
            n0s = n0s * (1.0 - fracdelm) * (Pnew / (Pnew + sim_params.pressure_step))
            slopes = (
                slopes * (1.0 - fracdelm) * (Pnew / (Pnew + sim_params.pressure_step))
            )

        if fsatnew < frise:
            fcondens = frise - fsatnew
            frisenew = fsatnew
            if w > 0.001:
                frdrho = (
                    -sim_params.pressure_step / Pnew - (Trisenew - Trise) / Trisenew
                ) + (
                    fcondens
                    * const.epsilon
                    * (1.0 - const.epsilon)
                    * const.mu
                    / ((const.epsilon + (1.0 - const.epsilon) * frisenew) ** 2)
                ) / muecurr
                frdRpl = 0.5 * fracdelm - 0.5 * frdrho
                Rplume = min(
                    Rplume * (1.0 + frdRpl), sim_params.plume_base_radius * np.sqrt(2.0)
                )
        else:
            fcondens = 0.0
            frisenew = frise
            if w > 0.001:
                frdrho = (
                    -sim_params.pressure_step - (Trisenew - Trise) * Pnew / Trisenew
                ) / Pnew
                frdRpl = 0.5 * fracdelm - 0.5 * frdrho
                Rplume = min(
                    Rplume * (1.0 + frdRpl), sim_params.plume_base_radius * np.sqrt(2.0)
                )

        if Trisenew > (const.temp_freeze - sim_params.temp_supercool):
            Eij = (
                np.ones([sim_params.n_bins, sim_params.n_bins])
                * sim_params.water_collision_efficiency
            )
            wjQQ = vrQQ * np.sqrt(Trise / P)
        else:
            Eij = (
                np.ones([sim_params.n_bins, sim_params.n_bins])
                * sim_params.ice_collision_efficiency
                * (rhrro**2)
            )
            wjQQ = vrQQ * np.sqrt(Trise / P) / (rhrro)

        for ie in range(sim_params.n_bins):
            for j in range(sim_params.n_bins):
                wiprea = wjQQ * np.sqrt(binbounds[max(ie, j) + 1])
                wjprea = wjQQ * np.sqrt(binbounds[min(ie, j)])
                vrel[ie, j] = abs(wiprea - wjprea)

        verticalrise = (
            sim_params.pressure_step * Trisenew * const.R / (const.g * Pnew * const.mu)
        )
        timefly = verticalrise / wnew
        stepsfly = int(np.ceil(timefly / delt))

        condensate_new_init = condensate + fcondens

        if condensate_new_init <= 0 or wnew <= 0:
            condensate_new = 0.0
            togglecondens = 0
            n0s = np.zeros(sim_params.n_bins)
            slopes = np.zeros(sim_params.n_bins)
            upbs = upbsin + np.zeros(sim_params.n_bins)
        else:
            if togglecondens == 0:
                togglecondens = 1
                n0sin = np.zeros(sim_params.n_bins)
                lz = condensate_new_init
                n0sin[0] = (
                    (lz / (1.0 - lz))
                    * (Pnew * const.mu)
                    / (
                        const.R
                        * Trisenew
                        * 1000.0
                        * ((4.0 / 3.0) * np.pi * 0.00001189207**3)
                    )
                    / (binbounds[1] - binbounds[0])
                )
                sizecrit = min((w / wjQQ) ** 2, binbounds[-1])
                slopes = np.zeros(sim_params.n_bins)
                n0s = n0sin + np.zeros(sim_params.n_bins)
                upbs = upbsin + np.zeros(sim_params.n_bins)

                for q in range(stepsfly):
                    n0s, slopes, upbs = stepgrow(
                        n0s, slopes, binbounds, upbs, const.rho_water, Eij, vrel, delt
                    )
            else:
                nar = (
                    (fcondens / (1.0 - fcondens))
                    * (Pnew * const.mu)
                    / (
                        const.R
                        * Trisenew
                        * 1000.0
                        * ((4.0 / 3.0) * np.pi * 0.00001189207**3)
                    )
                    / (binbounds[1] - binbounds[0])
                )
                n0s[0] = n0s[0] + nar
                sizecrit = min((w / wjQQ) ** 2, binbounds[-1])

                for q in range(stepsfly):
                    n0s, slopes, upbs = stepgrow(
                        n0s, slopes, binbounds, upbs, const.rho_water, Eij, vrel, delt
                    )

            binsed = max(
                int(np.floor(np.log(sizecrit / 0.00001) / np.log(np.sqrt(2.0)))), 0
            )

            for ssa in range(sim_params.n_bins):
                if (
                    n0s[ssa] + 0.5 * (binbounds[ssa] - binbounds[ssa + 1]) * slopes[ssa]
                    <= 0
                ):
                    upbs[ssa] = binbounds[ssa]
                elif (
                    n0s[ssa] + 0.5 * (binbounds[ssa + 1] - binbounds[ssa]) * slopes[ssa]
                    >= 0
                ):
                    upbs[ssa] = binbounds[ssa + 1]
                else:
                    upbs[ssa] = (
                        0.5 * binbounds[ssa]
                        + 0.5 * binbounds[ssa + 1]
                        - n0s[ssa] / slopes[ssa]
                    )

            mpvout = 0.0
            mpvin = 0.0
            precipN = np.zeros(sim_params.n_bins)
            precipM = np.zeros(sim_params.n_bins)

            for ffg in range(binsed, sim_params.n_bins, 1):
                rpl = upbs[ffg]
                rmi = binbounds[ffg]
                R4 = 0.25 * (rpl**4 - rmi**4)
                R5 = 0.20 * (rpl**5 - rmi**5)
                RR = (
                    R4 * (n0s[ffg] - 0.5 * (binbounds[ffg + 1] + rmi) * slopes[ffg])
                    + R5 * slopes[ffg]
                )
                mpvout = mpvout + RR
                precipN[ffg] = (rpl - rmi) * (
                    n0s[ffg] - 0.5 * (binbounds[ffg + 1] + rmi) * slopes[ffg]
                ) + (0.5 * (rpl**2 - rmi**2)) * slopes[ffg]
                precipM[ffg] = RR
                n0s[ffg] = 0.0
                slopes[ffg] = 0.0
                if ffg > 0:
                    upbs[ffg] = binbounds[ffg]

            for ffgh in range(0, binsed, 1):
                rpl = upbs[ffgh]
                rmi = binbounds[ffgh]
                R4 = 0.25 * (rpl**4 - rmi**4)
                R5 = 0.20 * (rpl**5 - rmi**5)
                RRa = (
                    R4 * (n0s[ffgh] - 0.5 * (binbounds[ffgh + 1] + rmi) * slopes[ffgh])
                    + R5 * slopes[ffgh]
                )
                mpvin = mpvin + RRa

            condensate_new = condensate_new_init * mpvin / (mpvout + mpvin)

        P = Pnew
        Trise = Trisenew
        Tfall = Tfallnew
        frise = frisenew
        condensate = condensate_new
        w = wnew
        if w < 0:
            w = 0.001

        n0slist.append(n0s)
        slopeslist.append(slopes)
        Upbos.append(upbs)
        NPrecipitations.append(precipN)
        MPrecipitations.append(precipM)

    # Final precipitation
    precipN = np.zeros(sim_params.n_bins)
    precipM = np.zeros(sim_params.n_bins)
    mpvout = 0.0
    for ffg in range(0, sim_params.n_bins, 1):
        rpl = upbs[ffg]
        rmi = binbounds[ffg]
        R4 = 0.25 * (rpl**4 - rmi**4)
        R5 = 0.20 * (rpl**5 - rmi**5)
        RR = (
            R4 * (n0s[ffg] - 0.5 * (binbounds[ffg + 1] + rmi) * slopes[ffg])
            + R5 * slopes[ffg]
        )
        mpvout = mpvout + RR
        precipN[ffg] = (rpl - rmi) * (
            n0s[ffg] - 0.5 * (binbounds[ffg + 1] + rmi) * slopes[ffg]
        ) + (0.5 * (rpl**2 - rmi**2)) * slopes[ffg]
        precipM[ffg] = RR

    NPrecipitations[-1] = NPrecipitations[-1] + precipN
    MPrecipitations[-1] = MPrecipitations[-1] + precipM

    # Calculate flash rates
    J1ss = np.zeros(990)
    tcrits = np.zeros(990)

    for ib in range(990):
        if (ib / 100.0) == np.ceil(ib / 100.0):
            print(ib)

        i = ib * 10
        n0s = n0slist[i]
        slopes = slopeslist[i]
        P = Pressures[i]
        T = Tempsrise[i]
        w = Velocities[i]
        velpart = np.zeros(sim_params.n_bins)

        if T > (const.temp_freeze - sim_params.temp_supercool):
            wjQQ = vrQQ * np.sqrt(T / P)
        else:
            wjQQ = vrQQ * np.sqrt(T / P) / (rhrro)

        for s in range(sim_params.n_bins):
            r0 = binbounds[s] * 1.1892
            velpart[s] = w - wjQQ * np.sqrt(r0)

        Ns = np.zeros(sim_params.n_bins)
        Ms = np.zeros(sim_params.n_bins)

        for f in range(sim_params.n_bins):
            if n0s[f] + 0.5 * (binbounds[f] - binbounds[f + 1]) * slopes[f] <= 0:
                rpl = binbounds[f]
            elif n0s[f] + 0.5 * (binbounds[f + 1] - binbounds[f]) * slopes[f] >= 0:
                rpl = binbounds[f + 1]
            else:
                rpl = 0.5 * binbounds[f] + 0.5 * binbounds[f + 1] - n0s[f] / slopes[f]
            rmi = binbounds[f]
            R4 = 0.25 * (rpl**4 - rmi**4)
            R5 = 0.20 * (rpl**5 - rmi**5)
            Ms[f] = (
                R4 * (n0s[f] - 0.5 * (binbounds[f + 1] + rmi) * slopes[f])
                + R5 * slopes[f]
            )
            Ns[f] = (rpl - rmi) * (
                n0s[f] - 0.5 * (binbounds[f + 1] + rmi) * slopes[f]
            ) + (0.5 * (rpl**2 - rmi**2)) * slopes[f]

        precipN = np.zeros(sim_params.n_bins)
        precipM = np.zeros(sim_params.n_bins)
        precipC = np.zeros(sim_params.n_bins)

        for fg in range(sim_params.n_bins):
            rD = (w / wjQQ) ** 2
            if rD >= binbounds[fg + 1]:
                precipC[fg] = 0.0
            else:
                blow = max(binbounds[fg], rD)
                bhigh = blow * np.sqrt(2.0)
                pCN = binbounds[fg + 1] - blow
                pCD = blow * (3 * w - 2 * wjQQ * np.sqrt(blow)) - bhigh * (
                    3 * w - 2 * wjQQ * np.sqrt(bhigh)
                )
                precipC[fg] = abs(3.0 * pCN / pCD)

        precipN = precipN + np.sum(
            (
                NPrecipitations[i:stepmax]
                * np.array(Velocities[i:stepmax])[:, None]
                * precipC
            ),
            axis=0,
        )
        precipM = precipM + np.sum(
            (
                MPrecipitations[i:stepmax]
                * np.array(Velocities[i:stepmax])[:, None]
                * precipC
            ),
            axis=0,
        )

        Ns = Ns + precipN
        Ms = Ms + precipM

        n0snew = np.zeros(sim_params.n_bins)
        slopesnew = np.zeros(sim_params.n_bins)
        upboss = np.zeros(sim_params.n_bins)

        for s in range(sim_params.n_bins):
            R2 = 0.5 * (binbounds[s + 1] ** 2 - binbounds[s] ** 2)
            R4 = 0.25 * (binbounds[s + 1] ** 4 - binbounds[s] ** 4)
            R5 = 0.2 * (binbounds[s + 1] ** 5 - binbounds[s] ** 5)
            r0 = 0.5 * (binbounds[s + 1] + binbounds[s])
            n0snewtest = Ns[s] / (binbounds[s + 1] - binbounds[s])
            slopesnewtest = ((binbounds[s + 1] - binbounds[s]) * Ms[s] - Ns[s] * R4) / (
                (binbounds[s + 1] - binbounds[s]) * R5 - R2 * R4
            )
            ncrit = n0snewtest + slopesnewtest * (binbounds[s + 1] - r0)

            if (Ns[s] <= 0) or (Ms[s] <= 0):
                upboss[s] = binbounds[s]
                slopesnew[s] = 0.0
                n0snew[s] = 0.0
            elif ncrit >= 0:
                n0snew[s] = n0snewtest
                slopesnew[s] = slopesnewtest
                upboss[s] = binbounds[s + 1]
            else:
                bquar = binbounds[s]
                Nquar = Ns[s]
                Mquar = Ms[s]
                uarray = np.roots(
                    [
                        1.0,
                        2.0 * bquar,
                        3.0 * (bquar**2),
                        4.0 * (bquar**3) - 10.0 * Mquar / Nquar,
                    ]
                )
                uquar = binbounds[s]
                for tt in range(len(uarray)):
                    uposs = uarray[tt]
                    if uposs.real >= binbounds[s] and uposs.real < binbounds[s + 1]:
                        if abs(uposs.imag) < 10**-10:
                            uquar = uposs.real + 0.0
                upboss[s] = uquar

                if upboss[s] > binbounds[s]:
                    R2 = 0.5 * (upboss[s] ** 2 - binbounds[s] ** 2)
                    R4 = 0.25 * (upboss[s] ** 4 - binbounds[s] ** 4)
                    R5 = 0.2 * (upboss[s] ** 5 - binbounds[s] ** 5)
                    n0snewerm = Ns[s] / (upboss[s] - binbounds[s])
                    slopesnew[s] = ((upboss[s] - binbounds[s]) * Ms[s] - Ns[s] * R4) / (
                        (upboss[s] - binbounds[s]) * R5 - R2 * R4
                    )
                    n0snew[s] = (
                        n0snewerm
                        + (r0 - (binbounds[s] + upboss[s]) / 2.0) * slopesnew[s]
                    )
                else:
                    slopesnew[s] = 0.0
                    n0snew[s] = 0.0

        if T > (const.temp_freeze - sim_params.temp_supercool):
            Qcoeff = 1.0 - sim_params.water_collision_efficiency
            radju = 1.0
        else:
            Qcoeff = 1.0 - sim_params.ice_collision_efficiency
            radju = rhrro

        kara = dQdt(
            n0snew, slopesnew, binbounds, upboss, velpart, Qcoefff=Qcoeff, radju=radju
        )
        qara = dEdt(
            n0snew, slopesnew, binbounds, upboss, velpart, kara, 0.0, const=const
        )

        J1ss[ib] = (const.eps0) * qara

        Emax = 3.0 * P
        if qara != 0:
            tcrits[ib] = np.sqrt(2.0 * Emax / abs(qara))

    PPV = 5.0 * np.array(Pressures)[np.arange(990) * 10] * J1ss * tcrits / 2.0
    verticalrise = (
        100.0
        * np.array(Tempsrise)
        * const.R
        / (const.g * np.array(Pressures) * const.mu)
    )
    flash_rate = abs(
        (10**6) * verticalrise[np.arange(989) * 10] * PPV[:-1] / const.Eflash
    )

    Plim = np.array(Pressures)[np.arange(989) * 10]

    return (
        Pressures,
        Velocities,
        Tempsrise,
        Plim,
        flash_rate,
        Tempsdiff,
        Radii,
        Tempsfall,
        fsrise,
        lsrise,
    )


def plot_comparison(results, param_name, output_dir):
    """Create comparison plots for two temperature scenarios."""
    # Define plot configurations as a dict of PlotConfigs

    @dataclass
    class _PlotConfig:
        """Configuration for a single plot."""

        data_idx: int
        ylabel: str
        title: str
        units: str

    plot_configs = {
        "velocity": _PlotConfig(
            data_idx=1,
            ylabel="Vertical velocity",
            title="Vertical Plume Velocity",
            units="m/s",
        ),
        "plume_temp": _PlotConfig(
            data_idx=2, ylabel="Temperature", title="Plume Temperature", units="K"
        ),
        "env_temp": _PlotConfig(
            data_idx=7, ylabel="Temperature", title="Environment Temperature", units="K"
        ),
        "temp_diff": _PlotConfig(
            data_idx=5,
            ylabel="Temperature difference",
            title="Plume-Environment Temp Difference",
            units="K",
        ),
        "radius": _PlotConfig(
            data_idx=6, ylabel="Radius", title="Plume Radius", units="m"
        ),
        "flash_rate": _PlotConfig(
            data_idx=4,
            ylabel="Flash rate",
            title="Lightning Flash Rate",
            units="flashes/s/km2",
        ),
    }

    # Create figure with mosaic layout using plot_configs keys
    mosaic = [list(plot_configs.keys())[:3], list(plot_configs.keys())[3:]]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    axes = fig.subplot_mosaic(mosaic)
    fig.suptitle(param_name, fontsize=14, fontweight="bold", y=1.075)

    # Plot each variable
    handles, labels = None, None
    for name, config in plot_configs.items():
        ax = axes[name]

        for run_label, data in results.items():
            # Convert pressure to bar
            if config.data_idx == 4:  # Flash rate uses different pressure array
                x = data[3] / 1e5  # Plim
            else:
                x = np.array(data[0]) / 1e5  # P

            y = data[config.data_idx]

            line = ax.plot(x, y, label=run_label, linewidth=1.5)

            # Capture handles and labels from the first subplot
            if handles is None:
                handles, labels = [], []
            if name == list(plot_configs.keys())[0]:
                handles.extend(line)
                labels.append(run_label)

        ax.set_xlabel("Pressure [bar]")
        ax.set_ylabel(f"{config.ylabel} ({config.units})")
        ax.set_title(config.title)
        ax.grid(True, alpha=0.3)

    # Add a single legend for the entire figure
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    filename = f"{param_name}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def main():
    """Main simulation runner."""

    const = PhysicalConstants()  # centralized constants instance

    base_temps = [280.0, 310.0]

    results = {}
    for base_temp in base_temps:
        sim_params = SimulationParameters(
            n_bins=31,
            plume_base_temp=base_temp,
            base_humidity_fraction=0.9,
            plume_base_radius=1000.0,
            temp_supercool=40.0,
            water_collision_efficiency=0.5,
            ice_collision_efficiency=0.0,
        )
        param_desc = (
            f"base_temp_{sim_params.plume_base_temp:.0f}__{sim_params.temp_supercool:.0f}__"
            f"{sim_params.water_collision_efficiency:.1f}__{sim_params.ice_collision_efficiency:.1f}"
        ).replace(".", "p")

        print(f"Simulating {base_temp:.0f} K...")
        run_label = f"base_temp_{base_temp:.0f}K"

        results[run_label] = kayer(sim_params, const)
        print(
            f"{run_label}: Total flash rate = {float(np.sum(results[run_label][4])) * 1.5e3:.2f} W/m2"
        )

    print("Generating plots...")
    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True, parents=True)
    plot_comparison(results, param_desc, outdir)


if __name__ == "__main__":
    main()
