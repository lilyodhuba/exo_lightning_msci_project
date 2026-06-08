"""1-D Lightning Model for an Earth-like Atmosphere."""

from typing import Tuple

import numpy as np

# Local modules
from constants import PhysicalConstants, SimulationParameters
from droplets import droplet_terminal_velocity, liquid_surface_tension
from iris.coords import DimCoord
from iris.cube import Cube, CubeList


def air_density(pressure: float, temperature: float, const: PhysicalConstants) -> float:
    """Calculate air density using the ideal gas law."""
    return (
        pressure
        * const.molar_mass_dry_air
        / (temperature * const.universal_gas_constant)
    )


def saturation_vapour_pressure(temp: float) -> float:
    """
    Calculate saturation vapour pressure of water using Lowe 1977 polynomial fit.

    The polynomial is of the form:
    :math:`e_s = (a_0 + T(a_1 + T(a_2 + T(a_3 + T(a_4 + T(a_5 + Ta_6))))))`

    where :math:`e_s` is saturation vapour pressure in Pa and T is temperature in K.

    Parameters
    ----------
    temp : float
        Temperature in Kelvin

    Returns
    -------
    float
        Saturation vapour pressure in Pascal (Pa)

    Notes
    -----
    Based on Lowe, P.R., 1977: An approximating polynomial for the computation
    of saturation vapor pressure. J. Appl. Meteor., 16, 100-103.

    The polynomial coefficients are:

    The result is converted from mb to Pa (x100) and cannot be negative.
    """
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


def temp_strat(
    sim_params: SimulationParameters,
    pressure: float, 
)-> float:
    initial_pressure = sim_params.start_pressure
    sigma_p = pressure / initial_pressure
    if sigma_p > (5474.9e-5):
        return 216.6
    elif sigma_p > (868.02e-5):
        return 216.6 + 12.0 * np.log(sigma_p / 5474.9e-5) / np.log(868.02e-5 / 5474.9e-5)
    elif sigma_p > (110.91e-5):
        return 228.6 + 42.0 * np.log(sigma_p / 868.02e-5) / np.log(110.91e-5 / 868.02e-5)
    elif sigma_p > (66.939e-5):
        return 270.6
    else:
        return 270.6 - 56.0 * np.log(sigma_p / 66.939e-5) / np.log(3.9564e-5 / 66.939e-5)

def dry_adiabat(
    P: float,
    T: float,
    f: float,
    c_p: float,
    const: PhysicalConstants,
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
    const : PhysicalConstants
        Container with physical constants

    Returns
    -------
    float
        Temperature gradient dT/dP [K/Pa]
    """
    return (
        const.universal_gas_constant
        * T
        * ((1 + f / const.epsilon) / (1 + f))
        / (const.molar_mass_dry_air * P * c_p)
    )


def entrainment(
    T: float,
    P: float,
    plume_radius: float,
    const: PhysicalConstants,
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
    plume_radius : float
        Radius of updraft [m]
    const : PhysicalConstants
        Container with physical constants

    Returns
    -------
    float
        Entrainment parameter (phi) [1/Pa]
    """
    return (
        -0.2
        * const.universal_gas_constant
        * T
        / (plume_radius * const.molar_mass_dry_air * P * const.gravity)
    )


def moist_adiabat(
    P: float,
    Trise: float,
    Tfall: float,
    lcondensate: float,
    frise: float,
    ffall: float,
    satvappre: float,
    entrain_param: float,
    c_p: float,
    const: PhysicalConstants,
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
    c_p : float
        Specific heat capacity at constant pressure [J/kg/K]
    const : PhysicalConstants
        Container with physical constants

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
            + const.latent_heat_v
            * fS
            * const.molar_mass_dry_air
            / (const.universal_gas_constant * Tv)
            - ((Trise - Tfall) * entrain_param / Gamma)
            - const.latent_heat_v * (fS - ffall) * entrain_param / (Gamma * c_p)
        )
        denom = 1 + (
            const.latent_heat_v**2 * fS * const.epsilon * const.molar_mass_dry_air
        ) / (c_p * const.universal_gas_constant * Trise * Trise)
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
    const: PhysicalConstants,
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
        Container with physical constants

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
        -const.universal_gas_constant
        * (
            Trise * (1 - lcondensate) * ((1 + frise / const.epsilon) / (1 + frise))
            - Tfall * ((1 + ffall / const.epsilon) / (1 + ffall))
        )
        / (P * const.molar_mass_dry_air * w)
        - w * entrain_param
    )
    return dwdPn


def calculate_rpl(
    n0s: np.ndarray,
    slopes: np.ndarray,
    binbounds: np.ndarray,
) -> np.ndarray:
    """
    Calculate rpl (radius upper limit) for particle distribution bins.

    This function determines the upper radius limit where the piecewise linear
    distribution becomes zero or reaches the bin boundary. For each bin, the
    distribution is n(r) = n0 + s(r - r0), where n0 is the density at bin
    center and s is the slope.

    Parameters
    ----------
    n0s : np.ndarray
        Number density at bin centers [1/m^4]
    slopes : np.ndarray
        Slopes of linear distribution in each bin [1/m^5]
    binbounds : np.ndarray
        Boundaries of particle size bins [m], length n_bins + 1

    Returns
    -------
    np.ndarray
        Upper radius limits for each bin [m]

    Notes
    -----
    The function handles three cases:
    1. If n(r) <= 0 at the lower bound, rpl = lower bound
    2. If n(r) >= 0 at the upper bound, rpl = upper bound
    3. Otherwise, rpl is calculated using slopes and n0
    """
    # Vectorized conditions for rpl calculation
    cond1 = n0s + 0.5 * (binbounds[:-1] - binbounds[1:]) * slopes <= 0
    cond2 = n0s + 0.5 * (binbounds[1:] - binbounds[:-1]) * slopes >= 0
    other_values = 0.5 * (binbounds[:-1] + binbounds[1:]) - np.divide(
        n0s, slopes, out=np.zeros_like(n0s), where=slopes != 0.0
    )

    rpl = np.where(cond1, binbounds[:-1], np.where(cond2, binbounds[1:], other_values))
    return rpl


def stepgrow(
    sim_params: SimulationParameters,
    n0s: np.ndarray,
    slopes: np.ndarray,
    binbounds: np.ndarray,
    upbooms: np.ndarray,
    rho: float,
    Eij: np.ndarray,
    vrel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Simulate one time step of particle growth through collisional coalescence.

    This function advances the particle size distribution by one time step, accounting
    for collisions between particles in different size bins. Particles collide based on
    their relative velocities and collision efficiencies, causing smaller particles to
    be captured by larger ones and grow to larger size bins.

    The collision rate between particles in bins i and j is:

    .. math::
        \lambda_{ij} = \min\left(E_{ij} \pi (r_i^2 + r_j^2) v_{rel,ij} N_j \Delta t, 1\right)

    where :math:`E_{ij}` is the collision efficiency, :math:`r_i` and :math:`r_j` are
    characteristic radii, :math:`v_{rel,ij}` is the relative velocity, :math:`N_j` is
    the number of particles in bin j, and :math:`\Delta t` is the time step.

    The number and mass changes for bin j (smaller particle being captured):

    .. math::
        \Delta N_j = -\lambda_{ij} N_i

    .. math::
        \Delta M_j = -\lambda_{ij} N_i m_j

    The mass change for bin i (larger particle growing):

    .. math::
        \Delta M_i = +\lambda_{ij} N_i m_j

    After collisions, particles may move to adjacent bins if their mass increases beyond
    the bin boundaries. The new distribution is represented using a piecewise linear form:

    .. math::
        n(r) = n_0 + s(r - r_0)

    where :math:`n_0` is the density at bin center :math:`r_0` and :math:`s` is the slope.

    Parameters
    ----------
    sim_params : SimulationParameters
        Simulation configuration containing n_bins and other parameters
    n0s : np.ndarray
        Number density at bin centers [1/m^4]
    slopes : np.ndarray
        Slopes of linear distribution in each bin [1/m^5]
    binbounds : np.ndarray
        Boundaries of particle size bins [m], length n_bins + 1
    upbooms : np.ndarray
        Upper bounds of particle distribution in each bin [m]
    rho : float
        Particle density (water or ice) [kg/m^3]
    Eij : np.ndarray
        Collision efficiency matrix [dimensionless], shape (n_bins, n_bins)
    vrel : np.ndarray
        Relative velocity matrix [m/s], shape (n_bins, n_bins)

    Returns
    -------
    n0snew : np.ndarray
        Updated number density at bin centers [1/m^4]
    slopesnew : np.ndarray
        Updated slopes of linear distribution [1/m^5]
    upbos : np.ndarray
        Updated upper bounds of distribution [m]

    Notes
    -----
    The function uses a semi-Lagrangian approach where the particle size distribution
    is represented as a piecewise linear function within each bin. This allows for
    sub-bin resolution of the distribution while maintaining computational efficiency.

    The collision algorithm considers all pairs (i, j) where i >= j, processing only
    the lower triangle of the collision matrix to avoid double counting. When particles
    in bin i collect particles from bin j, the resulting mass may exceed the upper
    boundary of bin i, causing a transfer to bin i+1.

    The piecewise linear representation is reconstructed after collisions by solving
    for n0 and slope values that match the total number and mass in each bin while
    maintaining physical constraints (non-negative number densities).
    """
    r0s = np.zeros(sim_params.n_bins)
    Ns = np.zeros(sim_params.n_bins)
    Ms = np.zeros(sim_params.n_bins)
    n0snew = np.zeros(sim_params.n_bins)
    slopesnew = np.zeros(sim_params.n_bins)
    Nsnew = np.zeros(sim_params.n_bins)
    Msnew = np.zeros(sim_params.n_bins)
    mmean = np.zeros(sim_params.n_bins)
    upbos = np.zeros(sim_params.n_bins)

    # Calculate r0s array in one operation
    r0s = (binbounds[1:] + binbounds[:-1]) / 2.0

    # Vectorized upper bounds calculation
    upbos = calculate_rpl(n0s, slopes, binbounds)

    # Vectorized masks for different conditions
    mask_full = upbos >= binbounds[1:]
    mask_partial = (upbos > binbounds[:-1]) & ~mask_full

    # Initialize arrays
    Ns = np.zeros_like(n0s)
    Ms = np.zeros_like(n0s)
    mmean = np.zeros_like(n0s)

    # Full bins calculation
    Ns[mask_full] = n0s[mask_full] * (
        binbounds[1:][mask_full] - binbounds[:-1][mask_full]
    )
    Ms[mask_full] = (4 * np.pi * rho / 3.0) * (
        0.2
        * (binbounds[1:][mask_full] ** 5 - binbounds[:-1][mask_full] ** 5)
        * slopes[mask_full]
        + 0.25
        * (binbounds[1:][mask_full] ** 4 - binbounds[:-1][mask_full] ** 4)
        * (n0s[mask_full] - slopes[mask_full] * r0s[mask_full])
    )

    # Partial bins calculation
    Ns[mask_partial] = n0s[mask_partial] * (
        upbos[mask_partial] - binbounds[:-1][mask_partial]
    ) + slopes[mask_partial] * (upbos[mask_partial] - binbounds[:-1][mask_partial]) * (
        upbos[mask_partial] / 2.0
        + binbounds[:-1][mask_partial] / 2.0
        - r0s[mask_partial]
    )

    Ms[mask_partial] = (4 * np.pi * rho / 3.0) * (
        0.2
        * (upbos[mask_partial] ** 5 - binbounds[:-1][mask_partial] ** 5)
        * slopes[mask_partial]
        + 0.25
        * (upbos[mask_partial] ** 4 - binbounds[:-1][mask_partial] ** 4)
        * (n0s[mask_partial] - slopes[mask_partial] * r0s[mask_partial])
    )

    # Empty bins get default values
    mask_empty = ~(mask_full | mask_partial)
    mmean[mask_empty] = (np.pi * rho / 3.0) * (
        binbounds[1:][mask_empty] ** 4 - binbounds[:-1][mask_empty] ** 4
    )

    # Calculate mean mass for non-empty bins
    mask_nonempty = ~mask_empty
    mmean[mask_nonempty] = Ms[mask_nonempty] / Ns[mask_nonempty]

    # Assignment of Ns and Ms to Nsnew and Msnew
    Nsnew = Ns.copy()
    Msnew = Ms.copy()

    # Create arrays for i,j combinations
    i_idx, j_idx = np.tril_indices(sim_params.n_bins)

    # Calculate lambdaij matrix
    lambdaij = np.minimum(
        Eij[i_idx, j_idx]
        * np.pi
        * (r0s[i_idx] ** 2 + r0s[j_idx] ** 2)
        * vrel[i_idx, j_idx]
        * Ns[j_idx]
        * sim_params.dt,
        1.0,
    )

    # Update Nsnew and Msnew for j indices
    np.add.at(Nsnew, j_idx, -lambdaij * Ns[i_idx])
    np.add.at(Msnew, j_idx, -lambdaij * Ns[i_idx] * mmean[j_idx])

    # Update Msnew for i indices
    np.add.at(Msnew, i_idx, lambdaij * Ns[i_idx] * mmean[j_idx])

    # Calculate rx array
    rx = (binbounds[i_idx + 1] ** 3 - r0s[j_idx] ** 3) ** (1 / 3.0)

    # Calculate conditions for rxx
    full_bin = upbos[i_idx] >= binbounds[i_idx + 1]
    partial_bin = upbos[i_idx] > rx

    # Initialize rxx, Nxx, Mxx arrays
    rxx = np.zeros_like(rx)
    rxx[full_bin] = binbounds[i_idx[full_bin] + 1]
    rxx[partial_bin] = upbos[i_idx[partial_bin]]

    # Calculate Nxx and Mxx where applicable
    valid_bins = full_bin | partial_bin
    Nxx = np.zeros_like(rx)
    Mxx = np.zeros_like(rx)

    # Update for valid bins
    if np.any(valid_bins):
        Nxx[valid_bins] = (
            n0s[i_idx[valid_bins]] * (rxx[valid_bins] - rx[valid_bins])
            - slopes[i_idx[valid_bins]]
            * r0s[i_idx[valid_bins]]
            * (rxx[valid_bins] - rx[valid_bins])
            + slopes[i_idx[valid_bins]]
            * (rxx[valid_bins] - rx[valid_bins])
            * (rxx[valid_bins] + rx[valid_bins])
            / 2.0
        )

        Mxx[valid_bins] = (np.pi * rho / 3.0) * (
            rxx[valid_bins] ** 4 - rx[valid_bins] ** 4
        ) * (
            n0s[i_idx[valid_bins]] - slopes[i_idx[valid_bins]] * r0s[i_idx[valid_bins]]
        ) + (4 * np.pi * rho / 15.0) * slopes[i_idx[valid_bins]] * (
            rxx[valid_bins] ** 5 - rx[valid_bins] ** 5
        )

    # Update next bin values where i+1 exists
    next_bin_mask = i_idx + 1 < sim_params.n_bins
    if np.any(next_bin_mask):
        i_next = i_idx[next_bin_mask]
        lambda_next = lambdaij[next_bin_mask]
        Nxx_next = Nxx[next_bin_mask]
        Mxx_next = Mxx[next_bin_mask]
        mmean_j = mmean[j_idx[next_bin_mask]]

        np.add.at(
            Msnew, i_next + 1, Mxx_next * lambda_next + mmean_j * Nxx_next * lambda_next
        )
        np.add.at(
            Msnew, i_next, -(Mxx_next * lambda_next + mmean_j * Nxx_next * lambda_next)
        )
        np.add.at(Nsnew, i_next + 1, Nxx_next * lambda_next)
        np.add.at(Nsnew, i_next, -Nxx_next * lambda_next)

    # Pre-compute arrays for all bins at once
    R2 = 0.5 * (binbounds[1:] ** 2 - binbounds[:-1] ** 2)
    R4 = 0.25 * (binbounds[1:] ** 4 - binbounds[:-1] ** 4)
    R5 = 0.2 * (binbounds[1:] ** 5 - binbounds[:-1] ** 5)

    # Initialize output arrays
    n0snew = np.zeros(sim_params.n_bins)
    slopesnew = np.zeros(sim_params.n_bins)
    upbos = np.copy(binbounds[:-1])  # Default to lower bounds

    # Calculate initial test values
    bin_widths = binbounds[1:] - binbounds[:-1]
    n0snewtest = Nsnew / bin_widths
    slopesnewtest = (3 * bin_widths * Msnew / (4 * rho * np.pi) - Nsnew * R4) / (
        bin_widths * R5 - R2 * R4
    )
    ncrit = n0snewtest + slopesnewtest * (binbounds[1:] - r0s)

    # Handle simple cases with vectorized operations
    valid_bins = (Nsnew > 0) & (Msnew > 0)
    positive_ncrit = valid_bins & (ncrit >= 0)

    # Set values for positive ncrit cases
    n0snew[positive_ncrit] = n0snewtest[positive_ncrit]
    slopesnew[positive_ncrit] = slopesnewtest[positive_ncrit]
    upbos[positive_ncrit] = binbounds[1:][positive_ncrit]

    # Handle complex cases that need root finding
    complex_cases = valid_bins & ~positive_ncrit
    if np.any(complex_cases):
        for s in np.where(complex_cases)[0]:
            bquar = binbounds[s]
            Nquar = Nsnew[s]
            Mquar = 3.0 * Msnew[s] / (4.0 * rho * np.pi)

            # Find roots
            uarray = np.roots(
                [
                    1.0,
                    2.0 * bquar,
                    3.0 * bquar**2,
                    4.0 * bquar**3 - 10.0 * Mquar / Nquar,
                ]
            )

            # Find valid root in range
            valid_roots = uarray[
                (uarray.real >= binbounds[s])
                & (uarray.real < binbounds[s + 1])
                & (abs(uarray.imag) < 1e-10)
            ]

            if len(valid_roots) > 0:
                uquar = float(valid_roots[0].real)
                upbos[s] = uquar

                if uquar > binbounds[s]:
                    R4_s = 0.25 * (uquar**4 - binbounds[s] ** 4)
                    n0snewerm = Nsnew[s] / (uquar - binbounds[s])

                    # Optimized slope calculation
                    numer = (
                        (3 * (uquar - binbounds[s]) * Msnew[s]) / (4 * rho * np.pi)
                    ) - Nsnew[s] * R4_s
                    denom = (
                        3.0 * uquar**2
                        + 4.0 * uquar * binbounds[s]
                        + 3.0 * binbounds[s] ** 2
                    ) * (uquar - binbounds[s]) ** 4
                    slopesnew[s] = 40.0 * numer / denom

                    n0snew[s] = (
                        n0snewerm
                        + (r0s[s] - (binbounds[s] + uquar) / 2.0) * slopesnew[s]
                    )

    return n0snew, slopesnew, upbos


def charge_transfer(
    n0s: np.ndarray,
    slopes: np.ndarray,
    binbounds: np.ndarray,
    upperbound: np.ndarray,
    velocities: np.ndarray,
    ioncharges: np.ndarray = None,
    ionnumbers: np.ndarray = None,
    ionvelocities: np.ndarray = None,
    Qcoefff: float = 1.0,
    radju: float = 1.0,
) -> np.ndarray:
    r"""
    Calculate the rate of charge transfer between particle bins.

    This function computes the charging rate :math:`dQ/dt` for each particle size bin
    due to collisions with other particles and optionally with ions. The charge transfer
    depends on particle sizes, relative velocities, and collision geometries.

    The charge transfer rate for bin i is given by:

    .. math::
        \frac{dQ_i}{dt} = Q_{coeff} \sum_{j \neq i} \Delta Q_{ij} n_j \pi (r_i^2 + r_j^2) r_{adj}^2

    where :math:`\Delta Q_{ij}` is the charge exchanged per collision between bins i and j:

    .. math::
        \Delta Q_{ij} = \left(\frac{|v_i - v_j|}{3}\right)^{2.5} G_r(r_G) \times 10^{-15}

    The geometric factor :math:`r_G` is:

    .. math::
        r_G = r_{adj} \min(r_i, r_j)

    and :math:`G_r` is an empirical function:

    .. math::
        G_r(r_G) = \begin{cases}
        0.0271 (10^6 r_G)^{2.7} & \text{if } r_G \leq 0.000111 \text{ m} \\
        0.0988 (10^6 r_G)^{0.98} & \text{if } r_G > 0.000111 \text{ m}
        \end{cases}

    If ions are present, an additional contribution is added:

    .. math::
        \frac{dQ_i}{dt}|_{ions} = \sum_{k} q_k N_k v_k \pi r_i^2

    where :math:`q_k`, :math:`N_k`, and :math:`v_k` are the charge, number density,
    and velocity of ion species k.

    Parameters
    ----------
    n0s : np.ndarray
        Number density at bin centers [1/m^4]
    slopes : np.ndarray
        Slopes of linear distribution in each bin [1/m^5]
    binbounds : np.ndarray
        Boundaries of particle size bins [m]
    upperbound : np.ndarray
        Upper bounds of particle distribution in each bin [m]
    velocities : np.ndarray
        Drift velocities of particles in each bin [m/s]
    ioncharges : np.ndarray, optional
        Charges of ion species [C]
    ionnumbers : np.ndarray, optional
        Number densities of ion species [1/m^3]
    ionvelocities : np.ndarray, optional
        Drift velocities of ion species [m/s]
    Qcoefff : float, optional
        Global charge transfer coefficient (default 1.0) [dimensionless]
    radju : float, optional
        Radius adjustment factor for ice particles (default 1.0) [dimensionless]

    Returns
    -------
    np.ndarray
        Rate of charge change for each bin [C/s]

    Notes
    -----
    The charge transfer mechanism is based on collisional charging where particles
    of different sizes moving at different velocities exchange charge. The empirical
    functions for :math:`G_r` and the velocity dependence are derived from
    laboratory measurements of particle-particle collisions.

    The function uses a piecewise linear representation of the particle size distribution
    within each bin, with n0s representing the density at the bin center and slopes
    giving the linear variation with radius.
    """
    # Calculate r0s array
    r0s = (binbounds[1:] + binbounds[:-1]) / 2.0

    # Calculate ns array using vectorized operations
    rpl = calculate_rpl(n0s, slopes, binbounds)

    rmi = binbounds[:-1]

    ns = (rpl - rmi) * (n0s - 0.5 * (binbounds[1:] + rmi) * slopes) + 0.5 * (
        rpl**2 - rmi**2
    ) * slopes

    # Calculate all pairwise radius and velocity differences at once
    rG = radju * np.minimum.outer(r0s, r0s)
    vel_diff = np.abs(velocities[:, np.newaxis] - velocities)

    # Vectorized Gr calculation
    mask_small = rG <= 0.000111
    Gr = np.zeros_like(rG)
    Gr[mask_small] = 0.0271 * ((1e6 * rG[mask_small]) ** 2.7)
    Gr[~mask_small] = 0.0988 * ((1e6 * rG[~mask_small]) ** 0.98)

    # Calculate delQ matrix
    delQ = ((vel_diff / 3.0) ** 2.5) * Gr * 1e-15

    # Calculate geometric factor
    geom_factor = np.pi * (r0s[:, np.newaxis] ** 2 + r0s**2) * (radju**2)

    # Calculate charge transfer matrix
    charge_matrix = delQ * ns * geom_factor

    # Sum up contributions (upper triangle minus lower triangle)
    dQidt = np.sum(np.triu(charge_matrix, 1) - np.tril(charge_matrix), axis=1)

    # Add ion contributions if any
    if ioncharges:
        ion_contributions = np.sum(
            ioncharges * ionnumbers * ionvelocities * np.pi * r0s**2, axis=0
        )
        dQidt += ion_contributions

    return dQidt * Qcoefff


def electric_field_change(
    n0s,
    slopes,
    binbounds,
    upperbound,
    velocities,
    charges,
    ioncharges=None,
    ionnumbers=None,
    ionvelocities=None,
    ionmasses=None,
    electric_field_strength=0.0,
    const=None,
):
    """
    Calculate the rate of change of the electric field.

    This function computes dE/dt based on the continuity equation and current densities
    from both charged particles and ions (if present).

    Parameters
    ----------
    n0s : np.ndarray
        Initial number densities at bin centers.
    slopes : np.ndarray
        Slopes of the linear approximation in each bin.
    binbounds : np.ndarray
        Boundaries of the bins for particle size distribution.
    upperbound : np.ndarray
        Upper boundary values for the distribution.
    velocities : np.ndarray
        Drift velocities of charged particles.
    charges : np.ndarray
        Charge values for particles.
    ioncharges : np.ndarray, optional
        Charges of ion species.
    ionnumbers : np.ndarray, optional
        Number densities of ion species.
    ionvelocities : np.ndarray, optional
        Drift velocities of ion species.
    ionmasses : np.ndarray, optional
        Masses of ion species.
    electric_field_strength : float, optional
        External electric field strength, defaults to 0.0.
    const : PhysicalConstants, optional
        Container with physical constants

    Returns
    -------
    float
        Rate of change of electric field (dE/dt).

    Notes
    -----
    The electric field rate is calculated using:

    :math:`dE/dt = -(J_c + J_d)/epsilon_0`

    where:
    - :math:`J_c` is the conduction current density from charged particles
    - :math:`J_d` is the displacement current density from ions
    - :math:`epsilon_0` is the vacuum permittivity

    The conduction current is computed as:
    :math:`J_c = -sum_i n_i v_i q_i`

    For ions, the displacement current is:
    :math:`J_d = sum_i E cdot n_i tau e^2/m_i`

    where:
    - :math:`tau` is the mean free path time
    - :math:`e` is the elementary charge
    - :math:`m_i` is the ion mass
    """
    # Calculate rpl using helper function
    rpl = calculate_rpl(n0s, slopes, binbounds)

    # Get starting bounds
    rmi = binbounds[:-1]

    # Vectorized ns calculation
    ns = (rpl - rmi) * (n0s - 0.5 * (binbounds[1:] + rmi) * slopes) + (
        0.5 * (rpl**2 - rmi**2)
    ) * slopes

    # Vectorized current calculation
    Jc = -np.sum(ns * velocities * charges)

    # Ion current if ions present
    if ionmasses:
        Jd = np.sum(
            electric_field_strength
            * ionnumbers
            * const.mean_free_path_ion_coll
            * (const.e_charge) ** 2
            / ionmasses
        )
    else:
        Jd = 0.0

    return -(Jc + Jd) / const.vacuum_perm


def run_sim(sim_params: SimulationParameters, const: PhysicalConstants) -> dict:
    """
    Run a complete lightning plume simulation for a convective updraft.

    This function simulates the vertical evolution of a convective plume from surface
    to upper atmosphere, tracking thermodynamic properties, particle growth through
    collisional coalescence, charge separation, and lightning flash rates. The simulation
    uses a one-dimensional model with entrainment, moist adiabatic processes, and
    microphysical interactions between particles in logarithmically-spaced size bins.

    The model advances through pressure levels, computing at each step:
    1. Thermodynamic evolution (temperature, humidity, condensation)
    2. Dynamic evolution (vertical velocity, plume radius, entrainment)
    3. Particle growth via collisional coalescence
    4. Sedimentation and precipitation
    5. Charge separation and electric field buildup
    6. Lightning flash rate estimation

    Parameters
    ----------
    sim_params : SimulationParameters
        Simulation configuration (see the YAML files in the config/ folder)
    const : PhysicalConstants
        Container with physical constants (see the YAML files in the config/ folder)

    Returns
    -------
    CubeList
        Iris CubeList containing selected diagnostics w.r.t. pressure.

    Notes
    -----
    The simulation proceeds upward through the atmosphere in discrete pressure steps,
    starting at `start_pressure` and decreasing by `pressure_step` each iteration until
    reaching near-zero pressure or the velocity becomes negative.

    Particle size distribution is represented using a piecewise linear approach within
    logarithmically-spaced bins. Each bin is characterized by a center density (n0s)
    and slope, allowing sub-bin resolution of the distribution.

    Charge separation follows an empirical parameterization based on particle-particle
    collisions with relative velocities. The charging rate depends on particle sizes,
    collision efficiencies, and differential sedimentation velocities.

    Flash rate is calculated from the electric field growth rate and critical breakdown
    field strength (3P, where P is pressure in Pascals). The flash rate represents the
    energy dissipation rate divided by typical lightning flash energy (~1.5 GJ).

    The simulation uses vectorized numpy operations throughout for computational
    efficiency, particularly in the particle growth, charge transfer, and electric
    field calculations.
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
    rhrro = const.rhoro ** (1.0 / 3.0)

    Pressures = np.zeros(stepmax)
    Tempsrise = np.zeros(stepmax)
    Tempsfall = np.zeros(stepmax)
    Velocities = np.zeros(stepmax)
    fsrise = np.zeros(stepmax)
    lsrise = np.zeros(stepmax)
    Radii = np.zeros(stepmax)

    binbounds = np.geomspace(
        sim_params.min_radius, sim_params.max_radius, sim_params.n_bins + 1
    )
    if sim_params.method_terminal_velocity == "aglyamov21":
        vrQQ = np.sqrt(
            (8.0 / (3.0 * const.drag_coef))
            * const.rho_water
            * const.gravity
            * const.universal_gas_constant
            / const.molar_mass_dry_air
        )
    elif sim_params.method_terminal_velocity == "loftus21":
        r0s = (binbounds[1:] + binbounds[:-1]) / 2.0
        wjQQ = np.zeros(sim_params.n_bins)

    # Initialize upbsin more efficiently
    upbsin = np.full(sim_params.n_bins, binbounds[0])
    upbsin[0] = binbounds[1]

    togglecondens = 0

    # Pre-allocate arrays efficiently using zeros
    n0s_per_level = np.zeros((stepmax, sim_params.n_bins))
    slopes_per_level = np.zeros((stepmax, sim_params.n_bins))
    upper_bound_per_level = np.zeros((stepmax, sim_params.n_bins))
    n_precip_per_level = np.zeros((stepmax, sim_params.n_bins))
    m_precip_per_level = np.zeros((stepmax, sim_params.n_bins))

    # Initialize state arrays
    n0s = np.zeros(sim_params.n_bins)
    slopes = np.zeros(sim_params.n_bins)
    upbs = upbsin.copy()
    precipN = np.zeros(sim_params.n_bins)
    precipM = np.zeros(sim_params.n_bins)

    # Pre-allocate output arrays
    Pressures = np.zeros(stepmax)
    Tempsrise = np.zeros(stepmax)
    Tempsfall = np.zeros(stepmax)
    fsrise = np.zeros(stepmax)
    lsrise = np.zeros(stepmax)
    Velocities = np.zeros(stepmax)
    Radii = np.zeros(stepmax)

    for i in range(stepmax):
        # Store current state
        Pressures[i] = P
        Tempsrise[i] = Trise
        Tempsfall[i] = Tfall
        fsrise[i] = frise
        lsrise[i] = condensate
        Velocities[i] = w
        Radii[i] = Rplume
        Pnew = P - sim_params.pressure_step
        start_pressure = sim_params.start_pressure

        # Vectorized thermodynamic calculations
        fJrise = frise / (const.epsilon + frise * (1.0 - const.epsilon))
        muecurr = (1.0 - fJrise * (1.0 - const.epsilon)) * const.molar_mass_dry_air
        Cpcurr = 3.5 * const.universal_gas_constant / muecurr

        Tfallnew = Tfall - sim_params.pressure_step * dry_adiabat(
            P, Tfall, 0.0, c_p=Cpcurr, const=const
        )

        # Temperature stratification adjustment
        sigP = P / sim_params.start_pressure
        if (sigP < 22632e-5) or (Tfallnew < 216.6):
            fadjTf = sim_params.pressure_step / 100.0
            Tfallnew = (1 - fadjTf) * Tfallnew + fadjTf * temp_strat(sim_params, Pnew)

        entrain_param = entrainment(Trise, P, Rplume, const)

        # Calculate new temperatures and velocities
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

        # Vectorized saturation calculations
        sat_vapor_press = saturation_vapour_pressure(Trisenew)
        frsnew = const.epsilon * sat_vapor_press / (Pnew - sat_vapor_press)
        fsatnew = frsnew / (1.0 + frsnew)

        if w > sim_params.start_upward_velocity:
            entrain_param = entrainment(Trisenew, Pnew, Rplume, const)
            fracdelm = -entrain_param * sim_params.pressure_step

            # Vectorized updates
            frise *= 1.0 - fracdelm
            condensate *= 1.0 - fracdelm
            pressure_ratio = Pnew / (Pnew + sim_params.pressure_step)
            scale_factor = (1.0 - fracdelm) * pressure_ratio
            n0s *= scale_factor
            slopes *= scale_factor

        # Condensation calculations
        if fsatnew < frise:
            fcondens = frise - fsatnew
            frisenew = fsatnew
            if w > sim_params.start_upward_velocity:
                frdrho = (
                    -sim_params.pressure_step / Pnew - (Trisenew - Trise) / Trisenew
                ) + (
                    fcondens
                    * const.epsilon
                    * (1.0 - const.epsilon)
                    * const.molar_mass_dry_air
                    / ((const.epsilon + (1.0 - const.epsilon) * frisenew) ** 2)
                ) / muecurr
                frdRpl = 0.5 * fracdelm - 0.5 * frdrho
                Rplume = min(
                    Rplume * (1.0 + frdRpl), sim_params.plume_base_radius * np.sqrt(2.0)
                )
        else:
            fcondens = 0.0
            frisenew = frise
            if w > sim_params.start_upward_velocity:
                frdrho = (
                    -sim_params.pressure_step - (Trisenew - Trise) * Pnew / Trisenew
                ) / Pnew
                frdRpl = 0.5 * fracdelm - 0.5 * frdrho
                Rplume = min(
                    Rplume * (1.0 + frdRpl), sim_params.plume_base_radius * np.sqrt(2.0)
                )

        # Vectorized collision efficiency matrix creation
        Eij = np.full(
            (sim_params.n_bins, sim_params.n_bins),
            sim_params.water_collision_efficiency
            if Trisenew > (const.temp_freeze - sim_params.temp_supercool)
            else sim_params.ice_collision_efficiency * (rhrro**2),
        )

        # Terminal velocity calculation
        if sim_params.method_terminal_velocity == "aglyamov21":
            wjQQ = (
                vrQQ
                * np.sqrt(Trise / P)
                / (
                    1.0
                    if Trisenew > (const.temp_freeze - sim_params.temp_supercool)
                    else rhrro
                )
            )
        elif sim_params.method_terminal_velocity == "loftus21":
            air_dens = air_density(P, Trise, const=const)
            surf_tens = liquid_surface_tension(Trise)
            for i_bin, i_radius in enumerate(r0s):
                wjQQ[i_bin] = (
                    droplet_terminal_velocity(
                        i_radius,
                        air_dens=air_dens,
                        liq_dens=const.rho_water,
                        surf_tens=surf_tens,
                        gravity=const.gravity,
                    )
                    / i_radius**0.5
                )

        # Create meshgrid of bin indices
        ie, j = np.meshgrid(np.arange(sim_params.n_bins), np.arange(sim_params.n_bins))

        # Calculate maximum and minimum bin bounds indices
        max_idx = np.maximum(ie, j) + 1
        min_idx = np.minimum(ie, j)

        # Compute velocity differences in one vectorized operation
        vrel = wjQQ * np.abs(np.sqrt(binbounds[max_idx]) - np.sqrt(binbounds[min_idx]))

        # Calculate vertical rise and time parameters
        verticalrise = (
            sim_params.pressure_step
            * Trisenew
            * const.universal_gas_constant
            / (const.gravity * Pnew * const.molar_mass_dry_air)
        )
        timefly = verticalrise / wnew
        stepsfly = int(np.ceil(timefly / sim_params.dt))

        # Initialize condensate
        condensate_new_init = condensate + fcondens

        # Handle zero or negative condensate/velocity cases
        if condensate_new_init <= 0 or wnew <= 0:
            condensate_new = 0.0
            togglecondens = 0
            n0s = np.zeros(sim_params.n_bins)
            slopes = np.zeros(sim_params.n_bins)
            upbs = np.copy(upbsin)
        else:
            # Common calculations
            base_density_factor = (
                (Pnew * const.molar_mass_dry_air)
                / (
                    const.universal_gas_constant
                    * Trisenew
                    * 1000.0
                    * ((4.0 / 3.0) * np.pi * 0.00001189207**3)
                )
            ) / (binbounds[1] - binbounds[0])

            if togglecondens == 0:
                # Initialize new condensation
                togglecondens = 1
                n0s = np.zeros(sim_params.n_bins)
                slopes = np.zeros(sim_params.n_bins)

                # Calculate initial number density for first bin
                lz = condensate_new_init
                n0s[0] = (lz / (1.0 - lz)) * base_density_factor

            else:
                # Add new condensation to existing distribution
                nar = (fcondens / (1.0 - fcondens)) * base_density_factor
                n0s[0] += nar

                # Vectorized growth steps
                for _ in range(stepsfly):
                    n0s, slopes, upbs = stepgrow(
                        sim_params,
                        n0s,
                        slopes,
                        binbounds,
                        upbs,
                        const.rho_water,
                        Eij,
                        vrel,
                    )

            # Calculate critical size and update arrays
            if sim_params.method_terminal_velocity == "aglyamov21":
                sizecrit = min((w / wjQQ) ** 2, binbounds[-1])
            else:
                # TODO: adapt for the new terminal velocity calculation
                sizecrit = (w / wjQQ) ** 2.0
                print("sizecrit: ", sizecrit)

            # Calculate binsed using np.floor and max
            binsed = max(
                int(np.floor(np.log(sizecrit / 0.00001) / np.log(np.sqrt(2.0)))), 0
            )

            # Calculate conditions for all bins at once
            upbs = calculate_rpl(n0s, slopes, binbounds)

            # Initialize variables
            mpvout = 0.0
            mpvin = 0.0
            precipN = np.zeros(sim_params.n_bins)
            precipM = np.zeros(sim_params.n_bins)

            # Vectorized calculations for bins >= binsed
            ffg_range = np.arange(binsed, sim_params.n_bins)
            rpl = upbs[ffg_range]
            rmi = binbounds[ffg_range]
            R4 = 0.25 * (rpl**4 - rmi**4)
            R5 = 0.20 * (rpl**5 - rmi**5)

            RR = (
                R4
                * (
                    n0s[ffg_range]
                    - 0.5 * (binbounds[ffg_range + 1] + rmi) * slopes[ffg_range]
                )
                + R5 * slopes[ffg_range]
            )

            mpvout = np.sum(RR)

            precipN[ffg_range] = (rpl - rmi) * (
                n0s[ffg_range]
                - 0.5 * (binbounds[ffg_range + 1] + rmi) * slopes[ffg_range]
            ) + (0.5 * (rpl**2 - rmi**2)) * slopes[ffg_range]

            precipM[ffg_range] = RR

            # Zero out processed bins
            n0s[ffg_range] = 0.0
            slopes[ffg_range] = 0.0
            upbs[ffg_range[ffg_range > 0]] = binbounds[ffg_range[ffg_range > 0]]

            # Vectorized calculations for bins < binsed
            ffgh_range = np.arange(binsed)
            if len(ffgh_range) > 0:
                rpl = upbs[ffgh_range]
                rmi = binbounds[ffgh_range]
                R4 = 0.25 * (rpl**4 - rmi**4)
                R5 = 0.20 * (rpl**5 - rmi**5)

                RRa = (
                    R4
                    * (
                        n0s[ffgh_range]
                        - 0.5 * (binbounds[ffgh_range + 1] + rmi) * slopes[ffgh_range]
                    )
                    + R5 * slopes[ffgh_range]
                )

                mpvin = np.sum(RRa)

            if (mpvout + mpvin) > 0.0:
                # Calculate final condensate
                condensate_new = condensate_new_init * mpvin / (mpvout + mpvin)
            else:
                condensate_new = condensate_new_init  # TODO: check

        P = Pnew
        Trise = Trisenew
        Tfall = Tfallnew
        frise = frisenew
        condensate = condensate_new
        w = wnew
        if w < 0:
            w = sim_params.start_upward_velocity

        n0s_per_level[i, :] = n0s
        slopes_per_level[i, :] = slopes
        upper_bound_per_level[i, :] = upbs
        n_precip_per_level[i, :] = precipN
        m_precip_per_level[i, :] = precipM

    # Final precipitation
    precipN = np.zeros(sim_params.n_bins)
    precipM = np.zeros(sim_params.n_bins)

    # Vectorize calculations for all bins at once
    rpl = upbs
    rmi = binbounds[:-1]

    # Calculate R4 and R5 arrays
    R4 = 0.25 * (rpl**4 - rmi**4)
    R5 = 0.20 * (rpl**5 - rmi**5)

    # Calculate RR array in one operation
    RR = R4 * (n0s - 0.5 * (binbounds[1:] + rmi) * slopes) + R5 * slopes

    # Total mpvout is sum of RR
    mpvout = np.sum(RR)

    # Calculate precipN array
    precipN = (rpl - rmi) * (n0s - 0.5 * (binbounds[1:] + rmi) * slopes) + (
        0.5 * (rpl**2 - rmi**2)
    ) * slopes

    # precipM is just RR
    precipM = RR

    # Update final precipitation levels
    n_precip_per_level[-1] += precipN
    m_precip_per_level[-1] += precipM

    # Calculate flash rates
    samples = stepmax // sim_params.flash_rate_sampling
    J1ss = np.zeros(samples)
    tcrits = np.zeros(samples)

    # Pre-calculate array indices
    sample_indices = np.arange(samples, dtype=int) * sim_params.flash_rate_sampling

    Qcoeff = 1.0
    radju = 1.0

    for ib, i in enumerate(sample_indices):
        n0s = n0s_per_level[i, :]
        slopes = slopes_per_level[i, :]
        P = Pressures[i]
        T = Tempsrise[i]
        w = Velocities[i]

        # Terminal velocity calculation
        if sim_params.method_terminal_velocity == "aglyamov21":
            wjQQ = (
                vrQQ
                * np.sqrt(T / P)
                / (
                    1.0
                    if T > (const.temp_freeze - sim_params.temp_supercool)
                    else rhrro
                )
            )
        elif sim_params.method_terminal_velocity == "loftus21":
            r0s = (binbounds[1:] + binbounds[:-1]) / 2.0
            air_dens = air_density(P, T, const=const)
            surf_tens = liquid_surface_tension(T)
            for i_bin, i_radius in enumerate(r0s):
                wjQQ[i_bin] = (
                    droplet_terminal_velocity(
                        i_radius,
                        air_dens=air_dens,
                        liq_dens=const.rho_water,
                        surf_tens=surf_tens,
                        gravity=const.gravity,
                    )
                    / i_radius**0.5
                )

        # Vectorized particle velocities
        velpart = w - wjQQ * np.sqrt(binbounds[:-1] * 1.1892)

        # Vectorized calculation of rpl
        rpl = calculate_rpl(n0s, slopes, binbounds)

        rmi = binbounds[:-1]
        R4 = 0.25 * (rpl**4 - rmi**4)
        R5 = 0.20 * (rpl**5 - rmi**5)

        # Vectorized Ms and Ns calculations
        Ms = R4 * (n0s - 0.5 * (binbounds[1:] + rmi) * slopes) + R5 * slopes
        Ns = (rpl - rmi) * (n0s - 0.5 * (binbounds[1:] + rmi) * slopes) + (
            0.5 * (rpl**2 - rmi**2)
        ) * slopes

        # Initialize precipitation arrays
        precipN = np.zeros(sim_params.n_bins)
        precipM = np.zeros(sim_params.n_bins)

        # Calculate precipitation coefficients
        rD = (w / wjQQ) ** 2
        blow = np.maximum(binbounds[:-1], rD)
        bhigh = blow * np.sqrt(2.0)
        pCN = binbounds[1:] - blow
        pCD = blow * (3 * w - 2 * wjQQ * np.sqrt(blow)) - bhigh * (
            3 * w - 2 * wjQQ * np.sqrt(bhigh)
        )
        precipC = np.where(rD >= binbounds[1:], 0.0, np.abs(3.0 * pCN / pCD))

        # Vectorized precipitation calculations
        velocities_matrix = Velocities[i:stepmax, None]
        precipN = np.sum(
            n_precip_per_level[i:stepmax] * velocities_matrix * precipC, axis=0
        )
        precipM = np.sum(
            m_precip_per_level[i:stepmax] * velocities_matrix * precipC, axis=0
        )

        # Update total Ns and Ms
        Ns += precipN
        Ms += precipM

        # Initialize output arrays
        n0snew = np.zeros(sim_params.n_bins)
        slopesnew = np.zeros(sim_params.n_bins)
        upboss = np.zeros(sim_params.n_bins)

        # Vectorized calculations for R values
        R2 = 0.5 * (binbounds[1:] ** 2 - binbounds[:-1] ** 2)
        R4 = 0.25 * (binbounds[1:] ** 4 - binbounds[:-1] ** 4)
        R5 = 0.2 * (binbounds[1:] ** 5 - binbounds[:-1] ** 5)
        r0 = 0.5 * (binbounds[1:] + binbounds[:-1])

        # Calculate test values
        valid_bins = (Ns > 0) & (Ms > 0)
        bin_widths = binbounds[1:] - binbounds[:-1]
        n0snewtest = np.where(valid_bins, Ns / bin_widths, 0)
        slopesnewtest = np.where(
            valid_bins, (bin_widths * Ms - Ns * R4) / (bin_widths * R5 - R2 * R4), 0
        )
        ncrit = n0snewtest + slopesnewtest * (binbounds[1:] - r0)

        # Handle simple cases
        simple_case = valid_bins & (ncrit >= 0)
        n0snew[simple_case] = n0snewtest[simple_case]
        slopesnew[simple_case] = slopesnewtest[simple_case]
        upboss[simple_case] = binbounds[1:][simple_case]

        # Handle complex cases
        complex_case = valid_bins & (ncrit < 0)
        if np.any(complex_case):
            for s in np.where(complex_case)[0]:
                bquar = binbounds[s]
                coeffs = [
                    1.0,
                    2.0 * bquar,
                    3.0 * (bquar**2),
                    4.0 * (bquar**3) - 10.0 * Ms[s] / Ns[s],
                ]
                uarray = np.roots(coeffs)
                valid_roots = uarray[
                    (uarray.imag < 1e-10)
                    & (uarray.real >= binbounds[s])
                    & (uarray.real < binbounds[s + 1])
                ]

                if len(valid_roots) > 0:
                    uquar = float(valid_roots[0].real)
                    upboss[s] = uquar
                    if uquar > binbounds[s]:
                        R2_s = 0.5 * (uquar**2 - binbounds[s] ** 2)
                        R4_s = 0.25 * (uquar**4 - binbounds[s] ** 4)
                        R5_s = 0.2 * (uquar**5 - binbounds[s] ** 5)
                        n0snew[s] = Ns[s] / (uquar - binbounds[s])
                        slopesnew[s] = (
                            (uquar - binbounds[s]) * Ms[s] - Ns[s] * R4_s
                        ) / ((uquar - binbounds[s]) * R5_s - R2_s * R4_s)
                        n0snew[s] = (
                            n0snew[s]
                            + (r0[s] - (binbounds[s] + uquar) / 2.0) * slopesnew[s]
                        )
                        # Set charge coefficients using vectorized condition
                        Qcoeff = np.where(
                            T > (const.temp_freeze - sim_params.temp_supercool),
                            1.0 - sim_params.water_collision_efficiency,
                            1.0 - sim_params.ice_collision_efficiency,
                        )
                        radju = np.where(
                            T > (const.temp_freeze - sim_params.temp_supercool),
                            1.0,
                            rhrro,
                        )

        # Calculate charge and electric field rates
        kara = charge_transfer(
            n0snew, slopesnew, binbounds, upboss, velpart, Qcoefff=Qcoeff, radju=radju
        )
        qara = electric_field_change(
            n0snew,
            slopesnew,
            binbounds,
            upboss,
            velpart,
            kara,
            electric_field_strength=0.0,
            const=const,
        )

        # Store current value
        J1ss[ib] = const.vacuum_perm * qara

        # Calculate critical time with vectorized max
        Emax = 3.0 * P
        tcrits[ib] = np.sqrt(2.0 * Emax / np.abs(qara)) if qara != 0 else 0.0

    # Vectorized flash rate calculations
    # Calculate PPV in one operation
    PPV = 5.0 * Pressures[:: sim_params.flash_rate_sampling] * J1ss * tcrits / 2.0

    # Calculate verticalrise array in one operation
    verticalrise = (
        100.0
        * Tempsrise
        * const.universal_gas_constant
        / (const.gravity * Pressures * const.molar_mass_dry_air)
    )[:: sim_params.flash_rate_sampling]

    # Calculate flash rate in one operation
    fl_rate = np.abs((10**6) * verticalrise * PPV / const.energy_per_flash)

    # Pack the results
    p_coord = DimCoord(
        Pressures, var_name="pressure", standard_name="air_pressure", units="Pa"
    )
    p_coord_lfr = DimCoord(
        Pressures[:: sim_params.flash_rate_sampling],
        var_name="pressure_lfr",
        standard_name="air_pressure",
        units="Pa",
    )
    velocity = Cube(
        Velocities,
        dim_coords_and_dims=[(p_coord, 0)],
        var_name="velocity",
        units="m s-1",
    )
    plume_temp = Cube(
        Tempsrise,
        dim_coords_and_dims=[(p_coord, 0)],
        var_name="plume_temp",
        units="K",
    )
    env_temp = Cube(
        Tempsfall, dim_coords_and_dims=[(p_coord, 0)], var_name="env_temp", units="K"
    )
    plume_radius = Cube(
        Radii, dim_coords_and_dims=[(p_coord, 0)], var_name="plume_radius", units="m"
    )
    fs_rise = Cube(
        fsrise,
        dim_coords_and_dims=[(p_coord, 0)],
        var_name="fs_rise",
        units="kg kg-1",
    )
    ls_rise = Cube(
        lsrise,
        dim_coords_and_dims=[(p_coord, 0)],
        var_name="ls_rise",
        units="kg kg-1",
    )

    flash_rate = Cube(
        fl_rate,
        dim_coords_and_dims=[(p_coord_lfr, 0)],
        var_name="flash_rate",
        units="s-1 km-2",
    )

    return CubeList(
        [velocity, plume_temp, env_temp, plume_radius, fs_rise, ls_rise, flash_rate]
    )
