"""
Earth 1-D lightning simulation for comparison with Jupiter.

Optimized for performance and readability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Physical Constants
@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used in simulation."""

    G: float = 9.81  # gravity (m/s^2)
    R_GAS: float = 8.31446  # gas constant (J/(mol·K))
    MU_AIR: float = 0.02896  # molecular mass of air (kg/mol)
    EPSILON: float = 0.6222  # ratio of molecular masses (water/air)
    RHO_WATER: float = 1000.0  # water density (kg/m^3)
    L_VAPORIZATION: float = 2257000.0  # latent heat (J/kg)
    PERMITTIVITY: float = 8.854e-12  # vacuum permittivity (F/m)
    ELEMENTARY_CHARGE: float = 1.602e-19  # Coulombs

    # Default simulation parameters
    CP_DEFAULT: float = 14500.0  # specific heat (J/(kg·K))
    DRAG_COEFF: float = 0.5  # particle drag coefficient
    FLASH_ENERGY: float = 1.5e9  # lightning flash energy (J)


CONST = PhysicalConstants()


class ParticleGrowth:
    """Handles particle growth and collision dynamics."""

    def __init__(self, binbounds: np.ndarray, rho: float):
        self.binbounds = binbounds
        self.rho = rho
        self.n_bins = len(binbounds) - 1
        self.r0s = (binbounds[1:] + binbounds[:-1]) / 2.0

    def calculate_upperbounds(self, n0s: np.ndarray, slopes: np.ndarray) -> np.ndarray:
        """Calculate upper bounds for each bin."""
        upbos = np.zeros(self.n_bins)

        for s in range(self.n_bins):
            lower_check = (
                n0s[s] + 0.5 * (self.binbounds[s] - self.binbounds[s + 1]) * slopes[s]
            )
            upper_check = (
                n0s[s] + 0.5 * (self.binbounds[s + 1] - self.binbounds[s]) * slopes[s]
            )

            if lower_check <= 0:
                upbos[s] = self.binbounds[s]
            elif upper_check >= 0:
                upbos[s] = self.binbounds[s + 1]
            else:
                upbos[s] = (
                    0.5 * (self.binbounds[s] + self.binbounds[s + 1])
                    - n0s[s] / slopes[s]
                )

        return upbos

    def calculate_moments(
        self, n0s: np.ndarray, slopes: np.ndarray, upbos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate number and mass moments for each bin."""
        Ns = np.zeros(self.n_bins)
        Ms = np.zeros(self.n_bins)
        mmean = np.zeros(self.n_bins)

        for s in range(self.n_bins):
            if upbos[s] >= self.binbounds[s + 1]:
                bin_width = self.binbounds[s + 1] - self.binbounds[s]
                Ns[s] = n0s[s] * bin_width
                Ms[s] = self._calculate_mass_full_bin(s, n0s[s], slopes[s])
            elif upbos[s] > self.binbounds[s]:
                Ns[s] = self._calculate_number_partial_bin(
                    s, n0s[s], slopes[s], upbos[s]
                )
                Ms[s] = self._calculate_mass_partial_bin(s, n0s[s], slopes[s], upbos[s])
            else:
                Ns[s] = 0.0
                Ms[s] = 0.0

            mmean[s] = (
                Ms[s] / Ns[s]
                if Ns[s] > 0
                else (np.pi * self.rho / 3.0)
                * (self.binbounds[s + 1] ** 4 - self.binbounds[s] ** 4)
            )

        return Ns, Ms, mmean

    def _calculate_mass_full_bin(self, s: int, n0: float, slope: float) -> float:
        """Calculate mass for a fully occupied bin."""
        r_low, r_high = self.binbounds[s], self.binbounds[s + 1]
        r0 = self.r0s[s]

        term1 = 0.2 * (r_high**5 - r_low**5) * slope
        term2 = 0.25 * (r_high**4 - r_low**4) * (n0 - slope * r0)

        return (4 * np.pi * self.rho / 3.0) * (term1 + term2)

    def _calculate_mass_partial_bin(
        self, s: int, n0: float, slope: float, r_up: float
    ) -> float:
        """Calculate mass for a partially occupied bin."""
        r_low = self.binbounds[s]
        r0 = self.r0s[s]

        term1 = 0.2 * (r_up**5 - r_low**5) * slope
        term2 = 0.25 * (r_up**4 - r_low**4) * (n0 - slope * r0)

        return (4 * np.pi * self.rho / 3.0) * (term1 + term2)

    def _calculate_number_partial_bin(
        self, s: int, n0: float, slope: float, r_up: float
    ) -> float:
        """Calculate particle number for a partially occupied bin."""
        r_low = self.binbounds[s]
        r0 = self.r0s[s]

        return n0 * (r_up - r_low) + slope * (r_up - r_low) * (
            0.5 * (r_up + r_low) - r0
        )

    def step(
        self,
        n0s: np.ndarray,
        slopes: np.ndarray,
        upbos: np.ndarray,
        Eij: np.ndarray,
        vrel: np.ndarray,
        delt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform one time step of particle growth."""
        Ns, Ms, mmean = self.calculate_moments(n0s, slopes, upbos)
        Nsnew = Ns.copy()
        Msnew = Ms.copy()

        # Collision-coalescence
        for i in range(self.n_bins):
            for j in range(i + 1):
                collision_prob = min(
                    Eij[i, j]
                    * np.pi
                    * (self.r0s[i] ** 2 + self.r0s[j] ** 2)
                    * vrel[i, j]
                    * Ns[j]
                    * delt,
                    1.0,
                )

                Nsnew[j] -= collision_prob * Ns[i]
                Msnew[j] -= collision_prob * Ns[i] * mmean[j]
                Msnew[i] += collision_prob * Ns[i] * mmean[j]

                # Handle particle transfer to next bin
                rx = (self.binbounds[i + 1] ** 3 - self.r0s[j] ** 3) ** (1 / 3)

                if upbos[i] >= self.binbounds[i + 1]:
                    rxx = self.binbounds[i + 1]
                elif upbos[i] > rx:
                    rxx = upbos[i]
                else:
                    continue

                Nxx, Mxx = self._calculate_transfer_moments(
                    i, rx, rxx, n0s[i], slopes[i], mmean[j]
                )

                if i + 1 < self.n_bins:
                    Msnew[i + 1] += (Mxx + mmean[j] * Nxx) * collision_prob
                    Msnew[i] -= (Mxx + mmean[j] * Nxx) * collision_prob
                    Nsnew[i + 1] += Nxx * collision_prob
                    Nsnew[i] -= Nxx * collision_prob

        # Recalculate distribution parameters
        n0snew, slopesnew, upbos_new = self._recalculate_distribution(Nsnew, Msnew)

        return n0snew, slopesnew, upbos_new

    def _calculate_transfer_moments(
        self, i: int, rx: float, rxx: float, n0: float, slope: float, mmean_j: float
    ) -> Tuple[float, float]:
        """Calculate moments for particles transferring between bins."""
        r0 = self.r0s[i]

        Nxx = (
            n0 * (rxx - rx)
            - slope * r0 * (rxx - rx)
            + slope * (rxx - rx) * (rxx + rx) / 2.0
        )
        Mxx = (np.pi * self.rho / 3.0) * (rxx**4 - rx**4) * (n0 - slope * r0) + (
            4 * np.pi * self.rho / 15.0
        ) * slope * (rxx**5 - rx**5)

        return Nxx, Mxx

    def _recalculate_distribution(
        self, Nsnew: np.ndarray, Msnew: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recalculate distribution parameters from moments."""
        n0snew = np.zeros(self.n_bins)
        slopesnew = np.zeros(self.n_bins)
        upbos = np.zeros(self.n_bins)

        for s in range(self.n_bins):
            if Nsnew[s] <= 0 or Msnew[s] <= 0:
                upbos[s] = self.binbounds[s]
                continue

            bin_width = self.binbounds[s + 1] - self.binbounds[s]
            R2 = 0.5 * (self.binbounds[s + 1] ** 2 - self.binbounds[s] ** 2)
            R4 = 0.25 * (self.binbounds[s + 1] ** 4 - self.binbounds[s] ** 4)
            R5 = 0.2 * (self.binbounds[s + 1] ** 5 - self.binbounds[s] ** 5)

            n0_test = Nsnew[s] / bin_width
            slope_test = (
                3 * bin_width * Msnew[s] / (4 * self.rho * np.pi) - Nsnew[s] * R4
            ) / (bin_width * R5 - R2 * R4)

            ncrit = n0_test + slope_test * (self.binbounds[s + 1] - self.r0s[s])

            if ncrit >= 0:
                n0snew[s] = n0_test
                slopesnew[s] = slope_test
                upbos[s] = self.binbounds[s + 1]
            else:
                upbos[s] = self._solve_upperbound(s, Nsnew[s], Msnew[s])
                if upbos[s] > self.binbounds[s]:
                    n0snew[s], slopesnew[s] = self._fit_distribution(
                        s, Nsnew[s], Msnew[s], upbos[s]
                    )

        return n0snew, slopesnew, upbos

    def _solve_upperbound(self, s: int, N: float, M: float) -> float:
        """Solve for upper bound using quartic equation."""
        b = self.binbounds[s]
        coeffs = [1.0, 2.0 * b, 3.0 * b**2, 4.0 * b**3 - 10.0 * M / N]
        roots = np.roots(coeffs)

        for root in roots:
            if (
                root.real >= self.binbounds[s]
                and root.real < self.binbounds[s + 1]
                and abs(root.imag) < 1e-10
            ):
                return root.real

        return self.binbounds[s]

    def _fit_distribution(
        self, s: int, N: float, M: float, upb: float
    ) -> Tuple[float, float]:
        """Fit linear distribution parameters to moments."""
        bin_low = self.binbounds[s]
        bin_width = upb - bin_low

        R4 = 0.25 * (upb**4 - bin_low**4)

        n0_base = N / bin_width
        slope = (
            40.0
            * (3 * bin_width * M / (4 * self.rho * np.pi) - N * R4)
            / ((3 * upb**2 + 4 * upb * bin_low + 3 * bin_low**2) * bin_width**4)
        )
        n0 = n0_base + (self.r0s[s] - (bin_low + upb) / 2.0) * slope

        return n0, slope


class ChargingModel:
    """Handles particle charging calculations."""

    @staticmethod
    def calculate_graupel_factor(radius_m: float) -> float:
        """Calculate graupel charging factor based on radius."""
        radius_microns = radius_m * 1e6

        if radius_microns <= 111:
            return 0.0271 * radius_microns**2.7
        else:
            return 98.8 * radius_microns**0.98

    @staticmethod
    def calculate_charge_rate(
        n0s: np.ndarray,
        slopes: np.ndarray,
        binbounds: np.ndarray,
        velocities: np.ndarray,
        Q_coeff: float = 1.0,
        radius_adj: float = 1.0,
    ) -> np.ndarray:
        """Calculate charging rate for each particle bin."""
        n_bins = len(n0s)
        r0s = (binbounds[1:] + binbounds[:-1]) / 2.0

        # Calculate particle concentrations
        ns = np.zeros(n_bins)
        upbos = np.zeros(n_bins)

        for s in range(n_bins):
            lower_check = n0s[s] + 0.5 * (binbounds[s] - binbounds[s + 1]) * slopes[s]
            upper_check = n0s[s] + 0.5 * (binbounds[s + 1] - binbounds[s]) * slopes[s]

            if lower_check <= 0:
                upbos[s] = binbounds[s]
            elif upper_check >= 0:
                upbos[s] = binbounds[s + 1]
            else:
                upbos[s] = 0.5 * (binbounds[s] + binbounds[s + 1]) - n0s[s] / slopes[s]

            r_low = binbounds[s]
            bin_width = upbos[s] - r_low
            ns[s] = (
                bin_width * (n0s[s] - 0.5 * (binbounds[s + 1] + r_low) * slopes[s])
                + 0.5 * (upbos[s] ** 2 - r_low**2) * slopes[s]
            )

        # Calculate charge rates
        dQdt = np.zeros(n_bins)

        for i in range(n_bins):
            dQ_total = 0.0

            for j in range(n_bins):
                r_graupel = radius_adj * min(r0s[i], r0s[j])
                Gr = ChargingModel.calculate_graupel_factor(r_graupel)

                vel_diff = abs(velocities[i] - velocities[j])
                delQ = ((vel_diff / 3.0) ** 2.5) * Gr * 1e-15

                dQ_interaction = (
                    delQ * ns[j] * np.pi * (r0s[i] ** 2 + r0s[j] ** 2) * radius_adj**2
                )

                dQ_total += dQ_interaction if i < j else -dQ_interaction

            dQdt[i] = dQ_total * Q_coeff

        return dQdt

    @staticmethod
    def calculate_electric_field_rate(
        n0s: np.ndarray,
        slopes: np.ndarray,
        binbounds: np.ndarray,
        velocities: np.ndarray,
        charges: np.ndarray,
        E_field: float,
        mfp_time: float = 4e-11,
    ) -> float:
        """Calculate rate of change of electric field."""
        n_bins = len(n0s)

        # Calculate particle concentrations
        ns = np.zeros(n_bins)
        for s in range(n_bins):
            lower_check = n0s[s] + 0.5 * (binbounds[s] - binbounds[s + 1]) * slopes[s]
            upper_check = n0s[s] + 0.5 * (binbounds[s + 1] - binbounds[s]) * slopes[s]

            if lower_check <= 0:
                upb = binbounds[s]
            elif upper_check >= 0:
                upb = binbounds[s + 1]
            else:
                upb = 0.5 * (binbounds[s] + binbounds[s + 1]) - n0s[s] / slopes[s]

            r_low = binbounds[s]
            bin_width = upb - r_low
            ns[s] = (
                bin_width * (n0s[s] - 0.5 * (binbounds[s + 1] + r_low) * slopes[s])
                + 0.5 * (upb**2 - r_low**2) * slopes[s]
            )

        # Conduction current
        J_c = -np.sum(ns * velocities * charges)

        # Displacement current (simplified, no ions in this version)
        J_d = 0.0

        return -(J_c + J_d) / CONST.PERMITTIVITY


class ConvectionModel:
    """Handles moist convection calculations."""

    @staticmethod
    def dwdP(
        P: float,
        T_rise: float,
        T_fall: float,
        l_cond: float,
        f_rise: float,
        f_fall: float,
        w: float,
        radius: float = 5000.0,
    ) -> float:
        """Calculate vertical velocity change with pressure."""
        phi = -0.2 * CONST.R_GAS * T_rise / (radius * CONST.MU_AIR * P * CONST.G)

        term1 = (
            T_rise * (1.0 - l_cond) * ((1.0 + f_rise / CONST.EPSILON) / (1.0 + f_rise))
        )
        term2 = T_fall * ((1.0 + f_fall / CONST.EPSILON) / (1.0 + f_fall))

        dwdP = -CONST.R_GAS * (term1 - term2) / (P * CONST.MU_AIR * w) - w * phi

        return dwdP

    @staticmethod
    def dTdP_dry(P: float, T: float, f: float, Cp: float = None) -> float:
        """Calculate dry adiabatic temperature gradient."""
        if Cp is None:
            Cp = CONST.CP_DEFAULT

        return (
            CONST.R_GAS
            * T
            * ((1.0 + f / CONST.EPSILON) / (1.0 + f))
            / (CONST.MU_AIR * P * Cp)
        )

    @staticmethod
    def dTdP_moist(
        P: float,
        T_rise: float,
        T_fall: float,
        l_cond: float,
        f_rise: float,
        f_fall: float,
        sat_vap_pres: float,
        Cp: float = None,
        radius: float = 5000.0,
    ) -> float:
        """Calculate moist adiabatic temperature gradient."""
        if Cp is None:
            Cp = CONST.CP_DEFAULT

        Gamma = ConvectionModel.dTdP_dry(P, T_rise, f_rise, Cp)
        f_S = CONST.EPSILON * sat_vap_pres / P

        if f_S <= f_rise:
            Tv = T_rise * ((1.0 + f_rise / CONST.EPSILON) / (1.0 + f_rise))
            phi = -0.2 * CONST.R_GAS * T_rise / (radius * CONST.MU_AIR * P * CONST.G)

            numer = (
                1.0
                + CONST.L_VAPORIZATION * f_S * CONST.MU_AIR / (CONST.R_GAS * Tv)
                - (T_rise - T_fall) * phi / Gamma
                - CONST.L_VAPORIZATION * (f_S - f_fall) * phi / (Gamma * Cp)
            )

            denom = 1.0 + (
                CONST.L_VAPORIZATION**2 * f_S * CONST.EPSILON * CONST.MU_AIR
            ) / (Cp * CONST.R_GAS * T_rise**2)

            return Gamma * numer / denom
        else:
            phi = -0.2 * CONST.R_GAS * T_rise / (radius * CONST.MU_AIR * P * CONST.G)
            return Gamma - (T_rise - T_fall) * phi

    @staticmethod
    def saturation_vapor_pressure(T: float) -> float:
        """Calculate saturation vapor pressure of water (Lowe 1977)."""
        coeffs = [
            6984.505294,
            -188.9039310,
            2.133357675,
            -0.01288580973,
            4.393587233e-5,
            -8.023923082e-8,
            6.136820929e-11,
        ]

        mb = sum(c * T**i for i, c in enumerate(coeffs))
        return max(mb * 100.0, 0.0)

    @staticmethod
    def stratospheric_temperature(P: float) -> float:
        """Earth stratospheric temperature (International Standard Atmosphere)."""
        if P > 5474.9:
            return 216.6
        elif P > 868.02:
            LP_off = np.log(P / 5474.9) / np.log(868.02 / 5474.9)
            return 216.6 + 12.0 * LP_off
        elif P > 110.91:
            LP_off = np.log(P / 868.02) / np.log(110.91 / 868.02)
            return 228.6 + 42.0 * LP_off
        elif P > 66.939:
            return 270.6
        else:
            LP_off = np.log(P / 66.939) / np.log(3.9564 / 66.939)
            return 270.6 - 56.0 * LP_off


@dataclass
class SimulationParams:
    """Parameters for lightning simulation."""

    T_base: float  # Base plume temperature (K)
    humidity: float  # Relative humidity (0-1)
    radius: float  # Plume radius (m)
    supercool: float  # Supercooling threshold (K)
    water_efficiency: float = 0.8  # Water collision efficiency
    ice_efficiency: float = 0.0  # Ice collision efficiency
    n_bins: int = 31  # number of size bins (default kept to previous value)


class LightningSimulation:
    """Main lightning simulation class."""

    def __init__(self, params: SimulationParams):
        self.params = params
        self.n_bins = max(2, int(self.params.n_bins))
        self.binbounds = np.geomspace(1e-5, 0.4634, self.n_bins + 1)
        self.particle_growth = ParticleGrowth(self.binbounds, CONST.RHO_WATER)
        self.convection = ConvectionModel()

        # Simulation parameters
        self.P_step = 10.0  # Pressure step (Pa)
        self.dt = 0.01  # Time step (s)
        self.rho_ratio = 2.5  # Liquid to solid density ratio

    def run(self) -> dict:
        """Run the lightning simulation."""
        # Initialize
        P = 1e5  # Starting pressure (1 bar)

        T_plume_diff = 10.0 + 3.0 * (self.params.T_base - 295.0) / 10.0
        T_rise = self.params.T_base
        T_fall = self.params.T_base + T_plume_diff

        svp_surface = self.convection.saturation_vapor_pressure(T_rise)
        f_pre = self.params.humidity * CONST.EPSILON * svp_surface / (1e5 - svp_surface)
        f_rise = f_pre / (1.0 + f_pre)

        w = 0.001  # Initial velocity
        condens_liq = 0.0  # Liquid condensate
        R_plume = self.params.radius

        # Storage arrays
        results = {
            "pressure": [],
            "velocity": [],
            "T_rise": [],
            "T_fall": [],
            "T_diff": [],
            "radius": [],
            "f_rise": [],
            "l_rise": [],
            "flash_rate": [],
        }

        # Particle distribution
        n0s = np.zeros(self.n_bins)
        slopes = np.zeros(self.n_bins)
        condensed = False

        step_max = int(P / self.P_step) - 10

        for step in range(step_max):
            # Store current state
            results["pressure"].append(P)
            results["velocity"].append(w)
            results["T_rise"].append(T_rise)
            results["T_fall"].append(T_fall)
            results["T_diff"].append(T_rise - T_fall)
            results["radius"].append(R_plume)
            results["f_rise"].append(f_rise)
            results["l_rise"].append(condens_liq)

            # Update state
            P_new = P - self.P_step

            # Calculate specific heat
            f_J = f_rise / (0.6222 + f_rise * 0.3778)
            mu_curr = (1.0 - f_J * 0.3778) * CONST.MU_AIR
            Cp_curr = 3.5 * CONST.R_GAS / mu_curr

            # Update temperatures
            T_fall_new = T_fall - self.P_step * self.convection.dTdP_dry(
                P, T_fall, 0.0, Cp_curr
            )

            if P < 22632 or T_fall_new < 216.6:
                f_adj = self.P_step / 100.0
                T_fall_new = (
                    1 - f_adj
                ) * T_fall_new + f_adj * self.convection.stratospheric_temperature(
                    P_new
                )

            svp_new = self.convection.saturation_vapor_pressure(T_rise)
            T_rise_new = T_rise - self.P_step * self.convection.dTdP_moist(
                P,
                T_rise,
                T_fall,
                condens_liq,
                f_rise / (1.0 - f_rise),
                0.0,
                svp_new,
                Cp_curr,
                R_plume,
            )

            w_new = w - self.P_step * self.convection.dwdP(
                P, T_rise, T_fall, condens_liq, f_rise / (1.0 - f_rise), 0.0, w, R_plume
            )

            # Calculate saturation
            svp_new = self.convection.saturation_vapor_pressure(T_rise_new)
            f_sat_new = 0.6222 * svp_new / (P - svp_new)
            f_sat_new = f_sat_new / (1.0 + f_sat_new)

            # Handle entrainment
            if w > 0.001:
                phi = (
                    -0.2
                    * CONST.R_GAS
                    * T_rise_new
                    / (R_plume * CONST.MU_AIR * P_new * CONST.G)
                )
                frac_del_m = -phi * self.P_step

                f_rise *= 1.0 - frac_del_m
                condens_liq *= 1.0 - frac_del_m
                n0s *= (1.0 - frac_del_m) * (P_new / (P_new + self.P_step))
                slopes *= (1.0 - frac_del_m) * (P_new / (P_new + self.P_step))

            # Handle condensation/evaporation and plume radius
            if f_sat_new < f_rise:
                f_condense = f_rise - f_sat_new
                f_rise_new = f_sat_new
                condens_liq_new_init = condens_liq + f_condense

                if w > 0.001:
                    frac_drho = (
                        -self.P_step / P_new
                        - (T_rise_new - T_rise) / T_rise_new
                        + f_condense
                        * 0.6222
                        * 0.3778
                        * CONST.MU_AIR
                        / ((0.6222 + 0.3778 * f_rise_new) ** 2)
                        / mu_curr
                    )
                    frac_dR = 0.5 * frac_del_m - 0.5 * frac_drho
                    R_plume = min(
                        R_plume * (1.0 + frac_dR), self.params.radius * np.sqrt(2.0)
                    )
            else:
                f_condense = 0.0
                f_rise_new = f_rise
                condens_liq_new_init = condens_liq

                if w > 0.001:
                    frac_drho = (
                        -self.P_step - (T_rise_new - T_rise) * P_new / T_rise_new
                    ) / P_new
                    frac_dR = 0.5 * frac_del_m - 0.5 * frac_drho
                    R_plume = min(
                        R_plume * (1.0 + frac_dR), self.params.radius * np.sqrt(2.0)
                    )

            # Particle growth
            if condens_liq_new_init > 0 and w_new > 0:
                n0s, slopes = self._handle_particle_growth(
                    n0s,
                    slopes,
                    T_rise_new,
                    P_new,
                    w_new,
                    condens_liq_new_init,
                    f_condense,
                    condensed,
                )
                condensed = True

                # Calculate precipitation
                condens_liq_new = self._calculate_precipitation(
                    n0s, slopes, T_rise_new, P_new, w_new, condens_liq_new_init
                )
            else:
                condens_liq_new = 0.0
                condensed = False
                n0s = np.zeros(self.n_bins)
                slopes = np.zeros(self.n_bins)

            # Update state variables
            P = P_new
            T_rise = T_rise_new
            T_fall = T_fall_new
            f_rise = f_rise_new
            condens_liq = condens_liq_new
            w = max(w_new, 0.001)

        # Calculate flash rates
        results["flash_rate"] = self._calculate_flash_rates(results)

        return results

    def _handle_particle_growth(
        self,
        n0s: np.ndarray,
        slopes: np.ndarray,
        T: float,
        P: float,
        w: float,
        l_total: float,
        f_condense: float,
        already_condensed: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Handle particle growth through collision-coalescence."""
        # Determine collision efficiency and particle properties
        if T > (273.1 - self.params.supercool):
            Eij = np.ones([self.n_bins, self.n_bins]) * self.params.water_efficiency
            rho_ratio_factor = 1.0
        else:
            rho_ratio_1_3 = self.rho_ratio ** (1 / 3)
            Eij = (
                np.ones([self.n_bins, self.n_bins])
                * self.params.ice_efficiency
                * rho_ratio_1_3**2
            )
            rho_ratio_factor = rho_ratio_1_3

        # Terminal velocity factor
        vrQQ = np.sqrt(
            (8.0 / (3.0 * CONST.DRAG_COEFF))
            * CONST.RHO_WATER
            * CONST.G
            * CONST.R_GAS
            / CONST.MU_AIR
        )
        wjQQ = vrQQ * np.sqrt(T / P) / rho_ratio_factor

        # Calculate relative velocities
        vrel = np.zeros([self.n_bins, self.n_bins])
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                wi = wjQQ * np.sqrt(self.binbounds[max(i, j) + 1])
                wj = wjQQ * np.sqrt(self.binbounds[min(i, j)])
                vrel[i, j] = abs(wi - wj)

        # Time for this pressure step
        vertical_rise = self.P_step * T * CONST.R_GAS / (CONST.G * P * CONST.MU_AIR)
        time_fly = vertical_rise / w
        n_steps = int(np.ceil(time_fly / self.dt))

        # Initialize or add new condensate
        if not already_condensed:
            n0s = np.zeros(self.n_bins)
            # Add initial condensate to smallest bin
            n0s[0] = (
                (l_total / (1.0 - l_total))
                * (P * CONST.MU_AIR)
                / (CONST.R_GAS * T * 1000.0 * (4.0 / 3.0) * np.pi * 1.189207e-5**3)
                / (self.binbounds[1] - self.binbounds[0])
            )
            slopes = np.zeros(self.n_bins)
        else:
            # Add new condensate
            if f_condense > 0:
                n_add = (
                    (f_condense / (1.0 - f_condense))
                    * (P * CONST.MU_AIR)
                    / (CONST.R_GAS * T * 1000.0 * (4.0 / 3.0) * np.pi * 1.189207e-5**3)
                    / (self.binbounds[1] - self.binbounds[0])
                )
                n0s[0] += n_add

        # Grow particles
        upbos = self.particle_growth.calculate_upperbounds(n0s, slopes)

        for _ in range(n_steps):
            n0s, slopes, upbos = self.particle_growth.step(
                n0s, slopes, upbos, Eij, vrel, self.dt
            )
            # Handle NaN values
            n0s = np.nan_to_num(n0s, nan=0.0)
            slopes = np.nan_to_num(slopes, nan=0.0)

        return n0s, slopes

    def _calculate_precipitation(
        self,
        n0s: np.ndarray,
        slopes: np.ndarray,
        T: float,
        P: float,
        w: float,
        l_total: float,
    ) -> float:
        """Calculate remaining liquid after precipitation removal."""
        # Determine terminal velocity
        if T > (273.1 - self.params.supercool):
            rho_ratio_factor = 1.0
        else:
            rho_ratio_factor = self.rho_ratio ** (1 / 3)

        vrQQ = np.sqrt(
            (8.0 / (3.0 * CONST.DRAG_COEFF))
            * CONST.RHO_WATER
            * CONST.G
            * CONST.R_GAS
            / CONST.MU_AIR
        )
        wjQQ = vrQQ * np.sqrt(T / P) / rho_ratio_factor

        # Critical size for sedimentation
        size_crit = min((w / wjQQ) ** 2, self.binbounds[-1])
        bin_sed = max(int(np.floor(np.log(size_crit / 1e-5) / np.log(np.sqrt(2.0)))), 0)

        # Recalculate upperbounds
        upbos = self.particle_growth.calculate_upperbounds(n0s, slopes)

        # Calculate mass in and above critical bin
        m_out = 0.0
        m_in = 0.0

        for s in range(bin_sed, self.n_bins):
            r_low = self.binbounds[s]
            r_up = upbos[s]
            R4 = 0.25 * (r_up**4 - r_low**4)
            R5 = 0.20 * (r_up**5 - r_low**5)

            mass_bin = (
                R4 * (n0s[s] - 0.5 * (self.binbounds[s + 1] + r_low) * slopes[s])
                + R5 * slopes[s]
            )
            m_out += mass_bin

            # Remove precipitating particles
            n0s[s] = 0.0
            slopes[s] = 0.0

        for s in range(bin_sed):
            r_low = self.binbounds[s]
            r_up = upbos[s]
            R4 = 0.25 * (r_up**4 - r_low**4)
            R5 = 0.20 * (r_up**5 - r_low**5)

            mass_bin = (
                R4 * (n0s[s] - 0.5 * (self.binbounds[s + 1] + r_low) * slopes[s])
                + R5 * slopes[s]
            )
            m_in += mass_bin

        # Return remaining liquid fraction
        if m_out + m_in > 0:
            return l_total * m_in / (m_out + m_in)
        else:
            return 0.0

    def _calculate_flash_rates(self, results: dict) -> np.ndarray:
        """Calculate lightning flash rates at different pressure levels."""
        # Simplified flash rate calculation
        # In full version, this would involve detailed charging calculations

        n_levels = len(results["pressure"]) // 10
        flash_rates = np.zeros(n_levels)

        # This is a placeholder - full implementation would calculate
        # charge separation and electric field buildup

        return flash_rates


def run_simulation_suite(
    output_dir: Optional[str, Path] = Path(__file__).parent / "output",
):
    """Run suite of simulations with different parameters."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    print("Earth Lightning Simulation Suite")
    print("=" * 50)

    # Parameter ranges
    ice_efficiencies = [0.0]
    water_efficiencies = [0.5, 0.8, 1.0]
    supercooling_values = [0.0, 10.0, 20.0, 30.0, 40.0]
    base_temperatures = [280.0, 290.0, 300.0, 310.0]

    all_results = {}

    for ice_eff in ice_efficiencies:
        for water_eff in water_efficiencies:
            for supercool in supercooling_values:
                case_name = f"ice{ice_eff}_water{water_eff}_sc{supercool}K"
                print(f"\nRunning case: {case_name}")

                case_results = {}

                for T_base in base_temperatures:
                    print(f"  Temperature: {T_base}K")

                    params = SimulationParams(
                        T_base=T_base,
                        humidity=0.9,
                        radius=1000.0,
                        supercool=supercool,
                        water_efficiency=water_eff,
                        ice_efficiency=ice_eff,
                    )

                    sim = LightningSimulation(params)
                    results = sim.run()
                    case_results[f"{T_base}K"] = results

                all_results[case_name] = case_results

                # Generate plots for this case
                _plot_case_results(case_results, case_name, output_dir)

    print("\n" + "=" * 50)
    print("Simulation complete!")

    return all_results


def _plot_case_results(case_results: dict, case_name: str, output_dir: str):
    """Generate plots for a case."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
    fig.suptitle(case_name, fontsize=16)

    colors = {"280.0K": "black", "290.0K": "red", "300.0K": "green", "310.0K": "blue"}

    for temp_key, results in case_results.items():
        P_bar = np.array(results["pressure"]) / 1e5

        axes[0, 0].plot(
            P_bar, results["velocity"], color=colors[temp_key], label=temp_key
        )
        axes[0, 1].plot(
            P_bar, results["T_rise"], color=colors[temp_key], label=temp_key
        )
        axes[0, 2].plot(
            P_bar, results["T_fall"], color=colors[temp_key], label=temp_key
        )
        axes[1, 0].plot(
            P_bar, results["T_diff"], color=colors[temp_key], label=temp_key
        )
        axes[1, 1].plot(
            P_bar, results["radius"], color=colors[temp_key], label=temp_key
        )

    axes[0, 0].set(xlabel="Pressure (bar)", ylabel="Velocity (m/s)")
    axes[0, 1].set(xlabel="Pressure (bar)", ylabel="Plume Temp (K)")
    axes[0, 2].set(xlabel="Pressure (bar)", ylabel="Environment Temp (K)")
    axes[1, 0].set(xlabel="Pressure (bar)", ylabel="Temp Difference (K)")
    axes[1, 1].set(xlabel="Pressure (bar)", ylabel="Plume Radius (m)")
    axes[1, 2].axis("off")

    for ax in axes.flat:
        if ax.has_data():
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / f"{case_name}.png", dpi=150, bbox_inches="tight")
    fig.close()


if __name__ == "__main__":
    results = run_simulation_suite()
    print("\nSimulation results saved to 'output' directory")
