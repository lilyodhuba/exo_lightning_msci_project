"""
Earth 1-D Lightning Simulation - Minimally Modified from Original

Only changes from original:
1. Made n_bins a parameter (default 31)
2. Created plot_comparison() function for organized plotting
3. Slightly cleaner main() function
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def saturation_vapor_pressure(temp):
    """Saturation vapor pressure of water (Lowe 1977)."""
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


def T_stratosphere(pressure):
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


def dT_dP_dry(P, T, f, Cp=14500.0, g=9.81, R=8.31446, mu=0.02896, epsilon=0.6222):
    """Dry adiabatic temperature gradient."""
    return R * T * ((1 + f / epsilon) / (1 + f)) / (mu * P * Cp)


def dT_dP_moist(
    P,
    Trise,
    Tfall,
    lcondensate,
    frise,
    ffall,
    satvappre,
    Cp=14500.0,
    radius=5000.0,
    g=9.81,
    R=8.31446,
    mu=0.02896,
    epsilon=0.6222,
    L=2257000.0,
):
    """Moist adiabatic temperature gradient."""
    Gamma = dT_dP_dry(P, Trise, frise, Cp, g, R, mu, epsilon)
    fS = epsilon * satvappre / P

    if fS <= frise:
        Tv = Trise * ((1 + frise / epsilon) / (1 + frise))
        phi = -0.2 * R * Trise / (radius * mu * P * g)
        numer = (
            1
            + L * fS * mu / (R * Tv)
            - ((Trise - Tfall) * phi / Gamma)
            - L * (fS - ffall) * phi / (Gamma * Cp)
        )
        denom = 1 + (L * L * fS * epsilon * mu) / (Cp * R * Trise * Trise)
        return Gamma * numer / denom
    else:
        phi = -0.2 * R * Trise / (radius * mu * P * g)
        return Gamma - ((Trise - Tfall) * phi)


def dw_dP(
    P,
    Trise,
    Tfall,
    lcondensate,
    frise,
    ffall,
    w,
    radius=5000.0,
    g=9.81,
    R=8.31446,
    mu=0.02896,
    epsilon=0.6222,
):
    """Vertical velocity gradient."""
    phi = -0.2 * R * Trise / (radius * mu * P * g)
    dwdPn = (
        -R
        * (
            Trise * (1 - lcondensate) * ((1 + frise / epsilon) / (1 + frise))
            - Tfall * ((1 + ffall / epsilon) / (1 + ffall))
        )
        / (P * mu * w)
        - w * phi
    )
    return dwdPn


def stepgrow(n0s, slopes, binbounds, upbooms, rho, Eij, vrel, delt, showdetails=0):
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
    n0s,
    slopes,
    binbounds,
    upperbound,
    velocities,
    Qcoefff=1.0,
    ioncharges=[],
    ionnumbers=[],
    ionvelocities=[],
    radju=1,
):
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
    n0s,
    slopes,
    binbounds,
    upperbound,
    velocities,
    charges,
    Efield,
    mfptime=4.0 * (10**-11),
    ioncharges=[],
    ionnumbers=[],
    ionvelocities=[],
    ionmasses=[],
):
    """Calculate electric field rate - COPIED FROM ORIGINAL."""
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
        Jdk = (Efield * ionnumbers[k] * mfptime * (1.602 * 10**-19) ** 2) / ionmasses[k]
        Jd = Jd + Jdk

    return -(Jc + Jd) / (8.854 * (10**-12))


def kayer(Tuu, humid, Ruu, suu, waterS=0.8, iceS=0.0, n_bins=31):
    """
    Run simulation - COPIED FROM ORIGINAL with n_bins parameter added.

    Parameters:
    -----------
    Tuu : Base plume temperature (K)
    humid : Relative humidity (0-1)
    Ruu : Initial plume radius (m)
    suu : Supercooling threshold (K)
    waterS : Water collision efficiency
    iceS : Ice collision efficiency
    n_bins : Number of particle size bins (NEW PARAMETER)
    """
    anlT = 10.0 + 3.0 * (Tuu - 295.0) / 10.0
    Trin = Tuu
    Tfin = Tuu + anlT
    fprea = (
        humid
        * 0.6222
        * saturation_vapor_pressure(Trin)
        / (100000.0 - saturation_vapor_pressure(Trin))
    )
    frise = fprea / (1.0 + fprea)
    Rplume = Ruu
    Cdrag = 0.5
    Eflash = 1.5 * (10**9)
    supercoolK = suu

    Pmax = 1.0
    P = (10**5) * Pmax
    Pstep = 10.0
    w = 0.001
    condensate = 0.0
    Trise = Trin
    Tfall = Tfin
    stepmax = int(P / Pstep) - 10

    Pressures = []
    Tempsrise = []
    Tempsfall = []
    Tempsdiff = []
    Velocities = []
    fsrise = []
    lsrise = []
    Radii = []

    binbounds = np.geomspace(0.00001, 0.46340950011842, n_bins + 1)
    rhoin = 1000.0
    rhoro = 2.5
    rhrro = rhoro ** (1.0 / 3.0)
    vrQQ = np.sqrt((8.0 / (3.0 * Cdrag)) * rhoin * 9.81 * 8.31446 / 0.02896)
    vrel = np.ones([n_bins, n_bins]) * 10.0
    delt = 0.01
    upbsin = np.zeros(n_bins)
    for s in range(len(binbounds) - 1):
        upbsin[s] = binbounds[s]
    upbsin[0] = binbounds[1]

    togglecondens = 0
    n0slist = []
    slopeslist = []
    Upbos = []
    NPrecipitations = []
    MPrecipitations = []
    n0s = np.zeros(n_bins)
    slopes = np.zeros(n_bins)
    upbs = upbsin + np.zeros(n_bins)
    precipN = np.zeros(n_bins)
    precipM = np.zeros(n_bins)

    for i in range(stepmax):
        Pressures.append(P)
        Tempsrise.append(Trise)
        Tempsfall.append(Tfall)
        fsrise.append(frise)
        lsrise.append(condensate)
        Velocities.append(w)
        Tempsdiff.append(Trise - Tfall)
        Radii.append(Rplume)

        Pnew = P - Pstep
        fJrise = frise / (0.6222 + frise * 0.3778)
        muecurr = (1.0 - fJrise * 0.3778) * 0.02896
        Cpcurr = 3.5 * 8.31446 / muecurr
        Tfallnew = Tfall - Pstep * dT_dP_dry(P, Tfall, 0.0, Cpcurr)

        if (P < 22632) or (Tfallnew < 216.6):
            fadjTf = Pstep / 100.0
            Tfallnew = (1 - fadjTf) * Tfallnew + fadjTf * T_stratosphere(Pnew)

        Trisenew = Trise - Pstep * dT_dP_moist(
            P,
            Trise,
            Tfall,
            condensate,
            frise / (1.0 - frise),
            0.0,
            saturation_vapor_pressure(Trise),
            Cpcurr,
            radius=Rplume,
        )

        wnew = w - Pstep * dw_dP(
            P, Trise, Tfall, condensate, frise / (1.0 - frise), 0.0, w, radius=Rplume
        )

        frsnew = 0.6222 * (
            saturation_vapor_pressure(Trisenew)
            / (Pnew - saturation_vapor_pressure(Trisenew))
        )
        fsatnew = frsnew / (1.0 + frsnew)

        if w > 0.001:
            phinew = -0.2 * 8.31446 * Trisenew / (Rplume * 0.02896 * Pnew * 9.81)
            fracdelm = -phinew * Pstep
            frise = frise * (1.0 - fracdelm)
            condensate = condensate * (1.0 - fracdelm)
            n0s = n0s * (1.0 - fracdelm) * (Pnew / (Pnew + Pstep))
            slopes = slopes * (1.0 - fracdelm) * (Pnew / (Pnew + Pstep))

        if fsatnew < frise:
            fcondens = frise - fsatnew
            frisenew = fsatnew
            if w > 0.001:
                frdrho = (-Pstep / Pnew - (Trisenew - Trise) / Trisenew) + (
                    fcondens
                    * 0.6222
                    * 0.3778
                    * 0.02896
                    / ((0.6222 + 0.3778 * frisenew) ** 2)
                ) / muecurr
                frdRpl = 0.5 * fracdelm - 0.5 * frdrho
                Rplume = min(Rplume * (1.0 + frdRpl), Ruu * np.sqrt(2.0))
        else:
            fcondens = 0.0
            frisenew = frise
            if w > 0.001:
                frdrho = (-Pstep - (Trisenew - Trise) * Pnew / Trisenew) / Pnew
                frdRpl = 0.5 * fracdelm - 0.5 * frdrho
                Rplume = min(Rplume * (1.0 + frdRpl), Ruu * np.sqrt(2.0))

        if Trisenew > (273.15 - supercoolK):
            Eij = np.ones([n_bins, n_bins]) * waterS
            wjQQ = vrQQ * np.sqrt(Trise / P)
        else:
            Eij = np.ones([n_bins, n_bins]) * iceS * (rhrro**2)
            wjQQ = vrQQ * np.sqrt(Trise / P) / (rhrro)

        for ie in range(n_bins):
            for j in range(n_bins):
                wiprea = wjQQ * np.sqrt(binbounds[max(ie, j) + 1])
                wjprea = wjQQ * np.sqrt(binbounds[min(ie, j)])
                vrel[ie, j] = abs(wiprea - wjprea)

        verticalrise = Pstep * Trisenew * 8.31446 / (9.81 * Pnew * 0.02896)
        timefly = verticalrise / wnew
        stepsfly = int(np.ceil(timefly / delt))

        condensate_new_init = condensate + fcondens

        if condensate_new_init <= 0 or wnew <= 0:
            condensate_new = 0.0
            togglecondens = 0
            n0s = np.zeros(n_bins)
            slopes = np.zeros(n_bins)
            upbs = upbsin + np.zeros(n_bins)
        else:
            if togglecondens == 0:
                togglecondens = 1
                n0sin = np.zeros(n_bins)
                lz = condensate_new_init
                n0sin[0] = (
                    (lz / (1.0 - lz))
                    * (Pnew * 0.02896)
                    / (
                        8.31446
                        * Trisenew
                        * 1000.0
                        * ((4.0 / 3.0) * np.pi * 0.00001189207**3)
                    )
                    / (binbounds[1] - binbounds[0])
                )
                sizecrit = min((w / wjQQ) ** 2, binbounds[-1])
                slopes = np.zeros(n_bins)
                n0s = n0sin + np.zeros(n_bins)
                upbs = upbsin + np.zeros(n_bins)

                for q in range(stepsfly):
                    n0s, slopes, upbs = stepgrow(
                        n0s, slopes, binbounds, upbs, rhoin, Eij, vrel, delt
                    )
                    for qq in range(n_bins):
                        if not (n0s[qq] > 0 or n0s[qq] < 0):
                            n0s[qq] = 0.0
                        if not (slopes[qq] > 0 or slopes[qq] < 0):
                            slopes[qq] = 0.0
            else:
                nar = (
                    (fcondens / (1.0 - fcondens))
                    * (Pnew * 0.02896)
                    / (
                        8.31446
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
                        n0s, slopes, binbounds, upbs, rhoin, Eij, vrel, delt
                    )
                    for qq in range(n_bins):
                        if not (n0s[qq] > 0 or n0s[qq] < 0):
                            n0s[qq] = 0.0
                        if not (slopes[qq] > 0 or slopes[qq] < 0):
                            slopes[qq] = 0.0

            binsed = max(
                int(np.floor(np.log(sizecrit / 0.00001) / np.log(np.sqrt(2.0)))), 0
            )

            for ssa in range(n_bins):
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
            precipN = np.zeros(n_bins)
            precipM = np.zeros(n_bins)

            for ffg in range(binsed, n_bins, 1):
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
    precipN = np.zeros(n_bins)
    precipM = np.zeros(n_bins)
    mpvout = 0.0
    for ffg in range(0, n_bins, 1):
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
    Rflash = np.zeros(989)

    for ib in range(990):
        if (ib / 100.0) == np.ceil(ib / 100.0):
            print(ib)

        i = ib * 10
        n0s = n0slist[i]
        slopes = slopeslist[i]
        P = Pressures[i]
        T = Tempsrise[i]
        w = Velocities[i]
        velpart = np.zeros(n_bins)

        if T > (273.15 - supercoolK):
            wjQQ = vrQQ * np.sqrt(T / P)
        else:
            wjQQ = vrQQ * np.sqrt(T / P) / (rhrro)

        for s in range(n_bins):
            r0 = binbounds[s] * 1.1892
            velpart[s] = w - wjQQ * np.sqrt(r0)

        Ns = np.zeros(n_bins)
        Ms = np.zeros(n_bins)

        for f in range(n_bins):
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

        precipN = np.zeros(n_bins)
        precipM = np.zeros(n_bins)
        precipC = np.zeros(n_bins)

        for fg in range(n_bins):
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

        for k in range(i, 9990):
            aNp = NPrecipitations[k]
            aMp = MPrecipitations[k]
            for f in range(n_bins):
                precipN[f] = precipN[f] + aNp[f] * Velocities[k] * precipC[f]
                precipM[f] = precipM[f] + aMp[f] * Velocities[k] * precipC[f]

        Ns = Ns + precipN
        Ms = Ms + precipM

        n0snew = np.zeros(n_bins)
        slopesnew = np.zeros(n_bins)
        upboss = np.zeros(n_bins)

        for s in range(n_bins):
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

        if T > (273.15 - supercoolK):
            Qcoeff = 1.0 - waterS
            radju = 1.0
        else:
            Qcoeff = 1.0 - iceS
            radju = rhrro

        kara = dQdt(
            n0snew, slopesnew, binbounds, upboss, velpart, Qcoefff=Qcoeff, radju=radju
        )
        qara = dEdt(n0snew, slopesnew, binbounds, upboss, velpart, kara, 0.0)

        J1ss[ib] = (8.854 * (10**-12)) * qara

        Emax = 3.0 * P
        if qara != 0:
            tcrits[ib] = np.sqrt(2.0 * Emax / abs(qara))

    for io in range(989):
        if tcrits[io] != 0:
            i = io * 10
            P = Pressures[i]
            T = Tempsrise[i]
            tc = tcrits[io]
            PPV = 5.0 * P * J1ss[io] * tc / 2.0
            verticalrise = 100.0 * T * 8.31446 / (9.81 * P * 0.02896)
            Rflash[io] = abs((10**6) * verticalrise * PPV / Eflash)

    Plim = np.zeros(989)
    for a in range(989):
        Plim[a] = Pressures[10 * a]

    return (
        Pressures,
        Velocities,
        Tempsrise,
        Plim,
        Rflash,
        Tempsdiff,
        Radii,
        Tempsfall,
        fsrise,
        lsrise,
    )


def plot_comparison(results_280, results_310, param_name, output_dir):
    """Create comparison plots for two temperature scenarios."""
    P280, v280, T280, Pl280, Rf280, Td280, R280, Tf280, fs280, lc280 = results_280
    P310, v310, T310, Pl310, Rf310, Td310, R310, Tf310, fs310, lc310 = results_310

    Pbar280 = np.array(P280) / 1e5
    Pbar310 = np.array(P310) / 1e5
    Plbar280 = Pl280 / 1e5
    Plbar310 = Pl310 / 1e5

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(param_name, fontsize=14, fontweight="bold")

    # Velocities
    axes[0, 0].plot(Pbar280, v280, "k-", label="280K base", linewidth=1.5)
    axes[0, 0].plot(Pbar310, v310, "b-", label="310K base", linewidth=1.5)
    axes[0, 0].set_xlabel("Pressure (bar)")
    axes[0, 0].set_ylabel("Vertical velocity (m/s)")
    axes[0, 0].set_title("Vertical Plume Velocity")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plume temperatures
    axes[0, 1].plot(Pbar280, T280, "k-", label="280K base", linewidth=1.5)
    axes[0, 1].plot(Pbar310, T310, "b-", label="310K base", linewidth=1.5)
    axes[0, 1].set_xlabel("Pressure (bar)")
    axes[0, 1].set_ylabel("Temperature (K)")
    axes[0, 1].set_title("Plume Temperature")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Environment temperatures
    axes[0, 2].plot(Pbar280, Tf280, "k-", label="280K base", linewidth=1.5)
    axes[0, 2].plot(Pbar310, Tf310, "b-", label="310K base", linewidth=1.5)
    axes[0, 2].set_xlabel("Pressure (bar)")
    axes[0, 2].set_ylabel("Temperature (K)")
    axes[0, 2].set_title("Environment Temperature")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Temperature differences
    axes[1, 0].plot(Pbar280, Td280, "k-", label="280K base", linewidth=1.5)
    axes[1, 0].plot(Pbar310, Td310, "b-", label="310K base", linewidth=1.5)
    axes[1, 0].set_xlabel("Pressure (bar)")
    axes[1, 0].set_ylabel("Temperature difference (K)")
    axes[1, 0].set_title("Plume-Environment Temp Difference")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plume radii
    axes[1, 1].plot(Pbar280, R280, "k-", label="280K base", linewidth=1.5)
    axes[1, 1].plot(Pbar310, R310, "b-", label="310K base", linewidth=1.5)
    axes[1, 1].set_xlabel("Pressure (bar)")
    axes[1, 1].set_ylabel("Radius (m)")
    axes[1, 1].set_title("Plume Radius")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Flash rates
    axes[1, 2].plot(Plbar280, Rf280, "k-", label="280K base", linewidth=1.5)
    axes[1, 2].plot(Plbar310, Rf310, "b-", label="310K base", linewidth=1.5)
    axes[1, 2].set_xlabel("Pressure (bar)")
    axes[1, 2].set_ylabel("Flash rate (flashes/s/km²)")
    axes[1, 2].set_title("Lightning Flash Rate")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"{param_name.replace(' ', '_')}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {filename}")


def main():
    """Main simulation runner."""
    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True, parents=True)

    n_bins = 31  # Number of particle size bins

    for effICE in [0.0]:
        for effWAT in [0.5]:
            for tempSC in [40.0]:
                param_desc = (
                    f"{tempSC}K supercool, {effICE} ice eff, {effWAT} water eff"
                )
                print(f"\n{'=' * 60}")
                print(f"Running: {param_desc}")
                print(f"{'=' * 60}")

                print("\nSimulating 280K...")
                R280 = kayer(
                    280.0,
                    0.9,
                    1000.0,
                    tempSC,
                    waterS=effWAT,
                    iceS=effICE,
                    n_bins=n_bins,
                )

                print("\nSimulating 310K...")
                R310 = kayer(
                    310.0,
                    0.9,
                    1000.0,
                    tempSC,
                    waterS=effWAT,
                    iceS=effICE,
                    n_bins=n_bins,
                )

                print("\nGenerating plots...")
                plot_comparison(R280, R310, param_desc, outdir)

                total_280 = np.sum(R280[4]) * 1.5e3
                total_310 = np.sum(R310[4]) * 1.5e3

                print(f"\n{'=' * 60}")
                print("Results Summary:")
                print(f"{'=' * 60}")
                print(f"280K: Total flash rate = {total_280:.2f} W/m²")
                print(
                    f"      Supercooling = {tempSC}K, efficiencies ice={effICE}, water={effWAT}"
                )
                print(f"\n310K: Total flash rate = {total_310:.2f} W/m²")
                print(
                    f"      Supercooling = {tempSC}K, efficiencies ice={effICE}, water={effWAT}"
                )
                print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
