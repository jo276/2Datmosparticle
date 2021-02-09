import numpy as np
#### written by JO Dec 2020 ###'#
#### This file contains analysis functions

def get_transmission_spectrum(grid,field,rays,get_Q_ext,wav):

    # this function performs a transmission spectrum as a function of wavelenth
    # Qext_vs_x is a Python interpolation object that provides the particles
    # extinction cross-section as a function of optical paramter lambda and a
    # Qext_vs_x is assumed to be a log-function so Qext = 10.** get_Q_ext(log10(lambda),log10(size))
    # lambda and size should be in cm

    # find the optical depth as a function of impact parameter through the domain 
    # we'll use the ray-tracing routine setup in the rays object

    Rp = np.zeros(np.size(wav))

    for i in range(np.size(wav)):

        Qext_3D = 10.**get_Q_ext(np.log10(wav[i]),np.log10(field.par_size),grid=False)

        kappa_3D = 3./4. * Qext_3D / field.par_size / field.par_dens_in

        extinction_3D = kappa_3D * field.par_dens 
        extinction_par_2D = np.sum(extinction_3D,axis=2) # sum all the size bins

        gas_opacity = 6e-3*(wav[i]/(0.3*1e-4))**(-4.) # H2 rayleigh scattering (Figure 1 Freedman et al. 2014)

        extinction_total = extinction_par_2D + gas_opacity * field.gas_dens

        rays.do_ray_trace(extinction_total)

        # now calculate Rp in transmission
        Flux_obs = np.trapz(2.*np.pi*rays.Xrays[rays.id_terminator:]*np.exp(-rays.tau_end[rays.id_terminator:]),rays.Xrays[rays.id_terminator:])
        Flux_exp = np.pi*rays.Xrays[-1]**2.

        Rp[i] = rays.Xrays[-1]*np.sqrt(1.-Flux_obs/Flux_exp)

    return Rp