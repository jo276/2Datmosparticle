import numpy as np
#### written by JO Dec 2020/Jan 2021 ###'#
#### This file contains analysis functions

def get_transmission_spectrum(grid,field,rays,get_Q_ext,wav,no_aerosol_flag=False,use_external_tran_spec=False,get_tau=None):

    # this function performs a transmission spectrum as a function of wavelenth
    # Qext_vs_x is a Python interpolation object that provides the particles
    # extinction cross-section as a function of optical paramter lambda and a
    # Qext_vs_x is assumed to be a log-function so Qext = 10.** get_Q_ext(log10(size),log10(lam))
    # lambda and size should be in cm

    # find the optical depth as a function of impact parameter through the domain 
    # we'll use the ray-tracing routine setup in the rays object

    # no_haze_flag allows the switching off of the hazes to calculate pure gas spectra

    Rp = np.zeros(np.size(wav))

    for i in range(np.size(wav)):

        Qext_3D = 10.**get_Q_ext(np.log10(field.par_size),np.log10(wav[i]),grid=False)

        kappa_3D = 3./4. * Qext_3D / field.par_size / field.par_dens_in

        extinction_3D = kappa_3D * field.par_dens 
        extinction_par_2D = np.sum(extinction_3D,axis=2) # sum all the size bins

        gas_opacity = 2.3e-3*(wav[i]/(0.3*1e-4))**(-4.) # H2-He rayleigh scattering extracted by fitting opacity data in petitRADTRANS code

        if (use_external_tran_spec):
            weight = 0.
        else:
            weight = 1.

        if (no_aerosol_flag):
            extinction_total = weight * gas_opacity * field.gas_dens
        else:
            extinction_total = extinction_par_2D + weight * gas_opacity * field.gas_dens

        rays.do_ray_trace(extinction_total)

        if (use_external_tran_spec):
            ### get taus from gas coming from external transmission spectrum
            if (get_tau == None):
                print( "Error Must specifiy tau_interpolation for aersol free spectrum")
                Rp = -1.
                return

            tau_gas = get_tau(rays.Xrays[rays.id_terminator:],np.log10(wav[i]),grid=False)
        else:
            tau_gas = 0.

        # now calculate Rp in transmission
        Flux_obs = np.trapz(2.*np.pi*rays.Xrays[rays.id_terminator:]*np.exp(-rays.tau_end[rays.id_terminator:]-tau_gas),rays.Xrays[rays.id_terminator:])
        Flux_exp = np.pi*rays.Xrays[-1]**2.

        #Area_effective = np.trapz(2.*np.pi*rays.Xrays[rays.id_terminator:]*(1.-np.exp(-rays.tau_end[rays.id_terminator:])),rays.Xrays[rays.id_terminator:]) + np.pi * rays.Xrays[rays.id_terminator]**2.

        Rp[i] = rays.Xrays[-1]*np.sqrt(1.-Flux_obs/Flux_exp) # np.sqrt(Area_effective/np.pi) 

    return Rp

def get_eclipse_spectrum(grid,field,rays,get_Q_back,get_Q_ext,wav,no_haze_flag=False):

    # this function calculates an eclispe spectrum assuming pure back-scattering
    # either from the haze particles of Rayleigh scattering - assumes thermal emission is 
    # unimportant at wavelengths of interest. 

    Rp = np.zeros(np.size(wav))

    for i in range(np.size(wav)):

        # first task is to calculate optical depth to each point in the domain 
        # so we can consider how much stellar light is scattered back. 

        Qext_3D = 10.**get_Q_ext(np.log10(field.par_size),np.log10(wav[i]),grid=False)

        kappa_3D = 3./4. * Qext_3D / field.par_size / field.par_dens_in

        extinction_3D = kappa_3D * field.par_dens 
        extinction_par_2D = np.sum(extinction_3D,axis=2) # sum all the size bins

        gas_opacity = 2.3e-3*(wav[i]/(0.3*1e-4))**(-4.) # H2 rayleigh scattering

        if (no_haze_flag):
            extinction_total = gas_opacity * field.gas_dens
        else:
            extinction_total = extinction_par_2D + gas_opacity * field.gas_dens

        rays.do_ray_trace(extinction_total)
        rays.get_tau_grid(grid)

        # update total optical depth
        tau_b_wav = np.copy(rays.tau_b_par)

        # now need to find flux in each cell back-scattered to observer
        ## Qback is taken to be the radar backscattering efficiency / 4pi (see Bohren & Huffman)
        Qback_par = 10.**get_Q_back(np.log10(field.par_size),np.log10(wav[i]),grid=False)

        kappa_back = 3./4. * Qback_par / field.par_size / field.par_dens_in

        kappa_rayleigh = 3./(16.*np.pi) * gas_opacity * (1. + np.cos(np.pi)**2.) # Rayleigh scattering is dipole

        if (no_haze_flag):
            flux_scat_back = grid.cell_dZ * (kappa_rayleigh * field.gas_dens) * np.exp(-2.*tau_b_wav) # 2 includes optical depth of incoming and outgoing 
        else:
            flux_scat_back = grid.cell_dZ * (kappa_rayleigh * field.gas_dens + np.sum(kappa_back*field.par_dens,axis=2)) * np.exp(-2.*tau_b_wav) # 2 includes optical depth of incoming and outgoing 

        # now integrate over entire disc
        Rcyl_disc = np.outer(grid.Rb,np.sin(grid.Tb))
        Area_disc = 2.*np.pi * Rcyl_disc * grid.dRpro_back

        Total_flux_scat_back = np.sum(np.concatenate(Area_disc*flux_scat_back))

        # now use total flux back scattered to estimate radius of perfectly refelecting disc
        Rp[i] = (Total_flux_scat_back/np.pi) ** (0.5)    

    return Rp



