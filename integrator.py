import numpy as np
import particle_updates as pu 
import boundary as boundary
import source as source
import copy

# for profiling
from timeit import default_timer as timer


#### written by JO Dec 2020 ###'#
#### This contains a spherical finite volume grid 
#### This grid is based on the staggered grids, ala zeus (Stone & Norman 1992)
#### The grid will include one ghost cell (2nd order method)
#### This file contains integrator routines (1st and 2nd order)

def first_order_update(CFL,grid,field,rays,system,boundary_args=(),source_args=()):

    # this routine performs a first-order in time explicit update

    # calculate particle velocities
    call = np.array([timer()])
    pu.get_par_acc(grid,field,rays,system)
    call = np.append(timer())
    pu.get_par_vel_terminal_velocity(grid,field,system)
    call = np.append(timer())
    pu.get_tgrow(grid,field,system)
    call = np.append(timer())

    # get timestep
    dt = pu.get_timestep(CFL,grid,field)
    call = np.append(timer())
    # growth update
    pu.size_update(grid,field,dt)
    call = np.append(timer())
    # do source update

    source.update_source(grid,field,system,dt,source_args)
    call = np.append(timer())
    # now do advection update

    pu.advection_update(grid,field,dt)
    call = np.append(timer())
    # boundary update

    boundary.update_boundary(grid,field,boundary_args)
    call = np.append(timer())
    # now do diffusion update

    pu.diffusion_update(grid,field,dt)
    call = np.append(timer())
    # boundary update

    boundary.update_boundary(grid,field,boundary_args)
    call = np.append(timer())
    return dt, call

def first_order_update_semi_implict(dt,grid,field,rays,system,boundary_args=(),source_args=(),getQ=None):

    # this routine performs a first-order in time explicit update with semi-implicit update of particle
    # velocity

    # calculate particle velocities
    pu.get_par_acc(grid,field,rays,system,getQ)
    pu.get_velocity_semi_implicit(grid,field,system,dt)
    pu.get_tgrow(grid,field,system)

    # growth update
    pu.size_update(grid,field,dt)

    # do source update

    source.update_source(grid,field,system,dt,source_args)

    # now do advection update

    pu.advection_update(grid,field,dt)
 
    # boundary update

    boundary.update_boundary(grid,field,boundary_args)

    # now do diffusion update

    pu.diffusion_update(grid,field,dt)
 
    # boundary update

    boundary.update_boundary(grid,field,boundary_args)

    return dt

def first_order_update_semi_implict_numba(dt,grid,field,rays,system,boundary_args=(),source_args=(),get_Q=None):

    # this routine performs a first-order in time explicit update with a possible semi-implicit update of particle
    # velocity - it includes parallel numba optimization for use on large grids
    # the boundary update should be fast enough in pure numpy

    # calculate particle velocities
    pu.get_par_acc_jd(grid,field,rays,system,get_Q)
    pu.get_velocity_semi_implicit_jd(grid,field,system,dt)
    pu.get_tgrow_jd(grid,field,system)

    # growth update
    pu.size_update_jd(grid,field,dt)

    # do source update

    source.update_source_jd(grid,field,rays,system,dt,source_args)

    # now do advection updates

    if (field.short_friction):

        ### just density transport
        pu.advection_update_jd(grid,field,dt)
    else:
        ## Diffusive updated is included as velocity term here - velocities computed in size update
        # as velocities have changed due to source update need to update boundries
        boundary.update_boundary(grid,field,boundary_args)
        ## compute momenta
        pu.compute_momenta_jd(grid,field)
        ## do density advenction
        pu.advection_update_jd(grid,field,dt)
        ## do momentum advection
        pu.momentum_advection_update_jd(grid,field,dt)
        ## recalculate velocities
        pu.recalculate_velocities_jd(grid,field)

 
    # boundary update

    if (field.short_friction):
        # do seperate diffusion update - otherwise included in advection as velocity
        boundary.update_boundary(grid,field,boundary_args)

        # now do diffusion update
        pu.diffusion_update_jd(grid,field,dt)
 
    # boundary update

    boundary.update_boundary(grid,field,boundary_args)

    return dt


def runner_semi_implicit_numba(CFL,Nsteps,Ndump,Noptical_depth,initial_dt,grid,field,rays,system,boundary_args=(),source_args=(),get_Qpr=None,get_Qext=None):

    # this is a runner function
    # it runs for Nsteps outputing to file every Ndump steps
    # it updates the optical depth due to dust every Noptical_depth steps

    dt = copy.deepcopy(initial_dt)
    sim_time = 0.
    counter = 0
    output_counter = 0
    output_tuple = [sim_time,dt,counter,field,rays,grid,system] # grid and system are appended to intial output
    np.save("output%0d" %(output_counter),output_tuple,allow_pickle=True)

    while counter < Nsteps:
        first_order_update_semi_implict_numba(dt,grid,field,rays,system,boundary_args=boundary_args,source_args=source_args,get_Q=get_Qpr)
        sim_time += dt
        dtnew = pu.get_timestep_jd(CFL,grid,field)
        if (dtnew > 1.25 * dt):
            dt = 1.25 * dt
        else:
            dt = dtnew
        counter +=1

        if (counter % Noptical_depth == 0):
            # update optical depth
            pu.update_tau_b_par(field,grid,rays,get_Qext)
        # check for output
        if (counter % Ndump == 0):
            output_counter +=1
            output_tuple = [sim_time,dt,counter,field,rays]
            np.save("output%0d" %(output_counter),output_tuple,allow_pickle=True)

    return sim_time, dt
