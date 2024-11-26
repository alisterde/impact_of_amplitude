import numpy as np
import copy
import math
import os
from electrochemistry_modelling import harmonics_and_fourier_transform
from electrochemistry_modelling import wrappedNewtonCapFaradaicDip

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'lines.linewidth': 3})

if __name__ == "__main__":
     
    # declaring fake times
    # delacring know model paramters


    rateOfPotentialChange = 22.35E-3
    freq = 8.88
    # freq = 1
    freq_name = 'high'

     # for deltaepislon in [50.0E-3, 150.0E-3, 300.0E-3, 1000.0E-3]:
    for deltaepislon in [50.0E-3, 300.0E-3]:
    # for deltaepislon in [150.0E-3]:
    # for deltaepislon in [150.0E-3]:

        electrode_area = 0.07


        inital_current =  -7.735012e-08
        # startPotential= -325.00E-3
        startPotential= 500.00E-3
        inital_current = startPotential
        # revPotential = 725.00E-3
        revPotential = -500.00E-3

        numberConc = 6.0221409e+23*0.1*10**-3 # 6.0221409e+23*0.01*10**6
        #6.0221409e+23*2*10**6 #i.e a 2 molar solution converting to per cm^3

        measurements = 200000
        startT = 0.0#specify in seconds
        revT =  abs((revPotential - startPotential)/(rateOfPotentialChange))#specify in seconds
        endT = revT*2.0
        first_half_times = np.linspace(startT, revT, measurements)
        last_half_times = np.linspace(revT, endT, measurements)
        times = np.hstack((first_half_times, last_half_times[1:]))

        length = times.shape
        numberOfMeasurements = int(length[0])

        timeStepSize = times[-1]/(numberOfMeasurements - 1)

        fit_alpha = False
        nodes = 32

        
        protocol = 'FTACV'

        model = wrappedNewtonCapFaradaicDip(times, startPotential=startPotential, revPotential=revPotential,
                                rateOfPotentialChange=rateOfPotentialChange, deltaepislon = deltaepislon,
                                inital_current=inital_current, protocol= protocol, electrode_area = electrode_area,
                                numberConc=numberConc, freq = freq, nodes = nodes)
        cap_options_avaliable = ['doublePoly', '3rd_order_poly', 'helmholtz', 'gouyChapman', 'stern', '2dPoly', '1st_order_poly', '2nd_order_poly', '4th_order_poly', '5th_order_poly']
        cap = cap_options_avaliable[1]
        model.set_cap_model(cap)
        cap_parameters = np.asarray(model.suggested_capacitance_params())
        # output = ['gamma0', 'gamma1', 'gamma2', 'gamma3','resistance', 'omega', 'phase']
        cap_parameters = [0,0,0,0,0,freq,0]
        cap_parameters[-2] = freq
        model.set_capacitance_params(cap_parameters)
        modelling_faradaic_current = True
        model.set_modelling_faradaic_current(modelling_faradaic_current)
        model.set_modeling_experimental_data(False)
        model.set_dispersion_modeling(e_dip=False)

        # model.set_solver_option('scipy_newton_raphson')
        # model.set_solver_option('my_riddlers')
        model.set_solver_option('my_newton_raphson')

        # parameters[1] = 550.0
        # print('parameters: ', parameters)

        parameters = model.suggested_parameters()
        k_is = 1.00E5
        # ['kappa0', 'epsilon0', 'mu', 'resistance', 'surface_coverage']
        parameters = [k_is, 0.0, 0.0, 0.0, 1E-10]
        parameters[0] = k_is
        print('dip free parameters: ', parameters)
        og_i = model.simulate(parameters, times)

        non_dim_constants = model.get_non_dimensionality_constants()
        I0 = non_dim_constants[-1]
        og_i = og_i*I0

        np.save(arr=og_i, file ='amp_'+str(round(deltaepislon*1000))+'_e_dip_free_current_k_is_'+str(int(parameters[0])))
        np.save(arr=times, file = 'amp_'+str(round(deltaepislon*1000))+'_e_dip_times_k_is_'+str(int(parameters[0])))
        og_appliedPotential = model.appliedPotential
        og_drivAppliedPotential = model.drivAppliedPotential

        model.set_dispersion_modeling(e_dip=True)
        # parameters = model.suggested_parameters()
        plt.figure()
        plt.plot(times[:], og_i[:], label = 'dispersion_free')

        # for k_dip in [2,12]:
        for e_dip in [20.0,50.0]:
            parameters = [k_is, 0.0, 0.0, 0.0, 1E-10, e_dip*1E-3]
            print('dip parameters: ', parameters)
            e_dip_model_i = model.simulate(parameters, times)
            e_dip_model_i = e_dip_model_i*I0
            np.save(arr=e_dip_model_i, file = 'amp_'+str(round(deltaepislon*1000))+'_e_dip_is_'+str(int(e_dip))+'_current_k_is_'+str(int(parameters[0])))

    
            plt.plot(times[:], e_dip_model_i[:], label = 'dip_'+str(int(e_dip)))
        plt.xlabel("times/s")
        plt.ylabel("current/A")
        plt.legend(loc = 'best')
        plt.savefig('amp_'+str(round(deltaepislon*1000))+'_k_is_'+str(int(parameters[0])))
        plt.close()