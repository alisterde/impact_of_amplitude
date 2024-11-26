import numpy as np
import matplotlib.pyplot as plt
import pints
import os
import math
import time

from electrochemistry_modelling import wrappedNewtonCapFaradaicDip
from electrochemistry_modelling import  newtonRaphsonCapFaradaicDip

from electrochemistry_modelling import harmonics_and_fourier_transform

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.linewidth': 1.0})


# from data.ph9_2m_nacl_25oC.blanks import processed
from data.dec.collection import pH4
# from data.Alister_blank_data.DCV._500mM import graphite_electrode_alpha

# TODO:
# modify to be a forward simualtion cap only model
# check base measurements
# increase measurements and see what happens

colour_sequence = [
                '#000000','#E69F00',
                '#56B4E9','#009E73',
                '#CC79A7','#0072B2',
                '#D55E00','#F0E442']

alpha_sequence = [1.0,0.65,0.65,0.65,0.65]
# alpha_sequence = [1.0,0.5,0.5,0.5,0.5,]

if __name__ == "__main__":

    for electrode in ['simulation']:

        amplitude_vals=[50, 150, 300]
        # experiment_frequency = [1,1,1,1]
        experiment_frequency = [0.5,0.5,0.5,0.5]
        rateOfPotentialChange = [22.35E-3, 22.35E-3, 22.35E-3, 22.35E-3]
        endPureCapatianceFor =[0.15, 0.15, 0.15, 0.15]
        beingPureCapitanceto = [0.3, 0.3, 0.3, 0.3]

        startPotential = [-500.0E-3, -500.0E-3, -500.0E-3, -500.0E-3]
        revPotential =  [500.0E-3, 500.0E-3, 500.0E-3, 500.0E-3]

        fitting_range = 0.1
        nodes = 32

        # base_input_folder = os.path.dirname(pH4_2.__file__)
        # output_directory_base = input_directory

        base_input_folder = os.path.dirname(os.path.realpath(__file__))
        output_electrode_base = os.path.join(base_input_folder, electrode)


        page_width = (7.75/2.5)
        PAGE_lenght = 2.5

        for i_amp in range(0,len(amplitude_vals)):
            output_freq_base =  os.path.join(output_electrode_base,'freq_'+str(int(experiment_frequency[i_amp])))
            output_amp_base = os.path.join(output_freq_base,'amplitude_'+str(amplitude_vals[i_amp]))

            plt.figure(figsize=(page_width, PAGE_lenght))
            # plt.tight_layout(pad=2.2 , w_pad=-0.5 , h_pad=0.0)
    
            # exp_to_fit = [1]
            if electrode == 'yellow' and amplitude_vals[i_amp] == 250:
                exp_to_fit = [2]
            else:
                exp_to_fit = [1]
                    
            for experiment in exp_to_fit:
                output_exp_base = os.path.join(output_amp_base, 'sim_' + str(experiment))
                if not os.path.exists(output_exp_base):
                    os.makedirs(output_exp_base)

                output_folder = os.path.join(output_exp_base, 'varying_dispersion')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                index = 0

                for dispersion_wanted in [False, True]:
                    # for thermo_dispersion in [25E-3, 50E-3, 60E-3, 70E-3]:
                    for thermo_dispersion in [50E-3]:
                        
                        kinetics = 10E2
                        experimental_resistance = 0.0
                        helmholtz_cap = 0.0e-05

                        measurements = 200000
                        startT = 0.0#specify in seconds
                        revT =  abs((revPotential[i_amp] - startPotential[i_amp])/(rateOfPotentialChange[i_amp]))#specify in seconds
                        endT = revT*2.0
                        first_half_times = np.linspace(startT, revT, measurements)
                        last_half_times = np.linspace(revT, endT, measurements)
                        times = np.hstack((first_half_times, last_half_times[1:]))

                        length = times.shape
                        numberOfMeasurements = int(length[0])
                        half_numberOfMeasurements = int(numberOfMeasurements/2)

                        timeStepSize = times[-1]/(numberOfMeasurements - 1)

                        dip_sim = np.load(os.path.join(output_folder,'theta_ox_list_Dispersion_is_'+str(thermo_dispersion)+'_dip_is_'+str(dispersion_wanted)+'_amplitude_'+str(int(amplitude_vals[i_amp]))+ '.npy'))
                        appliedPotentialDCVComponetent = np.load(os.path.join(output_folder,'appliedPotentialDCVComponetent_Dispersion_is_'+str(thermo_dispersion)+'_dip_is_'+str(dispersion_wanted)+'_amplitude_'+str(int(amplitude_vals[i_amp]))+ '.npy'))
                        
                        if dispersion_wanted == True:
                            label = '$\mathrm{E^{0}_{\sigma} = }$'+str(int(thermo_dispersion*1000))+' mV'
                        elif dispersion_wanted == False:
                            label = '$\mathrm{E^{0}_{\sigma} = }$'+str(int(0))+' mV'
                       
                        plt.plot(times[:half_numberOfMeasurements],dip_sim[:half_numberOfMeasurements], color = colour_sequence[index],
                                 label = label, alpha = alpha_sequence[index])
                        
                        # if amplitude_vals[i_amp] == 300:
                        #     index =+ 1
                        # if amplitude_vals[i_amp] == 150:
                        #     index =+ 2
                        # if amplitude_vals[i_amp] == 50:
                        #     index =+ 3
                        index =+ 1

                        

            # plt.title('$\Delta E = $'+str(amplitude_vals[i_amp])+' mV', pad= 0)   
            if amplitude_vals[i_amp] == 300:
                plt.title('(d)', loc ='left')
            if amplitude_vals[i_amp] == 150:
                plt.title('(c)', loc ='left')
            if amplitude_vals[i_amp] == 50:
                plt.title('(b)', loc ='left')
            # plt.xlabel("Time \ s")
            plt.xlabel("Time / s")
            plt.ylabel('$\mathrm{\Theta_{ox}}$')
            plt.legend(loc ="upper center", bbox_to_anchor = [0.53, 1.17], frameon =False, ncol = 2, columnspacing = 5, labelspacing = 0.0, handlelength =  0.75)
            # plt.legend(loc='best',  frameon = False, columnspacing = 0.3, labelspacing = 0.0, handlelength = 1.0)
            plt.tight_layout()
            if amplitude_vals[i_amp] == 300:
                plt.savefig('figure_3_d.png', dpi = 500, transparent = True)
            if amplitude_vals[i_amp] == 150:
                 plt.savefig('figure_3_c.png', dpi = 500, transparent = True)
            if amplitude_vals[i_amp] == 50:
                 plt.savefig('figure_3_b.png', dpi = 500, transparent = True)