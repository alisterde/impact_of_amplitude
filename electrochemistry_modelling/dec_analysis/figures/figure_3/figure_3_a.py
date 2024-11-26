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
                '#000000','#CC79A7',
                '#56B4E9','#009E73',
                '#E69F00','#0072B2',
                '#D55E00','#F0E442']

# colour_sequence = ['#E69F00',
#                 '#56B4E9','#009E73',
#                 '#CC79A7','#0072B2',
#                 '#D55E00','#F0E442']

alpha_sequence = [1.0,0.825, 0.75]
if __name__ == "__main__":

    for electrode in ['simulation']:

        amplitude_vals=[300, 150, 50]
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
        # fig, ax=plt.subplots(1, 2, figsize=(page_width, PAGE_lenght))
        
        # bewteen_subplot_pad = 0.5# 3.0
        # boundary_pad = 5.1
        # fig.tight_layout(pad=2.2 , w_pad=-0.5 , h_pad=0.0)

        plt.figure(figsize=(page_width, PAGE_lenght))
        # plt.tight_layout(pad=6)

        # ploting 2 standard devaitons for 25 mV shaded from y axis about zero


        for i_amp in range(0,len(amplitude_vals)):
            output_freq_base =  os.path.join(output_electrode_base,'freq_'+str(int(experiment_frequency[i_amp])))
            output_amp_base = os.path.join(output_freq_base,'amplitude_'+str(amplitude_vals[i_amp]))
    
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

                for dispersion_wanted in [False]:
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
                        appliedPotential = np.load(os.path.join(output_folder,'appliedPotential_Dispersion_is_'+str(thermo_dispersion)+'_dip_is_'+str(dispersion_wanted)+'_amplitude_'+str(int(amplitude_vals[i_amp]))+ '.npy'))

                        if i_amp == 0:
                            std = 50
                            drop = 6000

                            x = times[:half_numberOfMeasurements]
                            y_base = np.ones_like(x)
                            y1 = y_base* 2 * (std - 1)
                            y2 = y_base* 2 * (std)
                            plt.plot(x[drop:-drop], y1[drop:-drop], color = colour_sequence[1 +len(amplitude_vals)], linestyle = 'dashed', label = '$2\mathrm{E^0_\sigma} = $'+str(std*2) + ' mV')
                            plt.plot(x[drop:-drop], -y2[drop:-drop], color = colour_sequence[1 +len(amplitude_vals)], linestyle = 'dashed', label = None)
                            # y1 = std
                            # y2 = - std

                            # plt.fill_between(x, 1000*y1, 1000*y2, color = colour_sequence[-4], alpha = 1.0, label = '2$\mathrm{E^0_\sigma} = $'+'50 mV')
                                        
                        plt.plot(x, 1000*appliedPotential[:half_numberOfMeasurements], color = colour_sequence[i_amp], label = '$\mathrm{\Delta E} = $'+str(amplitude_vals[i_amp]) + ' mV', alpha = alpha_sequence[i_amp])


        # plt.tight_layout(pad=2)
        # plt.title('Applied Potential With Varying $\mathrm{\Delta E}$\n With The Region of $2\mathrm{E^0_\sigma}$ shaded')
        plt.title('(a)', loc = 'left')
        plt.ylabel('Applied Potential / mV')
        # plt.xlabel('Time / s')
        plt.xlabel("Time / s")
        plt.errorbar(x =x[0], y = 0, yerr=2*std, color = colour_sequence[1+len(amplitude_vals)], capsize=3)
        plt.errorbar(x =x[-1], y = 0, yerr=2*std, color = colour_sequence[1+len(amplitude_vals)], capsize=3)
        plt.legend(loc='best',  frameon = False, columnspacing = 0.3, labelspacing = 0.0, handlelength = 1.0)
        plt.tight_layout()
        plt.savefig('figure_3_a.png', dpi = 500, transparent = True)
        # plt.savefig('new_plot.png', dpi = 500)
        # plt.show()
                            # plt.close()

                        # plt.subplot(1, 4,2)
                        # ax=plt.gca()
                        # # plt.title("K_ox")
                        # ax.set_title("$\Delta E = $150 mV", pad= 2)
                        # ax.set_xlabel("Time/s", labelpad= 2)
                        # ax.set_ylabel('$\mathrm{k_{red}/s^{-1}}$', labelpad= 2)
                        # ax.plot(times, solved_low_amp_dip[0][k_red_end_drop:-k_red_end_drop], color = colour_sequence[2], label = string_3_2, alpha = 0.6)
                        # ax.plot(times, solved_low_amp_no_dip[0][k_red_end_drop:-k_red_end_drop], color = colour_sequence[0], label = string_3_1, alpha = 0.4)
