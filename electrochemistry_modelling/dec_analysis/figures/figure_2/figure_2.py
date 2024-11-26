import numpy as np
import copy
import math
import os
from electrochemistry_modelling import harmonics_and_fourier_transform


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'lines.linewidth': 0.75})


# defining model parameters
harm_range=list(range(1, 8))
# amplitude_vals=[50e-3, 150e-3, 300e-3, 1000E-3]
amplitude_vals=[50e-3, 300e-3]


# for parameters in [1e2, 5e2, 1e3, 1e4,1e5]:
for parameters in [1e5]:
        
    # amplitude_vals=[50e-3, 100e-3, 150e-3, 200e-3, 250e-3, 300e-3]
    # fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(18, 9))
    # fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(18, (9/4)*7))
    fig_width = (16.5/2)/2.5
    fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(fig_width, 3.75))
    # pad = 1.2
    # fig.tight_layout(pad=4.7, w_pad=pad, h_pad=pad)

    pad = 0.5
    fig.tight_layout(pad=3.0, w_pad=-1.5, h_pad=-1.8)

    # std_dev_vals_unit = [2.0, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0]
    # std_dev_vals_unit = [2.0, 4.0, 6.0, 10.0, 12.0]
    # std_dev_vals = np.log(std_dev_vals_unit)
    std_dev_vals = [20, 50]

    #plt.show()

    base_directory = os.path.dirname(os.path.realpath(__file__))

    grayscale_colour_list = ['#000000','#666666','#999999']
    selection_space = 0.35

    for i in range(0, len(amplitude_vals)):
        ax[0, i].set_title("$\Delta$E = {0} mV".format(int(1000*amplitude_vals[i])), fontsize = 6)
        h_class=harmonics_and_fourier_transform(harmonics = harm_range, experiment_frequency = 8.88, selection_space= selection_space)

        current_file = 'amp_'+str(round(amplitude_vals[i]*1000))+'_e_dip_free_current_k_is_'+str(int(parameters))
        time_file = 'amp_'+str(round(amplitude_vals[i]*1000))+'_e_dip_times_k_is_'+str(int(parameters))
            
        # data = np.load( os.path.join(base_directory,file_name+'.npy'))
        time = np.load(os.path.join(base_directory, time_file+'.npy'))
        time = time[:]
        current = np.load(os.path.join(base_directory, current_file+'.npy'))
        current = current[:]
        # plot_dict=dict(non_dispersed_time_series=current, hanning=False, plot_func=abs, axes_list=ax[:,i])#
        plot_dict=dict(non_dispersed_time_series=current, hanning=False, plot_func=abs, label = 'Non Dispersed_non_dispersed', alpha_iterative_drop = 0.0, axes_list=ax[:,i])
        if i ==0:
            plot_dict['ylabel']='|Current| / $\mathrm{\mu}$A'

        plot_dict['xlabel']='Time / $\mathrm{s}$'
        plot_dict['micro_amps']=True
        for j in range(0, len(std_dev_vals)):
            
            # file_name = 'dip_curves_dip_' + str(round(std_dev_vals[j]*1E3)) + '_amp_' + str(round(amplitude_vals[i]*1E3))
            file_name = 'amp_'+str(round(amplitude_vals[i]*1000))+'_e_dip_is_'+str(int(std_dev_vals[j]))+'_current_k_is_'+str(int(parameters))
            # dispersed_data = np.load(file_name+'.npy')
            dispersed_current =  np.load(os.path.join(base_directory, file_name+'.npy'))
            dispersed_current = dispersed_current[:]

            plot_dict["{0}mV_time_series".format(round(std_dev_vals[j]*1E3))]=dispersed_current
            plot_dict['label_'+str(round(std_dev_vals[j]*1E3))]='$E^0_\sigma$ = ' + str(round(std_dev_vals[j],3)) + " $\mathrm{mV}$_"+str(round(std_dev_vals[j]*1E3))+"mV"
        if i==1:
            # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
            plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[1.15, 1.95], "ncol":len(std_dev_vals)+1,
                                 'frameon':False,'columnspacing':0.3, 'handlelength': 1.0}
        else:
            plot_dict["legend"]=None
        h_class.harmonic_plotting(time, **plot_dict)

    indexes = len(harm_range)*len(amplitude_vals)
    harm_index = 0
    for index in range(1,1+indexes):
        if index%len(amplitude_vals) == 0:
            plt.subplot(len(harm_range), len(amplitude_vals),index)
            ax1=plt.gca()
            ax2 = ax1.twinx()
            ax2.set_yticks([])
            if harm_range[harm_index] == 1:
                ax2.set_ylabel(str(harm_range[harm_index]) +'st', labelpad=12.0, rotation=0 )
            elif harm_range[harm_index] == 2:
                ax2.set_ylabel(str(harm_range[harm_index]) +'nd', labelpad=12.0, rotation=0 )
            elif harm_range[harm_index] == 3:
                ax2.set_ylabel(str(harm_range[harm_index]) +'rd', labelpad=12.0, rotation=0 )
            else:
                ax2.set_ylabel(str(harm_range[harm_index]) +'th', labelpad=12.0, rotation=0 )
            harm_index += 1
            ax1.tick_params(axis='y', which='major', pad=0.0, length = 0)
    # fig.tight_layout(pad=0.0)
    # plt.savefig('figure_2.png', dpi=500, transparent = True)
    plt.savefig('figure_2.png', dpi=500, transparent = False)
    plt.close()