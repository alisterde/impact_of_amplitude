import numpy as np
import copy
import math
import os
from electrochemistry_modelling import harmonics_and_fourier_transform

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.linewidth': 1.25})


from data.dec.collection import pH4


# for electrode in ['blue', 'white', 'yellow']:
for electrode in ['yellow']:
# for electrode in ['yellow']:
    # defining model parameters
    harm_range=list(range(1, 4))
    # amplitude_vals=[50, 150, 300]
    # experiment_frequency = [8.941,8.941,8.978]
    amplitude_vals=[25,50,75,100,
                      125,150,175,200,
                      225,250,275,
                      300]
    experiment_frequency = [8.978,8.941,8.978,8.978,
                            9.015,8.941,8.978, 8.978,
                            8.941,8.978,9.015,
                            8.978]
    
    amplitude_vals=[25,75,
                    125,200
                      ,250,
                      300]
    experiment_frequency = [8.978,8.978,
                            9.015,8.978,
                            8.978, 8.978]

    # amplitude_vals=[50,150,
    #                   300]
    # experiment_frequency = [8.941,8.941,
    #                         8.978]


    # amplitude_vals=[50e-3, 100e-3, 150e-3, 200e-3, 250e-3, 300e-3]
    # fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(18, 9))
    # page_width = 25.5/2.5
    page_width = 18/2.5
    # page_width = 3.15*(3/4)
    # PAGE_lenght = int(len(harm_range)*3)
    PAGE_lenght = 6/2.5
    fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(page_width, PAGE_lenght))
    pad = 1.2
    fig.tight_layout(pad=2.0, w_pad=-0.5, h_pad=-1.8)

    # pad = 0.5
    # fig.tight_layout(pad=3.1, w_pad=0.5, h_pad=-1.3)
    # fig.tight_layout(pad=3.0, w_pad=-0.5, h_pad=-1.8)

    #plt.show()

    base_directory = os.path.dirname(os.path.realpath(__file__))

    base_input_folder = os.path.dirname(pH4.__file__)

    input_directory = os.path.join(base_input_folder, electrode)
    experiment = 1
    file_name_base = '_amp_ftacv_up_to_725_exp_'
    current_mod = '__cv_current'
    voltage_mod = '__cv_voltage'

    downsample = 400

    centre_plot = math.floor(len(amplitude_vals)/2)

    for i in range(0, len(amplitude_vals)):
        ax[0, i].set_title("$\Delta$E = {0} mV".format(amplitude_vals[i]), fontsize = 8, pad = 4.5)
        h_class=harmonics_and_fourier_transform(harmonics = harm_range, experiment_frequency = experiment_frequency[i], selection_space= 0.1)

        if electrode == 'yellow' and amplitude_vals[i] == 250:
            experiment = 2
        else:
            experiment = 1
        file_name = str(amplitude_vals[i])+file_name_base+str(experiment) + current_mod
        current_data = np.loadtxt( os.path.join(input_directory,file_name))
        times = np.asarray(current_data[:,0])
        dim_exp_current = np.asarray(current_data[:,1])
        # plot_dict=dict(non_dispersed_time_series=dim_exp_current, hanning=True, plot_func=abs, axes_list=ax[:,i])#
        plot_dict=dict(non_dispersed_time_series=dim_exp_current, hanning=False, plot_func=abs, label = 'Ferrocene _non_dispersed', alpha_iterative_drop = 0.15, axes_list=ax[:,i])
        if i ==0:
            plot_dict['ylabel']='|Current| / $\mathrm{\mu}$A'

        plot_dict['xlabel']='Time / s'
        plot_dict['micro_amps']=True

        # adding cleaned electrode

        cleaned_electrode_folder = os.path.join(input_directory, 'blank')
        clean_flie_name = file_name
        if electrode == 'blue':
            cleaned_electrode_folder = os.path.join(cleaned_electrode_folder, 'post clean')
            clean_flie_name = 'blank_' + file_name
        elif electrode == 'yellow' and amplitude_vals[i] == 250:
            experiment = 1
            file_name = str(amplitude_vals[i])+file_name_base+str(experiment) + current_mod
            clean_flie_name = file_name
        cleaned_electrode_data = np.loadtxt(os.path.join(cleaned_electrode_folder,clean_flie_name))
        cleaned_electrode_current = cleaned_electrode_data[:,1]

        plot_dict["Cleaned_time_series"]=cleaned_electrode_current
        plot_dict['label_'+'Cleaned'] = 'Cleaned_Cleaned'

        # adding blank electrode

        blank_electrode_folder =  os.path.join(base_input_folder, 'blank')
        blank_electrode_folder =  os.path.join(blank_electrode_folder, 'data')

        if amplitude_vals[i] == 300:
            file_name = str(amplitude_vals[i])+file_name_base+str(2) + current_mod

        blank_electrode_data = np.loadtxt( os.path.join(blank_electrode_folder,file_name))
        blank_electrode_current = blank_electrode_data[:,1]

        plot_dict["Blank_time_series"]=blank_electrode_current
        plot_dict['label_'+'Blank'] = 'Blank_Blank'
        # plot_dict["{0}mV_time_series".format(round(amplitude_vals[i]*1E3))]=cleaned_electrode_current
        # plot_dict['label_'+str(round(amplitude_vals[i]*1E3))]='$E^0\sigma$ = ' + str(round(amplitude_vals[i]*1E3)) + " mV_"+str(round(amplitude_vals[i]*1E3))+"mV"


        if i==centre_plot:
            # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
            plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[-0.2, 1.665], "ncol":3, 'frameon':False}
            if len(harm_range)>8:
                plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[0.5, 1.75], "ncol":3, 'frameon':False}
        else:
            plot_dict["legend"]=None

        h_class.harmonic_plotting(times, **plot_dict)
        
    
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


    # fig.tight_layout(pad=0.0)
    plt.savefig('figure_4.png', dpi=500, transparent = True)