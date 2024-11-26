import numpy as np
import copy
import math
import os
from electrochemistry_modelling import harmonics_and_fourier_transform

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.linewidth': 1.5})


from data.dec.collection import pH4


# for electrode in ['blue', 'white', 'yellow']:
for electrode in ['yellow']:
# for electrode in ['yellow']:
    harm_range=list(range(2, 9))

    amplitude_vals=[50,150, 300]
    experiment_frequency = [8.941,8.941,8.978]

    page_width = 16/2.5 #3.15*(3/4)*2  6.69291
    PAGE_lenght = 15/2.5
    fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(page_width, PAGE_lenght))
    pad = 0.5
    # fig.tight_layout(pad=3.1, w_pad=0.5, h_pad=-1.3)
    fig.tight_layout(pad=3.27, w_pad=-1.8, h_pad=-1.0)

    #plt.show()

    base_directory = os.path.dirname(os.path.realpath(__file__))

    base_input_folder = os.path.dirname(pH4.__file__)

    input_directory = os.path.join(base_input_folder, electrode)
    experiment = 1

    file_name_base = '_amp_ftacv_up_to_725_exp_'
    # current_mod = '_dec_1_8_cv_current'
    # voltage_mod = '_dec_1_8_cv_voltage'
    current_mod = '__cv_current'
    voltage_mod = '__cv_voltage'

    downsample = 400

    # line_style =[
    #     'dashed', 'dashed',
    #     'dashed', 'dashed',
    #     'dashed', 'solid',
    #     'dashed', 'dashed'
    # ]
    line_style =[
        'solid', 'solid',
        'solid', 'solid',
        'solid', 'solid',
        'solid', 'solid'
    ]

    colour_sequence = [
                '#CC79A7','#E69F00',
                '#56B4E9','#009E73', '#0072B2',
                '#000000',
                '#CC79A7','#0072B2',
                '#D55E00','#F0E442']

    centre_plot = math.floor(len(amplitude_vals)/2)

    # selection_space = 0.1
    selection_space = 0.35



    for i in range(0, len(amplitude_vals)):

        # ax.set_title('Title', pad=20)
        ax[0, i].set_title("$\Delta$E = {0} mV".format(amplitude_vals[i]), pad=1)
        h_class=harmonics_and_fourier_transform(harmonics = harm_range, experiment_frequency = experiment_frequency[i], selection_space= selection_space)

        if electrode == 'yellow' and amplitude_vals[i] == 250:
            experiment = 2
        else:
            experiment = 1
        file_name = str(amplitude_vals[i])+file_name_base+str(experiment) + current_mod
        current_data = np.loadtxt( os.path.join(input_directory,file_name))
        times = np.asarray(current_data[:,0])
        dim_exp_current = np.asarray(current_data[:,1])

        # simulated data
        input_simulations_base_directory = os.path.dirname(os.path.realpath(__file__))
        # input_simulations_base_directory = os.path.join(input_simulations_base_directory, electrode)
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'dispersion_with_resistance_64_bins')
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'amplitude_'+str(amplitude_vals[i]))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'exp_' + str(experiment))
        # input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'dispersion_with_resistance')
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'varying_dispersion')

        # first plot dip free model

        dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(25e-3)+'_dip_is_'+str(False)+'.npy'))

        # plot_dict=dict(non_dispersed_time_series=dim_exp_current, hanning=True, plot_func=abs, axes_list=ax[:,i])#
        plot_dict=dict(non_dispersed_time_series=dip_sim, hanning=False, plot_func=abs, colour_sequence = colour_sequence, line_style = line_style, label = '$E^0_\sigma$ = 0_non_dispersed', alpha_iterative_drop = 0.0, axes_list=ax[:,i])
        if i ==0:
            plot_dict['ylabel']='|Current| / $\mathrm{\mu}$A'

        plot_dict['xlabel']='Time / s'
        plot_dict['micro_amps']=True

        # adding simulated plots
        
        dispersion_list = [25e-3, 50e-3, 60e-3, 70e-3]
        # dispersion_list = [25e-3, 50e-3, 60e-3]


        # if amplitude_vals[i] == 50:
        #     dispersion_list = [25e-3, 50e-3, 60e-3]
        # else:
        #     dispersion_list = [25e-3, 50e-3, 60e-3, 70e-3]

        for dispersion in dispersion_list:

            if dispersion == 0:
                dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(25e-3)+'_dip_is_'+str(False)+'.npy'))
            else:
                dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(dispersion)+'_dip_is_'+str(True)+'.npy'))
            # dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(dispersion)+'_dip_is_'+str(True)+'.npy'))

            plot_dict["{0}mV_time_series".format(int(dispersion*1E3))]=dip_sim
            plot_dict['label_'+str(int(dispersion*1E3))]='$E^0_\sigma$ = ' + str(int(dispersion*1E3)) + " $\mathrm{mV}$_"+str(int(dispersion*1E3))+"mV"
            if i==1:
                # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
                plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[1.15, 1.875], "ncol":len(dispersion_list)+1,
                                    'frameon':False,'columnspacing':0.3, 'handlelength': 1.0}
            else:
                plot_dict["legend"]=None

        # adding experimental data

        plot_dict["{0}mV_time_series".format('Ferrocene')]=dim_exp_current
        plot_dict['label_'+str('Ferrocene')]='Ferrocene_FerrocenemV'

        if i==centre_plot:
            # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
            plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[0.5, 1.65], "ncol":6,
                                 'frameon':False, 'columnspacing':0.3, 'labelspacing':0.0, 'handlelength': 1.0}
            if len(harm_range)>8:
                plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[0.5, 1.8], "ncol":3,
                                     'frameon':False, 'columnspacing':0.3, 'labelspacing':0.0, 'handlelength': 1.0}
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
    # plt.show()
    plt.savefig('figure_7.png', dpi=500, transparent = True)


    # fig, ax=plt.subplots(len(harm_range), len(amplitude_vals), figsize=(page_width, PAGE_lenght))
    # pad = 0.5
    # # fig.tight_layout(pad=3.1, w_pad=0.5, h_pad=-1.3)
    # fig.tight_layout(pad=3.27, w_pad=-1.8, h_pad=-1.0)

    # #plt.show()

    # base_directory = os.path.dirname(os.path.realpath(__file__))

    # base_input_folder = os.path.dirname(pH4_2.__file__)

    # input_directory = os.path.join(base_input_folder, electrode)
    # experiment = 1

    # file_name_base = '_amp_ftacv_up_to_725_exp_'
    # # current_mod = '_dec_1_8_cv_current'
    # # voltage_mod = '_dec_1_8_cv_voltage'
    # current_mod = '__cv_current'
    # voltage_mod = '__cv_voltage'

    # downsample = 400

    # # line_style =[
    # #     'dashed', 'dashed',
    # #     'dashed', 'dashed',
    # #     'dashed', 'solid',
    # #     'dashed', 'dashed'
    # # ]
    # line_style =[
    #     'solid', 'solid',
    #     'solid', 'solid',
    #     'solid', 'solid',
    #     'dashed', 'dashed'
    # ]

    # colour_sequence = [
    #             '#CC79A7','#E69F00',
    #             '#56B4E9','#009E73', '#0072B2',
    #             '000000',
    #             '#CC79A7','#0072B2',
    #             '#D55E00','#F0E442']

    # centre_plot = math.floor(len(amplitude_vals)/2)


    # for i in range(0, len(amplitude_vals)):
    #     if amplitude_vals[i] == 50:
    #         colour_sequence = [
    #             '#CC79A7','#E69F00',
    #             '#56B4E9','#009E73', '#000000',
    #             '000000',
    #             '#CC79A7','#0072B2',
    #             '#D55E00','#F0E442']
    #     else:
    #         colour_sequence = [
    #             '#CC79A7','#E69F00',
    #             '#56B4E9','#009E73', '#0072B2',
    #             '000000',
    #             '#CC79A7','#0072B2',
    #             '#D55E00','#F0E442']

    #     # ax.set_title('Title', pad=20)
    #     ax[0, i].set_title("$\Delta$E = {0} mV".format(amplitude_vals[i]), pad=1)
    #     h_class=harmonics_and_fourier_transform(harmonics = harm_range, experiment_frequency = experiment_frequency[i], selection_space= selection_space)

    #     if electrode == 'yellow' and amplitude_vals[i] == 250:
    #         experiment = 2
    #     else:
    #         experiment = 1
    #     file_name = str(amplitude_vals[i])+file_name_base+str(experiment) + current_mod
    #     current_data = np.loadtxt( os.path.join(input_directory,file_name))
    #     times = np.asarray(current_data[:,0])
    #     dim_exp_current = np.asarray(current_data[:,1])

    #     # simulated data
    #     input_simulations_base_directory = os.path.dirname(os.path.realpath(__file__))
    #     input_simulations_base_directory = os.path.join(input_simulations_base_directory, electrode)
    #     input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'amplitude_'+str(amplitude_vals[i]))
    #     input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'exp_' + str(experiment))
    #     input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'dispersion_with_resistance')

    #     # first plot dip free model

    #     dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(25e-3)+'_dip_is_'+str(False)+'.npy'))

    #     # plot_dict=dict(non_dispersed_time_series=dim_exp_current, hanning=True, plot_func=abs, axes_list=ax[:,i])#
    #     plot_dict=dict(non_dispersed_time_series=dip_sim, hanning=False, plot_func=abs, colour_sequence = colour_sequence, line_style = line_style, label = '$E^0_\sigma$ = 0_non_dispersed', alpha_iterative_drop = 0.1, axes_list=ax[:,i])
    #     if i ==0:
    #         plot_dict['ylabel']='|Current| / $\mathrm{\mu}$A'

    #     plot_dict['xlabel']='Time / s'
    #     plot_dict['micro_amps']=True

    #     # adding simulated plots
        
    #     # dispersion_list = [25e-3, 50e-3, 60e-3, 70e-3]

    #     # if amplitude_vals[i] == 50:
    #     #     dispersion_list = [25e-3, 50e-3, 60e-3]
    #     # else:
    #     #     dispersion_list = [25e-3, 50e-3, 60e-3, 70e-3]

    #     for dispersion in dispersion_list:

    #         if dispersion == 0:
    #             dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(25e-3)+'_dip_is_'+str(False)+'.npy'))
    #         else:
    #             dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(dispersion)+'_dip_is_'+str(True)+'.npy'))
    #         # dip_sim = np.load(os.path.join(input_simulations_base_directory,'Dispersion_is_'+str(dispersion)+'_dip_is_'+str(True)+'.npy'))

    #         plot_dict["{0}mV_time_series".format(int(dispersion*1E3))]=dip_sim
    #         plot_dict['label_'+str(int(dispersion*1E3))]='$E^0_\sigma$ = ' + str(int(dispersion*1E3)) + " $\mathrm{mV}$_"+str(int(dispersion*1E3))+"mV"
    #         if i==1:
    #             # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
    #             plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[1.15, 1.875], "ncol":len(dispersion_list)+1,
    #                                 'frameon':False,'columnspacing':0.3, 'handlelength': 1.0}
    #         else:
    #             plot_dict["legend"]=None

    #     # adding experimental data

    #     plot_dict["{0}mV_time_series".format('Ferrocene')]=dim_exp_current
    #     plot_dict['label_'+str('Ferrocene')]='Ferrocene_FerrocenemV'

    #     if i==centre_plot:
    #         # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
    #         plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[0.5, 1.65], "ncol":6,
    #                              'frameon':False, 'columnspacing':0.3, 'labelspacing':0.0, 'handlelength': 1.0}
    #         if len(harm_range)>8:
    #             plot_dict["legend"]={"loc":"upper center", "bbox_to_anchor":[0.5, 1.8], "ncol":3,
    #                                  'frameon':False, 'columnspacing':0.3, 'labelspacing':0.0, 'handlelength': 1.0}
    #     else:
    #         plot_dict["legend"]=None

    #     h_class.freq_domain_harmonic_plotting(times, **plot_dict) 

        
    #     indexes = len(harm_range)*len(amplitude_vals)
    #     harm_index = 0
    #     for index in range(1,1+indexes):
    #         if index%len(amplitude_vals) == 0:
    #             plt.subplot(len(harm_range), len(amplitude_vals),index)
    #             ax1=plt.gca()
    #             ax2 = ax1.twinx()
    #             ax2.set_yticks([])
    #             ax2.set_ylabel('Harmonic ' + str(harm_range[harm_index]), labelpad=8.0, rotation=-90)
    #             harm_index += 1

       

    # # fig.tight_layout(pad=0.0)
    # # plt.show()
    # plt.savefig('variation_of_dispersion_freq.png', dpi=500)