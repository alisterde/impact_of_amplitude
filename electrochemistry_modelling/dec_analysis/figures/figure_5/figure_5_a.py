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
# plt.rcParams.update({'yticks.rotation': 45})


from data.dec.collection import pH4




# for electrode in ['blue', 'white', 'yellow']:
for electrode in ['yellow']:
# for electrode in ['yellow']:
    harm_range=list(range(1, 3))

    amplitude_vals=[50,300]
    experiment_frequency = [8.941,8.978]

    # page_width = 3.15*(3/4)*2
    page_width = 6.69291
    PAGE_lenght = 2.5/2
    total_sub_plots =  len(amplitude_vals)* (1 + len(harm_range))+1
    print('total_sub_plots: ', total_sub_plots)
    # gridspec = dict(hspace=0.0, height_ratios=[1, 1, 1, 1, 1, 3])
    gridspec = dict(width_ratios=[1, 1, 1,0.1, 1, 1,1])
    fig, ax=plt.subplots(1, total_sub_plots, figsize=(page_width, PAGE_lenght), gridspec_kw=gridspec)
    ax[3].set_visible(False)
    # ax[3].axis('off')
    # ax[3].axvline(x=0, ymin=0, ymax=1, color = '#000000', linestyle = 'dotted')
    pad = 0.5
    # fig.tight_layout(pad=3.1, w_pad=0.5, h_pad=-1.3)
    fig.tight_layout(pad=2.0, w_pad=-2.25, h_pad=0.0)

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

    centre_plot = math.floor(len(amplitude_vals)/2)

    line_style =[
        'solid', 'dashed',
        'solid', 'solid',
        'solid', 'solid',
        'dashed', 'dashed'
    ]

    colour_sequence = [
                '#E69F00','#000000',
                '#56B4E9','#009E73',
                '#0072B2',
                '#D55E00','#000000',
                '#D55E00','#F0E442']
    
    plot_count = 0
    sequence_count = 0
    alpha_iterative_drop = 0.4
    harmonic_spacing = 0.1

    # kinetic_list = [1000,400,200,100,50]

    ax[1].set_title("$\Delta$E = {0} mV".format(50), pad=11)
    ax[5].set_title("$\Delta$E = {0} mV".format(300), pad=11)

    for i in range(0, len(amplitude_vals)):
        # ax.set_title('Title', pad=20)
        # ax[0, i].set_title("{0} mV".format(amplitude_vals[i]), pad=0)
        # h_class=harmonics_and_fourier_transform(harmonics = harm_range, experiment_frequency = experiment_frequency[i], selection_space= 0.1)

        if electrode == 'yellow' and amplitude_vals[i] == 250:
            experiment = 2
        else:
            experiment = 1
        file_name = str(amplitude_vals[i])+file_name_base+str(experiment) + current_mod
        current_data = np.loadtxt( os.path.join(input_directory,file_name))
        times = np.asarray(current_data[:,0])
        dim_exp_current = np.asarray(current_data[:,1])

        # adding simulated plots

        input_simulations_base_directory = os.path.dirname(os.path.realpath(__file__))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, electrode)
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'amplitude_'+str(amplitude_vals[i]))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'exp_' + str(experiment))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'varying_kinetics')

        dip_sim = np.load(os.path.join(input_simulations_base_directory,'k_is_'+str(int(1E5))+'_dip_is_'+str(False)+'.npy'))
        dip_sim = dip_sim*1e6
        plot_name = "$\mathrm{R_{u}}$ = "+str(int(0.0))+' $\mathrm{\Omega}$,' + " $\mathrm{k^{0}} = \mathrm{10^5}\ \mathrm{s^{-1}}$, & "+ " $\mathrm{E^{0}_{\sigma}}$ = "+str(int(0))+' $\mathrm{mV}$'

        sequence_count = 0

        # plotting time domain
        plt.subplot(1, total_sub_plots,plot_count+1)
        ax=plt.gca()
        if plot_count==0:
            # ax.set_ylabel('|Current| / $\mathrm{\mu}$A', labelpad=2)
            ax.set_ylabel('Current / $\mathrm{\mu}$A', labelpad=2)
        ax.plot(times, dip_sim, color = colour_sequence[sequence_count], linestyle=line_style[sequence_count], label=plot_name, alpha=1-(sequence_count*alpha_iterative_drop))
        ax.set_xlabel('Time / $\mathrm{s}$', labelpad=2)
        ax.tick_params(axis='y', which='major', pad=0.0, length = 2.5)
        ax.tick_params(axis='x', which='major', pad=0.0, length = 2.5)
      
        sequence_count += 1

        # adding experimental data
        plot_name = 'Ferrocene'

        dim_exp_current = dim_exp_current

        micro_amp_dim_exp_current = dim_exp_current*1e6

        # ax.set_ylabel('|Current| / $\mathrm{\mu}$A', labelpad=8)
        ax.plot(times, micro_amp_dim_exp_current, color = colour_sequence[sequence_count], linestyle=line_style[sequence_count], label=plot_name, alpha=1-(sequence_count*alpha_iterative_drop))


        # print(plot_count)
        if plot_count==0:
                # plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2, "borderaxespad":5}
                if total_sub_plots == 3:
                    ax.legend(loc = "upper center", bbox_to_anchor =[2.0, 1.6], ncol = 2,
                            frameon = False,columnspacing = 0.3, handlelength = 1.0)
                elif total_sub_plots == 7:
                    ax.legend(loc = "upper center", bbox_to_anchor =[3.7, 1.43], ncol = 2,
                            frameon = False,columnspacing = 0.9, handlelength = 1.0)
                else:
                    ax.legend(loc = "upper center", bbox_to_anchor =[1.6, 1.6], ncol = 2,
                            frameon = False,columnspacing = 0.3, handlelength = 1.0)

        plot_count+=1 + len(harm_range) +1
    
    # plt.savefig('deseried_variation_of_resistance.png', dpi=500)


    plot_count = 0
    sequence_count = 0
    # alpha_iterative_drop = 0.075
    # harmonic_spacing = harmonic_spacing

    # kinetic_list = [50,100,200,400,1000]

    for i in range(0, len(amplitude_vals)):
        # ax.set_title('Title', pad=20)
        # ax[0, i].set_title("{0} mV".format(amplitude_vals[i]), pad=0)
        h_class=harmonics_and_fourier_transform(harmonics = harm_range, experiment_frequency = experiment_frequency[i], selection_space= harmonic_spacing)

        if electrode == 'yellow' and amplitude_vals[i] == 250:
            experiment = 2
        else:
            experiment = 1
        file_name = str(amplitude_vals[i])+file_name_base+str(experiment) + current_mod
        current_data = np.loadtxt( os.path.join(input_directory,file_name))
        times = np.asarray(current_data[:,0])
        dim_exp_current = np.asarray(current_data[:,1])

        # adding simulated plots

        input_simulations_base_directory = os.path.dirname(os.path.realpath(__file__))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, electrode)
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'amplitude_'+str(amplitude_vals[i]))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'exp_' + str(experiment))
        input_simulations_base_directory = os.path.join(input_simulations_base_directory, 'varying_kinetics')

        dip_sim = np.load(os.path.join(input_simulations_base_directory,'k_is_'+str(int(1E5))+'_dip_is_'+str(False)+'.npy'))
        dip_sim = dip_sim*1e6
        plot_name = '$\mathrm{k^{0}}$ = ' + str(int(1E5)) + " $\mathrm{s^{-1}}$"

        sequence_count = 0

        # plotting harmonics
        # for harmonic_index in range(1, 1+len(harm_range)):
        #     plt.subplot(1, total_sub_plots,plot_count+1+harmonic_index)
        #     ax=plt.gca()
        #     # generating harmonic
        #     harmonic = dip_sim
        #     ax.plot(times, harmonic, color = colour_sequence[sequence_count], linestyle=line_style[sequence_count], label=plot_name, alpha=1-(sequence_count*alpha_iterative_drop))
        # plotting harmonics
        fft_current, frequencies = h_class._FT(dip_sim, h_class._time_step_size(times), half=False)
        for harmonic_index in range(1, 1+len(harm_range)):
            plt.subplot(1, total_sub_plots,plot_count+1+harmonic_index) 
            ax=plt.gca()
            ax.set_xlabel('Time / $\mathrm{s}$', labelpad=2)
            ax.tick_params(axis='y', which='major', pad=0.0, length = 2.5)
            ax.tick_params(axis='x', which='major', pad=0.0, length = 2.5)
            number_of_measurements = times.shape[0]
            harmonic=np.zeros((number_of_measurements), dtype="complex")
            # generating harmonic
            top_hat, top_hat_index = h_class._select_harmonics(transformed_data=fft_current, 
                                                        frequencies= frequencies, 
                                                        low_harmonic=harm_range[harmonic_index-1], 
                                                        upper_harmonic=harm_range[harmonic_index-1], 
                                                        experimental_frequency=experiment_frequency[i],
                                                        spacing_ratio=harmonic_spacing)
            if harm_range[harmonic_index-1] > 0.0:
                    harmonic[top_hat_index] = top_hat*2.0
            else:
                harmonic[top_hat_index] = top_hat
            harmonic=abs(np.fft.ifft(harmonic))
            ax.plot(times, harmonic, color = colour_sequence[sequence_count], linestyle=line_style[sequence_count], label=plot_name, alpha=1-(sequence_count*alpha_iterative_drop))
            
        
        sequence_count += 1

        # adding experimental data
        plot_name = 'Ferrocene'

        dim_exp_current = dim_exp_current

        micro_amp_dim_exp_current = dim_exp_current*1e6

        # plotting harmonics
        # plotting harmonics
        fft_current, frequencies = h_class._FT(micro_amp_dim_exp_current, h_class._time_step_size(times), half=False)
        for harmonic_index in range(1, 1+len(harm_range)):
            plt.subplot(1, total_sub_plots,plot_count+1+harmonic_index) 
            ax=plt.gca()
            number_of_measurements = times.shape[0]
            harmonic=np.zeros((number_of_measurements), dtype="complex")
            # generating harmonic
            top_hat, top_hat_index = h_class._select_harmonics(transformed_data=fft_current, 
                                                        frequencies= frequencies, 
                                                        low_harmonic=harm_range[harmonic_index-1], 
                                                        upper_harmonic=harm_range[harmonic_index-1], 
                                                        experimental_frequency=experiment_frequency[i],
                                                        spacing_ratio=harmonic_spacing)
            if harm_range[harmonic_index-1] > 0.0:
                    harmonic[top_hat_index] = top_hat*2.0
            else:
                harmonic[top_hat_index] = top_hat
            harmonic=abs(np.fft.ifft(harmonic))
            ax.plot(times, harmonic, color = colour_sequence[sequence_count], linestyle=line_style[sequence_count], label=plot_name, alpha=1-(sequence_count*alpha_iterative_drop))
        
        # for harmonic_index in range(1, 1+len(harm_range)):
        #     plt.subplot(1, total_sub_plots,plot_count+1+harmonic_index) 
        #     ax=plt.gca()
        #     # generating harmonic
        #     harmonic = micro_amp_dim_exp_current
        #     ax.plot(times, harmonic, color = colour_sequence[sequence_count], linestyle=line_style[sequence_count], label=plot_name, alpha=1-(sequence_count*alpha_iterative_drop))
        
        sequence_count += 1


        plot_count+=1 + len(harm_range) +1

    # fig.tight_layout(pad=0.0)
    # plt.show()
    plt.savefig('figure_5_a.png', dpi=500)