import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter

class harmonics_and_fourier_transform():
    def __init__(self, harmonics, experiment_frequency, selection_space):
        '''
            harmonics: The harmonics of interest.
            experiment_frequency: The frequency the experiment took place at.
            selection_space: Percentage of the frequency to use as window either side
                             of the harmonic when selecting the harmonic.
        '''

        # harmonics to plot
        self.harmonics=harmonics
        self.num_harmonics=len(harmonics)

        # frequency of the data
        self.experiment_frequency=experiment_frequency
        
        # percentage of the frequency
        # to use as window either side
        # of each harmonic
        self.selection_space=selection_space

    def _iteratively_generate_harmonics(self, times, current_data, **kwargs):
        '''
            times:
            current_data:
            **kwargs: currently only takes the window type to be used, i.e hanning, which can be True or False,
            or kaiser which should be a number, allowing adjustment of the windowing function being used.
        '''

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
            kaiser = False
        else:
            kaiser = True


        # setting windowing function on time domain data
        number_of_measurements=int(current_data.shape[0])
        half_of_measuremnts = math.ceil(number_of_measurements/2)
        if kwargs["hanning"]==True:
            window=np.hanning(number_of_measurements)
            time_domain=np.multiply(current_data, window)
        elif kaiser == True:
            if kwargs["kaiser"] == True:
                window = np.kaiser(number_of_measurements, 6)
                time_domain=np.multiply(current_data, window)
            else:
                window = np.kaiser(number_of_measurements, kwargs["kaiser"])
                time_domain=np.multiply(current_data, window)
        else:
            time_domain=current_data

        # carrying out fourier transform

        self.fft_current , self.frequencies = self._FT(time_domain, self._time_step_size(times), half=False)

        # array for selected harmonics
        harmonics=np.zeros((self.num_harmonics, number_of_measurements), dtype="complex")
        # iteratively selecting harmonics
        transformed_data = (copy.deepcopy(self.fft_current[:]))
        frequencies = (copy.deepcopy(self.frequencies[:]))
        for i in range(0,self.num_harmonics):
           
            top_hat, top_hat_index = self._select_harmonics(transformed_data=transformed_data, 
                                                            frequencies= frequencies, 
                                                            low_harmonic=self.harmonics[i], 
                                                            upper_harmonic=self.harmonics[i], 
                                                            experimental_frequency=self.experiment_frequency,
                                                            spacing_ratio=self.selection_space)

            # doubling intensities, as it is split between -ve and +ve
            # frequencies and we have discard negative frequencies
            if self.harmonics[i] > 0 and self.harmonics[i] > 0.0:
                harmonics[i,top_hat_index] = top_hat*2.0
            else:
                harmonics[i,top_hat_index] = top_hat

            
            # plt.figure(figsize=(18,10))
            # plt.title("optimised and experimental values")
            # plt.ylabel("current/dimless")
            # plt.plot(times ,abs(harmonics[0,:]),'b', label='fft_harm_4')
            # plt.legend(loc='best')
            # plt.show()
            # plt.close()
          
            # inverse fourier transform  
            harmonics[i,:]=((np.fft.ifft(harmonics[i,:])))

        return harmonics
    
    def _iteratively_generate_frequncy_domain_harmonics(self, times, current_data, **kwargs):
        '''
            times:
            current_data:
            **kwargs: currently only takes the window type to be used, i.e hanning, which can be True or False,
            or kaiser which should be a number, allowing adjustment of the windowing function being used.
        '''

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
            kaiser = False
        else:
            kaiser = True


        # setting windowing function on time domain data
        number_of_measurements=int(current_data.shape[0])
        half_of_measuremnts = math.ceil(number_of_measurements/2)
        if kwargs["hanning"]==True:
            window=np.hanning(number_of_measurements)
            time_domain=np.multiply(current_data, window)
        elif kaiser == True:
            if kwargs["kaiser"] == True:
                window = np.kaiser(number_of_measurements, 6)
                time_domain=np.multiply(current_data, window)
            else:
                window = np.kaiser(number_of_measurements, kwargs["kaiser"])
                time_domain=np.multiply(current_data, window)
        else:
            time_domain=current_data

        # carrying out fourier transform

        self.fft_current , self.frequencies = self._FT(time_domain, self._time_step_size(times), half=False)

        # array for selected harmonics
        harmonics=np.zeros((self.num_harmonics, number_of_measurements), dtype="complex")
        # iteratively selecting harmonics
        transformed_data = (copy.deepcopy(self.fft_current[:]))
        frequencies = (copy.deepcopy(self.frequencies[:]))
        for i in range(0,self.num_harmonics):
           
            top_hat, top_hat_index = self._select_harmonics(transformed_data=transformed_data, 
                                                            frequencies= frequencies, 
                                                            low_harmonic=self.harmonics[i], 
                                                            upper_harmonic=self.harmonics[i], 
                                                            experimental_frequency=self.experiment_frequency,
                                                            spacing_ratio=self.selection_space)

            # doubling intensities, as it is split between -ve and +ve
            # frequencies and we have discard negative frequencies
            if i == 0:
                harmonics=np.zeros((self.num_harmonics, len(top_hat)+2), dtype="complex")
            if self.harmonics[i] > 0 and self.harmonics[i] > 0.0:
                # harmonics[i,:] = top_hat*2.0
                harmonics[i, :len(top_hat)] = top_hat*2.0
                # harmonics[i,:] = np.trim_zeros(harmonics[i,:], trim='fb')
            else:
                # harmonics[i,:] = top_hat
                harmonics[i,  :len(top_hat)] = top_hat
                # harmonics[i,:] = np.trim_zeros(harmonics[i,:], trim='fb')

            # non_zero = harmonics[harmonics != 0]
            
            # plt.figure(figsize=(18,10))
            # plt.title("optimised and experimental values")
            # plt.ylabel("current/dimless")
            # plt.plot(times ,abs(harmonics[0,:]),'b', label='fft_harm_4')
            # plt.legend(loc='best')
            # plt.show()
            # plt.close()

        return harmonics
    
    

    def _FT(self, array_for_FFT, timeStepSize, half = True):
        """Fourier transforms given data

        param: array_for_FFT -  data to Fourier transform and reduce
        param: timeStepSize -  times step of the data
        param: half -  weather to return half the spectrum or not.

        return: fft of array_for_FFT, frequencies
        """

        numberOfMeasurements = int(array_for_FFT.shape[0])
        half_of_measuremnts = math.ceil(numberOfMeasurements/2)

        full_frequencies = np.fft.fftfreq(numberOfMeasurements, d=timeStepSize)
        half_frequencies=full_frequencies [:half_of_measuremnts]

        sp = np.fft.fft(array_for_FFT)
        full_frequencies = np.fft.fftfreq(numberOfMeasurements, d=timeStepSize)
        if half is True:
            sp = 2.0*sp # doubling amplitudes as it is split between -ve and +ve frequencies and we are going to discard negative frequencies
            sp_reduced = np.asarray(sp[:half_of_measuremnts],  dtype=complex)#discarding -ve frequencies
            half_frequencies=full_frequencies [:half_of_measuremnts]
            return sp_reduced, half_frequencies
        else:
            return sp, full_frequencies
        
    def _select_harmonics(self, transformed_data, frequencies, low_harmonic, upper_harmonic, experimental_frequency, spacing_ratio = 0.5):
        """Fourier transforms given data and reduces it to harmonics

        param: Data data to Fourier transform and reduce
        low_harmonic: The lower harmonic to be selected
        upper_harmonic: The upper harmonic to be selected (can be the same as low_harmonic)
        spacing_ratio: The amount of data either side of the harmonic to select
        as a multiple i.e 0.5 give 3.5 to 4.5
        return: numpy array contain selected harmonic, and indexes of non_zero_data
        """
        
        lower_true_harm_freq=(low_harmonic-spacing_ratio)*experimental_frequency
        upper_true_harm_freq=(upper_harmonic+spacing_ratio)*experimental_frequency

        top_hat=np.where((frequencies<upper_true_harm_freq) & (frequencies>lower_true_harm_freq))

        return copy.deepcopy(transformed_data[top_hat]), top_hat
        
    def _time_step_size(self, times):
        '''
        calculates the average time step of the data
        '''
        numberOfMeasurements = int(times.shape[0])
        return times[-1]/(numberOfMeasurements - 1)
    
        
    def harmonic_plotting(self, times, **kwargs):
        '''
        helper function for plotting multiple harmonics
        params: times series for all desired plots
        params: hanning set to true for hanning window
        params: kaiser set to a number if a kaiser window is desired
        params: xaxis if a x axis other than times (i.e voltage) is desired pass it here
        params: abs_real_imag set to one of np.abs, np.real, or np.imag depending on what harmonic portion you want to plot
        params: xlabel x axis label name
        params: ylabel y axis label name
        params: legend location of the legend

        '''
        label_list=[]
        time_series_dict={}
        harm_dict={}

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
        if "xaxis" not in kwargs:
            kwargs["xaxis"]=times
        if "alpha_iterative_drop" not in kwargs:
            kwargs["alpha_iterative_drop"]=0
        if "abs_real_imag" not in kwargs:
            kwargs["abs_real_imag"]=np.abs
        if "xlabel" not in kwargs:
            kwargs["xlabel"]=""
        if "ylabel" not in kwargs:
            kwargs["ylabel"]=""
        if "micro_amps" not in kwargs:
            kwargs["micro_amps"]=False
        if "label" not in kwargs:
            kwargs["label"]=False
        if "colour_sequence" not in kwargs:
            kwargs["colour_sequence"]=False
            # list of 8 colour blind safe
            # colours
            colour_sequence = [
                '#000000','#E69F00',
                '#56B4E9','#009E73',
                '#CC79A7','#0072B2',
                '#D55E00','#F0E442']
        else:
            colour_sequence = kwargs["colour_sequence"]
        if "line_style" not in kwargs:
            kwargs["line_style"]=False
            # list of 8 colour blind safe
            # colours
            line_style = [
                'solid','solid',
                'solid','solid',
                'solid','solid',
                'solid','solid']
        else:
            line_style = kwargs["line_style"]

        if "legend" not in kwargs:
            kwargs["legend"]={"loc":"center"}
        if "axes_list" not in kwargs:
            define_axes=True
        else:
            if len(kwargs["axes_list"])!=self.num_harmonics:
                raise ValueError("Wrong number of axes for harmonics")
            else:
                define_axes=False
        label_counter=0

        for key in kwargs:
            if "time_series" in key:
                index=key.find("time_series")

                if key[index-1]=="_" or key[index-1]=="-":
                    index-=1

                if kwargs['label'] == False:
                    label_list.append(key[:index])
                    time_series_dict[key[:index]]=kwargs[key]
                else:
                    looking_for = '_'+str(key[0:index])
                    for key2 in kwargs:
                        if "label" in key2:
                            if looking_for in kwargs[key2]:
                                label = kwargs[key2]
                                index2=label.find(looking_for)
                                label = label[:index2]

                                label_list.append(label)
                                time_series_dict[label]=kwargs[key]


                label_counter+=1

        if label_counter==0:
            return
        for label in label_list:
            harm_dict[label]=self._iteratively_generate_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"], kaiser=kwargs["kaiser"])
        num_harms=self.num_harmonics

        for i in range(0, num_harms):
            if define_axes==True:
                plt.subplot(num_harms, 1,i+1)
                ax=plt.gca()
            else:
                ax=kwargs["axes_list"][i]

            # ax.tick_params(left = False) 
            # ax.yaxis.set_tick_params(width=1)
            ax.tick_params(axis='y', which='major', pad=0.0, length = 1)
            
            # ax2=ax.twinx()
            # ax2.set_yticks([])
            # # ax2.set_ylabel(self.harmonics[i], labelpad=8.0, rotation=0,  weight='extra bold', fontsize = 'x-small')
            # ax2.set_ylabel(self.harmonics[i], labelpad=4.0, rotation=0,  weight='extra bold',)
            plot_counter=0
            for plot_name in label_list:
                
                if kwargs["micro_amps"] == False:
                    y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                else:
                     y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                if i==0:
                    print(plot_name)
                ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], linestyle=line_style[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                plot_counter+=1
            if i==((num_harms)//2):
                ax.set_ylabel(kwargs["ylabel"], labelpad=8)

            # if plot_name == label_list[-1]:
            #     ax2 = ax.twinx()
            #     ax2.set_yticks([])
            #     ax2.set_ylabel(self.harmonics[i], labelpad=4.0, rotation=0,  weight='extra bold',)
            # ax.ax1('|i$_\mathrm{'+str(self.harmonics[i])+'\omega t}$|', labelpad=8)
            if i==num_harms-1:
                ax.set_xlabel(kwargs["xlabel"], labelpad=1)
            else:
                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            if i==0:
                if kwargs["legend"] is not None:
                    ax.legend(**kwargs["legend"])


    def freq_domain_harmonic_plotting(self, times, **kwargs):
        '''
        helper function for plotting multiple harmonics
        params: times series for all desired plots
        params: hanning set to true for hanning window
        params: kaiser set to a number if a kaiser window is desired
        params: xaxis if a x axis other than times (i.e voltage) is desired pass it here
        params: abs_real_imag set to one of np.abs, np.real, or np.imag depending on what harmonic portion you want to plot
        params: xlabel x axis label name
        params: ylabel y axis label name
        params: legend location of the legend

        '''
        label_list=[]
        time_series_dict={}
        harm_dict={}

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
        if "xaxis" not in kwargs:
            kwargs["xaxis"]=times
        if "alpha_iterative_drop" not in kwargs:
            kwargs["alpha_iterative_drop"]=0
        if "abs_real_imag" not in kwargs:
            kwargs["abs_real_imag"]=np.abs
        if "xlabel" not in kwargs:
            kwargs["xlabel"]=""
        if "ylabel" not in kwargs:
            kwargs["ylabel"]=""
        if "micro_amps" not in kwargs:
            kwargs["micro_amps"]=False
        if "label" not in kwargs:
            kwargs["label"]=False
        if "colour_sequence" not in kwargs:
            kwargs["colour_sequence"]=False
            # list of 8 colour blind safe
            # colours
            colour_sequence = [
                '#000000','#E69F00',
                '#56B4E9','#009E73',
                '#CC79A7','#0072B2',
                '#D55E00','#F0E442']
        else:
            colour_sequence = kwargs["colour_sequence"]

        if "legend" not in kwargs:
            kwargs["legend"]={"loc":"center"}
        if "axes_list" not in kwargs:
            define_axes=True
        else:
            if len(kwargs["axes_list"])!=self.num_harmonics:
                raise ValueError("Wrong number of axes for harmonics")
            else:
                define_axes=False
        label_counter=0

        for key in kwargs:
            if "time_series" in key:
                index=key.find("time_series")

                if key[index-1]=="_" or key[index-1]=="-":
                    index-=1

                if kwargs['label'] == False:
                    label_list.append(key[:index])
                    time_series_dict[key[:index]]=kwargs[key]
                else:
                    looking_for = '_'+str(key[0:index])
                    for key2 in kwargs:
                        if "label" in key2:
                            if looking_for in kwargs[key2]:
                                label = kwargs[key2]
                                index2=label.find(looking_for)
                                label = label[:index2]

                                label_list.append(label)
                                time_series_dict[label]=kwargs[key]


                label_counter+=1

        if label_counter==0:
            return
        for label in label_list:
            harm_dict[label]=self._iteratively_generate_frequncy_domain_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"], kaiser=kwargs["kaiser"])
        num_harms=self.num_harmonics

        for i in range(0, num_harms):
            if define_axes==True:
                plt.subplot(num_harms, 1,i+1)
                ax=plt.gca()
            else:
                ax=kwargs["axes_list"][i]
            
            # ax2=ax.twinx()
            # ax2.set_yticks([])
            # # ax2.set_ylabel(self.harmonics[i], labelpad=8.0, rotation=0,  weight='extra bold', fontsize = 'x-small')
            # ax2.set_ylabel(self.harmonics[i], labelpad=4.0, rotation=0,  weight='extra bold',)
            plot_counter=0
            for plot_name in label_list:
                
                if kwargs["micro_amps"] == False:
                    y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                else:
                     y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                if i==0:
                    print(plot_name)
                # ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.plot( y, color = colour_sequence[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                plot_counter+=1
            if i==((num_harms)//2):
                ax.set_ylabel(kwargs["ylabel"])
            if i==num_harms-1:
                ax.set_xlabel(kwargs["xlabel"])
            else:
                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            if i==0:
                if kwargs["legend"] is not None:
                    ax.legend(**kwargs["legend"])
        
    def normalised_harmonic_plot(self, times, **kwargs):
        '''
        helper function for plotting multiple harmonics
        params: times series for all desired plots
        params: hanning set to true for hanning window
        params: kaiser set to a number if a kaiser window is desired
        params: xaxis if a x axis other than times (i.e voltage) is desired pass it here
        params: abs_real_imag set to one of np.abs, np.real, or np.imag depending on what harmonic portion you want to plot
        params: xlabel x axis label name
        params: ylabel y axis label name
        params: legend location of the legend

        '''
        label_list=[]
        time_series_dict={}
        harm_dict={}

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
        if "xaxis" not in kwargs:
            kwargs["xaxis"]=times
        if "alpha_iterative_drop" not in kwargs:
            kwargs["alpha_iterative_drop"]=0
        if "abs_real_imag" not in kwargs:
            kwargs["abs_real_imag"]=np.abs
        if "xlabel" not in kwargs:
            kwargs["xlabel"]=""
        if "ylabel" not in kwargs:
            kwargs["ylabel"]=""
        if "micro_amps" not in kwargs:
            kwargs["micro_amps"]=False
        if "label" not in kwargs:
            kwargs["label"]=False
        if "colour_sequence" not in kwargs:
            kwargs["colour_sequence"]=False
            # list of 8 colour blind safe
            # colours
            colour_sequence = [
                '#000000','#E69F00',
                '#56B4E9','#009E73',
                '#CC79A7','#0072B2',
                '#D55E00','#F0E442']
        else:
            colour_sequence = kwargs["colour_sequence"]

        if "legend" not in kwargs:
            kwargs["legend"]={"loc":"center"}
        if "axes_list" not in kwargs:
            define_axes=True
        else:
            if len(kwargs["axes_list"])!=self.num_harmonics:
                raise ValueError("Wrong number of axes for harmonics")
            else:
                define_axes=False
        label_counter=0

        for key in kwargs:
            if "time_series" in key:
                index=key.find("time_series")

                if key[index-1]=="_" or key[index-1]=="-":
                    index-=1

                if kwargs['label'] == False:
                    label_list.append(key[:index])
                    time_series_dict[key[:index]]=kwargs[key]
                else:
                    looking_for = '_'+str(key[0:index])
                    for key2 in kwargs:
                        if "label" in key2:
                            if looking_for in kwargs[key2]:
                                label = kwargs[key2]
                                index2=label.find(looking_for)
                                label = label[:index2]

                                label_list.append(label)
                                time_series_dict[label]=kwargs[key]


                label_counter+=1

        if label_counter==0:
            return
        for label in label_list:
            harm_dict[label]=self._iteratively_generate_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"], kaiser=kwargs["kaiser"])
        num_harms=self.num_harmonics

        # we plot the residuals
        for i in range(0, num_harms):
            if define_axes==True:
                plt.subplot(num_harms, 1,i+1)
                ax=plt.gca()
            else:
                ax=kwargs["axes_list"][i]

            # ax2=ax.twinx()
            # ax2.set_yticks([])
            # ax2.set_ylabel(self.harmonics[i], rotation=0,  weight='extra bold', fontsize = 'x-small')
            plot_counter=0
            for label in range(0, len(label_list)):
                plot_name = label_list[label]

                # if plot_name == label_list[-1]:
                #     ax2=ax.twinx()
                #     ax2.set_yticks([])
                #     ax2.set_ylabel(self.harmonics[i], rotation=0,  weight='extra bold')

                if label == 0:
                    if kwargs["micro_amps"] == False:
                        ref_signal =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                    else:
                        ref_signal =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                    normalisation_factor = np.amax(ref_signal)
                    ref_signal = self.normalise(ref_signal, normalisation_factor)
                # else:
                if kwargs["micro_amps"] == False:
                    y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                    y = self.normalise(y, normalisation_factor)
                else:
                    y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                    y = self.normalise(y, normalisation_factor)

                # calculating resiudal
                # y = ref_signal - y

                ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                plot_counter+=1
                # ax.set_ylim(0.0,1.05)
                ax.set_yticks([0,1])
                ax.tick_params(left = False) 
                ax.tick_params(axis='y', which='major', pad=-1.5)
                if i==((num_harms)//2):
                    ax.set_ylabel(kwargs["ylabel"])
                if i==num_harms-1:
                    ax.set_xlabel(kwargs["xlabel"])
                else:
                    ax.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False) # labels along the bottom edge are off
                
                if i==0:
                    if kwargs["legend"] is not None:
                        ax.legend(**kwargs["legend"])

    def normalise(self, data, normalisation_factor = None):
        
        if normalisation_factor == None:
            normalisation_factor = np.amax(data)

        return np.divide(data,normalisation_factor)
    

    def harmonic_residuals_normalised_root_mean_square(self, times, **kwargs):
        '''
        helper function for plotting multiple harmonics
        params: times series for all desired plots
        params: hanning set to true for hanning window
        params: kaiser set to a number if a kaiser window is desired
        params: xaxis if a x axis other than times (i.e voltage) is desired pass it here
        params: abs_real_imag set to one of np.abs, np.real, or np.imag depending on what harmonic portion you want to plot
        params: xlabel x axis label name
        params: ylabel y axis label name
        params: legend location of the legend

        '''
        label_list=[]
        time_series_dict={}
        harm_dict={}

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
        if "micro_amps" not in kwargs:
            kwargs["micro_amps"]=False
        if "abs_real_imag" not in kwargs:
            kwargs["abs_real_imag"]=np.abs
        if 'zero_point' not in kwargs:
            kwargs['zero_point'] = 5.0E-2
    
       
        label_counter=0

        for key in kwargs:
            if "time_series" in key:
                index=key.find("time_series")

                if key[index-1]=="_" or key[index-1]=="-":
                    index-=1

                if kwargs['label'] == False:
                    label_list.append(key[:index])
                    time_series_dict[key[:index]]=kwargs[key]
                else:
                    looking_for = '_'+str(key[0:index])
                    for key2 in kwargs:
                        if "label" in key2:
                            if looking_for in kwargs[key2]:
                                label = kwargs[key2]
                                index2=label.find(looking_for)
                                label = label[:index2]

                                label_list.append(label)
                                time_series_dict[label]=kwargs[key]


                label_counter+=1

        if label_counter==0:
            return
        for label in label_list:
            harm_dict[label]=self._iteratively_generate_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"], kaiser=kwargs["kaiser"])
        num_harms=self.num_harmonics

        # we cacluate the root mean square of the individual harmonics

        matrix_of_harmonic_residuals_normalised_root_mean_square = np.zeros((len(label_list)-1, num_harms))
        helper_data_and_har_matrix = np.chararray((len(label_list)-1, num_harms), itemsize=64)
        for i in range(0, num_harms):

            for label in range(0, len(label_list)):

                plot_name = label_list[label]

                if label == 0:
                    ref_name = plot_name
                    if kwargs["micro_amps"] == False:
                        ref_signal =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                    else:
                        ref_signal =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                    ref_signal = abs(ref_signal)
                    normalisation_factor = np.amax(ref_signal)
                    ref_signal = self.normalise(ref_signal, normalisation_factor)
                else:    
                    if kwargs["micro_amps"] == False:
                        y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                    else:
                        y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                    y = abs(y)
                    y = self.normalise(y, normalisation_factor)

                    print('y max: ', np.amax(y))
                    print('ref_signal max: ', np.amax(ref_signal))
                    print('y min: ', np.amin(y))
                    print('ref_signal min: ', np.amin(ref_signal))


                    zero_point = kwargs['zero_point']
                    diff = []

                    # normalising by number of 
                    # non zero points to take
                    # into account harmonic broadening
                    normalisation = 0

                    for index_value in range(0, len(ref_signal)):

                        if ref_signal[index_value] > zero_point or y[index_value]>zero_point:
                            diff.append(ref_signal[index_value] - y[index_value])
                            normalisation += 1

                    # plt.plot(ref_signal, label = 'ref')
                    # plt.plot(y, label = 'y')
                    # plt.plot(diff, label = 'diff')
                    # plt.show()

                    diff =  np.asarray(diff)

                    absolute = abs(diff)

                    squares = np.square(absolute)

                    total_complex_diff = np.sum(squares)

                    print('number of non zero points: ', normalisation)
                    # print('diff min: ', np.amin(diff))
                    # print('absolute min: ', np.amin(absolute))
                    # print('squares min: ', np.amin(squares))
                    # print('diff min index: ', np.argmin(diff))
                    # print('absolute min index: ', np.argmin(absolute))
                    # print('squares min index: ', np.argmin(squares))
                    # print('diff max: ', np.amax(diff))
                    # # print('absolute max: ', np.amax(absolute))
                    # print('squares max: ', np.amax(squares))
                    # print('normalisation: ', normalisation)
                    total_complex_diff = total_complex_diff/normalisation
                    obj_func = np.sqrt(total_complex_diff)

                    matrix_of_harmonic_residuals_normalised_root_mean_square[label-1,i] = obj_func
                    helper_data_and_har_matrix[label-1,i]  = 'harmonic ' +str(self.harmonics[i]) +','+ str(plot_name)
                    # helper_data_and_har_matrix[label-1,i]  = '22'

        return matrix_of_harmonic_residuals_normalised_root_mean_square , helper_data_and_har_matrix
    
    def normalised_harmonic_residuals_plot(self, times, **kwargs):
        '''
        helper function for plotting multiple harmonics
        params: times series for all desired plots
        params: hanning set to true for hanning window
        params: kaiser set to a number if a kaiser window is desired
        params: xaxis if a x axis other than times (i.e voltage) is desired pass it here
        params: abs_real_imag set to one of np.abs, np.real, or np.imag depending on what harmonic portion you want to plot
        params: xlabel x axis label name
        params: ylabel y axis label name
        params: legend location of the legend

        '''
        label_list=[]
        time_series_dict={}
        harm_dict={}

        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "kaiser" not in kwargs:
            kwargs["kaiser"]=False
        if "xaxis" not in kwargs:
            kwargs["xaxis"]=times
        if "alpha_iterative_drop" not in kwargs:
            kwargs["alpha_iterative_drop"]=0
        if "abs_real_imag" not in kwargs:
            kwargs["abs_real_imag"]=np.abs
        if "xlabel" not in kwargs:
            kwargs["xlabel"]=""
        if "ylabel" not in kwargs:
            kwargs["ylabel"]=""
        if "micro_amps" not in kwargs:
            kwargs["micro_amps"]=False
        if "label" not in kwargs:
            kwargs["label"]=False
        if "colour_sequence" not in kwargs:
            kwargs["colour_sequence"]=False
            # list of 8 colour blind safe
            # colours
            colour_sequence = [
                '#000000','#E69F00',
                '#56B4E9','#009E73',
                '#CC79A7','#0072B2',
                '#D55E00','#F0E442']
        else:
            colour_sequence = kwargs["colour_sequence"]

        if "legend" not in kwargs:
            kwargs["legend"]={"loc":"center"}
        if "axes_list" not in kwargs:
            define_axes=True
        else:
            if len(kwargs["axes_list"])!=self.num_harmonics:
                raise ValueError("Wrong number of axes for harmonics")
            else:
                define_axes=False
        label_counter=0

        for key in kwargs:
            if "time_series" in key:
                index=key.find("time_series")

                if key[index-1]=="_" or key[index-1]=="-":
                    index-=1

                if kwargs['label'] == False:
                    label_list.append(key[:index])
                    time_series_dict[key[:index]]=kwargs[key]
                else:
                    looking_for = '_'+str(key[0:index])
                    for key2 in kwargs:
                        if "label" in key2:
                            if looking_for in kwargs[key2]:
                                label = kwargs[key2]
                                index2=label.find(looking_for)
                                label = label[:index2]

                                label_list.append(label)
                                time_series_dict[label]=kwargs[key]


                label_counter+=1

        if label_counter==0:
            return
        for label in label_list:
            harm_dict[label]=self._iteratively_generate_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"], kaiser=kwargs["kaiser"])
        num_harms=self.num_harmonics

        # we plot the residuals
        for i in range(0, num_harms):
            if define_axes==True:
                plt.subplot(num_harms, 1,i+1)
                ax=plt.gca()
            else:
                ax=kwargs["axes_list"][i]

            ax2=ax.twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(self.harmonics[i], rotation=0,  weight='extra bold', fontsize = 'x-small')
            plot_counter=0
            for label in range(0, len(label_list)):
                plot_name = label_list[label]

                if label == 0:
                    if kwargs["micro_amps"] == False:
                        ref_signal =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                        normalisation_factor = np.amax(ref_signal)
                        ref_signal = self.normalise(ref_signal, normalisation_factor)
                    else:
                        ref_signal =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                        normalisation_factor = np.amax(ref_signal)
                        ref_signal = self.normalise(ref_signal, normalisation_factor)
                else:
                    if kwargs["micro_amps"] == False:
                        y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                        y = self.normalise(y, normalisation_factor)
                    else:
                        y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6
                        y = self.normalise(y, normalisation_factor)

                    # calculating resiudal
                    y = ref_signal - y

                    ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                    plot_counter+=1
                    if i==((num_harms)//2):
                        ax.set_ylabel(kwargs["ylabel"])
                    if i==num_harms-1:
                        ax.set_xlabel(kwargs["xlabel"])
                    else:
                        ax.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
                    if i==0:
                        if kwargs["legend"] is not None:
                            ax.legend(**kwargs["legend"])