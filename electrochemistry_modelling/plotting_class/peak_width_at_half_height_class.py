import numpy as np
import copy
import matplotlib.pyplot as plt
import math

class peak_width_at_half_height():
    def __init__(self, harmonics, experiment_frequency, selection_space, scan_rate =40.047e-3):
        '''
            harmonics: The harmonics of interest.
            experiment_frequency: The frequency the experiment took place at.
            selection_space: Percentage of the frequency to use as window either side
                             of the harmonic when selecting the harmonic.
        '''

        # harmonics to plot
        self.harmonics=harmonics
        self.num_harmonics=len(harmonics)
        self.scan_rate = scan_rate

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
    
        
    def half_height_harmonic_plotting(self, times, **kwargs):
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
                # raise ValueError("Wrong number of axes for harmonics")
                define_axes=False
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

                number_of_measurements=int(y.shape[0])

                half_measurements = int(number_of_measurements/2)

                # calculating first half width
                print('harmonic: ', self.harmonics[i])
                first_half_maximum = np.max(y[int(half_measurements*0.1):half_measurements])
                global_minium = np.min(y)
                # print('first_half_maximum: ', first_half_maximum)
                index_first_half_maximum = np.argmax(y[int(half_measurements*0.1):half_measurements])
                index_first_half_maximum = int(half_measurements*0.1) + index_first_half_maximum
                # print('index_first_half_maximumlf_maximum: ', index_first_half_maximum)
                frist_desired_half_point = (first_half_maximum - global_minium)/2 + global_minium
                print('desired_half_point: ', frist_desired_half_point)

                differences = np.abs(y[int(half_measurements*0.1):index_first_half_maximum] - frist_desired_half_point)

                first_half_left_closest_index = np.argmin(differences) +  int(half_measurements*0.1)

                differences = np.abs(y[index_first_half_maximum: half_measurements] - frist_desired_half_point)

                first_half_right_closest_index = np.argmin(differences) +  int(index_first_half_maximum)


                # setting offset
                setting_offset = int(number_of_measurements*0.05)
                # taking region of interest
                differences = y[setting_offset:half_measurements] - frist_desired_half_point
                # differences = y[setting_offset:index_first_half_maximum] - frist_desired_half_point

                # tolerances
                minium = np.min(np.abs(differences))
                # print('intrested in: ', tolerance)
                tolerance = minium*1e2
                # finding region of minium value
                top_hat=np.where((differences<(minium + tolerance)) & (differences>(minium - tolerance)))
                print('intrested in: ', top_hat)

                # TODO: block of indexes sepearted by more than 1
                # TODO: take first and last block
                # TODO: find minium in both blocks as before
                # TODO: join up and that is your width for any harmonic and should translate to time domain

                offset_top_hat = np.asarray(top_hat) + setting_offset
                # print('offset_top_hat pre flattening: ', offset_top_hat)
                offset_top_hat = offset_top_hat.flatten()
                # print('offset_top_hat post flattening: ', offset_top_hat)
                # offset_top_list = offset_top_hat.tolist()
                # print('offset_top_hat post list: ', offset_top_list)
                # offset_top_hat = tuple([offset_top_list])
                # print('offset_top_hat post tuple: ', offset_top_hat)

                # print('offset_top_hat.shape: ',offset_top_hat.shape)

                last_index_of_first_block = 1
                while offset_top_hat[last_index_of_first_block] - offset_top_hat[last_index_of_first_block-1] == 1:
                    last_index_of_first_block = last_index_of_first_block+1

                first_block = offset_top_hat[:last_index_of_first_block]

                first_block_offset = first_block[0]
                differences = np.abs(y[first_block] - frist_desired_half_point)

                new_first_half_left_closest_index = np.argmin(differences) + first_block_offset

                first_block = first_block.tolist()
                first_block = tuple([first_block])


                first_index_of_last_block = offset_top_hat.shape[0] -1
                while offset_top_hat[first_index_of_last_block] - offset_top_hat[first_index_of_last_block-1] == 1:
                    first_index_of_last_block = first_index_of_last_block-1

                second_block = offset_top_hat[first_index_of_last_block:]

                second_block_offset = second_block[0]
                differences = np.abs(y[second_block] - frist_desired_half_point)

                new_first_half_right_closest_index = np.argmin(differences) + second_block_offset
                # print('new_first_half_right_closest_index: ', new_first_half_right_closest_index)
                # print('first_half_left_closest_index: ', first_half_right_closest_index)


                second_block = second_block.tolist()
                second_block = tuple([second_block])

                print('****')
                print('NEW first sweep width at half height in mV: ', (kwargs["xaxis"][new_first_half_right_closest_index] -  kwargs["xaxis"][new_first_half_left_closest_index])*self.scan_rate*1e3)
                print('****')



                # total_up_and_down_index = int(self.harmonics[i]*2) -1

                # offset_top_hat = tuple(first_block)

                # first_half_left_closest_index = np.argmin(differences) +  int(half_measurements*0.1)

                # differences = np.abs(y[index_first_half_maximum: half_measurements] - frist_desired_half_point)

                # first_half_right_closest_index = np.argmin(differences) +  int(index_first_half_maximum)



                # # calculating second half width
                # print('harmonic: ', self.harmonics[i])
                second_half_maximum = np.max(y[half_measurements:int(number_of_measurements*0.95)])
                # print('second_half_maximum: ', second_half_maximum)
                index_second_half_maximum = np.argmax(y[half_measurements:int(number_of_measurements*0.95)])
                index_second_half_maximum = int(half_measurements) + index_second_half_maximum
                # print('index_second_half_maximumlf_maximum: ', index_second_half_maximum)
                second_desired_half_point = (second_half_maximum - global_minium)/2 + global_minium
                # print('desired_half_point: ', second_desired_half_point)

                differences = np.abs(y[half_measurements:index_second_half_maximum] - second_desired_half_point)

                second_half_left_closest_index = np.argmin(differences) +  int(half_measurements)

                differences = np.abs(y[index_second_half_maximum:int(number_of_measurements*0.95)] - second_desired_half_point)

                second_half_right_closest_index = np.argmin(differences) +  int(index_second_half_maximum)

                # print('first time difference: ', kwargs["xaxis"][first_half_right_closest_index] -  kwargs["xaxis"][first_half_left_closest_index])
                print('first sweep width at half height in mV: ', (kwargs["xaxis"][first_half_right_closest_index] -  kwargs["xaxis"][first_half_left_closest_index])*self.scan_rate*1e3)

                # print('second time difference: ', kwargs["xaxis"][second_half_right_closest_index] -  kwargs["xaxis"][second_half_left_closest_index])
                print('second sweep width at half height in mV: ', (kwargs["xaxis"][second_half_right_closest_index] -  kwargs["xaxis"][second_half_left_closest_index])*self.scan_rate*1e3)

                dc_location_of_first_max = -350.0e-3 + self.scan_rate*kwargs["xaxis"][index_first_half_maximum]
                dc_location_of_second_max = 725.0e-3 - (self.scan_rate*(kwargs["xaxis"][index_second_half_maximum] - (1075.0e-3/self.scan_rate)))
                print('offset in mV: ', (dc_location_of_first_max -  dc_location_of_second_max)*1e3)
                print('dc_location_of_first_max in mV: ', dc_location_of_first_max*1e3)
                print('dc_location_of_second_max in mV: ', dc_location_of_second_max*1e3)


                ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], linestyle=line_style[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.plot(kwargs["xaxis"][first_block], y[first_block], color = '#000000', linestyle=line_style[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.plot(kwargs["xaxis"][second_block], y[second_block], color = '#000000', linestyle=line_style[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][index_first_half_maximum], y[index_first_half_maximum], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][first_half_left_closest_index], y[first_half_left_closest_index], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][first_half_right_closest_index], y[first_half_right_closest_index], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][new_first_half_left_closest_index], y[new_first_half_left_closest_index], color = '#000000', alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][new_first_half_right_closest_index], y[new_first_half_right_closest_index], color = '#000000', alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.axhline(y = frist_desired_half_point, color = 'r', linestyle = '-')
                # ax.scatter(kwargs["xaxis"][index_second_half_maximum], y[index_second_half_maximum], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                # ax.scatter(kwargs["xaxis"][second_half_left_closest_index], y[second_half_left_closest_index], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                # ax.scatter(kwargs["xaxis"][second_half_right_closest_index], y[second_half_right_closest_index], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                # ax.axhline(y = second_desired_half_point, color = 'r', linestyle = '-') 
                plot_counter+=1
            if i==((num_harms)//2):
                ax.set_ylabel(kwargs["ylabel"], labelpad=8)
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

    def half_height_calcutor(self, times, **kwargs):
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

            plot_counter=0
            for plot_name in label_list:
                
                if kwargs["micro_amps"] == False:
                    y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])
                else:
                     y =  kwargs["abs_real_imag"](harm_dict[plot_name][i,:])*1E6

                number_of_measurements=int(y.shape[0])

                half_measurements = int(number_of_measurements/2)

                 # calculating first half width
                print('harmonic: ', self.harmonics[i])
                first_half_maximum = np.max(y[int(half_measurements*0.1):half_measurements])
                global_minium = np.min(y)
                # print('first_half_maximum: ', first_half_maximum)
                index_first_half_maximum = np.argmax(y[int(half_measurements*0.1):half_measurements])
                index_first_half_maximum = int(half_measurements*0.1) + index_first_half_maximum
                # print('index_first_half_maximumlf_maximum: ', index_first_half_maximum)
                frist_desired_half_point = (first_half_maximum - global_minium)/2 + global_minium
                # print('desired_half_point: ', frist_desired_half_point)

                differences = np.abs(y[int(half_measurements*0.1):index_first_half_maximum] - frist_desired_half_point)

                first_half_left_closest_index = np.argmin(differences) +  int(half_measurements*0.1)

                differences = np.abs(y[index_first_half_maximum: half_measurements] - frist_desired_half_point)

                first_half_right_closest_index = np.argmin(differences) +  int(index_first_half_maximum)

                # # calculating second half width
                # print('harmonic: ', self.harmonics[i])
                second_half_maximum = np.max(y[half_measurements:int(number_of_measurements*0.95)])
                # print('second_half_maximum: ', second_half_maximum)
                index_second_half_maximum = np.argmax(y[half_measurements:int(number_of_measurements*0.95)])
                index_second_half_maximum = int(half_measurements) + index_second_half_maximum
                # print('index_second_half_maximumlf_maximum: ', index_second_half_maximum)
                second_desired_half_point = (second_half_maximum - global_minium)/2 + global_minium
                # print('desired_half_point: ', second_desired_half_point)

                differences = np.abs(y[half_measurements:index_second_half_maximum] - second_desired_half_point)

                second_half_left_closest_index = np.argmin(differences) +  int(half_measurements)

                differences = np.abs(y[index_second_half_maximum:int(number_of_measurements*0.95)] - second_desired_half_point)

                second_half_right_closest_index = np.argmin(differences) +  int(index_second_half_maximum)

                # print('first time difference: ', kwargs["xaxis"][first_half_right_closest_index] -  kwargs["xaxis"][first_half_left_closest_index])
                print('first sweep width at half height in mV: ', (kwargs["xaxis"][first_half_right_closest_index] -  kwargs["xaxis"][first_half_left_closest_index])*self.scan_rate*1e3)

                # print('second time difference: ', kwargs["xaxis"][second_half_right_closest_index] -  kwargs["xaxis"][second_half_left_closest_index])
                print('second sweep width at half height in mV: ', (kwargs["xaxis"][second_half_right_closest_index] -  kwargs["xaxis"][second_half_left_closest_index])*self.scan_rate*1e3)

                dc_location_of_first_max = -350.0e-3 + self.scan_rate*kwargs["xaxis"][index_first_half_maximum]
                dc_location_of_second_max = 725.0e-3 - (self.scan_rate*(kwargs["xaxis"][index_second_half_maximum] - (1075.0e-3/self.scan_rate)))
                print('offset in mV: ', (dc_location_of_first_max -  dc_location_of_second_max)*1e3)
                print('dc_location_of_first_max in mV: ', dc_location_of_first_max*1e3)
                print('dc_location_of_second_max in mV: ', dc_location_of_second_max*1e3)


                # calculating second half width
                # calculating max hight in first half
                # width

            #     if i==0:
            #         print(plot_name)
            #     ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], linestyle=line_style[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
            #     plot_counter+=1
            # if i==((num_harms)//2):
            #     ax.set_ylabel(kwargs["ylabel"], labelpad=8)
            # if i==num_harms-1:
            #     ax.set_xlabel(kwargs["xlabel"], labelpad=1)
            # else:
            #     ax.tick_params(
            #         axis='x',          # changes apply to the x-axis
            #         which='both',      # both major and minor ticks are affected
            #         bottom=False,      # ticks along the bottom edge are off
            #         top=False,         # ticks along the top edge are off
            #         labelbottom=False) # labels along the bottom edge are off
            # if i==0:
            #     if kwargs["legend"] is not None:
            #         ax.legend(**kwargs["legend"])

    
    def Peak_peak_seperation_2nd_harmonic_plotting(self, times, **kwargs):
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
                # raise ValueError("Wrong number of axes for harmonics")
                define_axes=False
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

                number_of_measurements=int(y.shape[0])

                half_measurements = int(number_of_measurements/2)

                # calculating first half width
                print('harmonic: ', self.harmonics[i])
                first_half_maximum = np.max(y[int(half_measurements*0.1):half_measurements])
                global_minium = np.min(y)
                # print('first_half_maximum: ', first_half_maximum)
                index_first_half_maximum = np.argmax(y[int(half_measurements*0.1):half_measurements])
                index_first_half_maximum = int(half_measurements*0.1) + index_first_half_maximum

                index_second_max_first_half = np.argmax(y[int(half_measurements*0.05):index_first_half_maximum -int(half_measurements*0.05)])
                index_second_max_first_half_maximum = int(half_measurements*0.05) + index_second_max_first_half


                if index_second_max_first_half_maximum == index_first_half_maximum -int(half_measurements*0.05) - 1:
                    index_second_max_first_half_maximum = np.argmax(y[index_first_half_maximum + int(half_measurements*0.05):half_measurements])
                    index_second_max_first_half_maximum = index_first_half_maximum  + int(half_measurements*0.05) + index_second_max_first_half_maximum

                    temp = index_first_half_maximum
                    index_first_half_maximum = index_second_max_first_half_maximum
                    index_second_max_first_half_maximum = temp

                print('******')
                print('two highest peak separation in mV: ', (kwargs["xaxis"][index_first_half_maximum] - kwargs["xaxis"][index_second_max_first_half_maximum])*self.scan_rate*1e3)
                print('******')


                

                ax.plot(kwargs["xaxis"], y, color = colour_sequence[plot_counter], linestyle=line_style[plot_counter], label=plot_name, alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][index_first_half_maximum], y[index_first_half_maximum], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                ax.scatter(kwargs["xaxis"][index_second_max_first_half_maximum], y[index_second_max_first_half_maximum], color = colour_sequence[plot_counter], alpha=1-(plot_counter*kwargs["alpha_iterative_drop"]))
                plot_counter+=1
            if i==((num_harms)//2):
                ax.set_ylabel(kwargs["ylabel"], labelpad=8)
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