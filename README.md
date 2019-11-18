# model_of_single_tactile_unit
This is the code for the paper "A Python Code for Simulating Single Tactile Receptors and the Spiking Responses of Their Afferents"  
 ![image](https://github.com/ouyangqq/model_of_single_tactile_unit/blob/master/Saved_figs/diagram.jpg)  
The source code of current model was presented in the function "tactile_units_simulating()" in file of receptor.py, which correctly implements the diagram as illustrated in Fig 1(a) in the paper (see above figure). All the simulation results in section 3 of the paper were obtained by calling this funtion. 

The result figures in the paper and their corresponding code file are shown as follows:  

 
(1)  Intermediate_signals.py ---> Fig 3. Details of intermediate signals..  
(2)  Single_adaption.py--->Fig 4. Adaptation properties of each afferent type...  
(3)  Single_frequency_response.py --->Fig 5. The characteristics of frequency-threshold...  
(4)  Spike_Timing.py --->Fig 6. Evaluation of spike-timing precision for each afferent type.  
(5)  Computation_efficiency.py--->Fig 7. Evaluation of computation efficiency for simulating population units.  
(6)  Comparisons_spiking_neuron_models.py --->Fig 8. Performance comparison between different spiking neuron models.   

We also provide the ipynb files in folder "ipynbs" for all the simulation files above, which allow users see the results of running the code online.  

spikes_Saal_et_al.m  ---> Spking trains shown in top of in each row in Fig. 6 (a). The file must be put in 'docs' folder of Touchsim code which can be downloaded from the link: http://bensmaialab.org/code/touchsim/
