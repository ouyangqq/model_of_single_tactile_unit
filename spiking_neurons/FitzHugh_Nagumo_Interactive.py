__author__ = "Devrim Celik"

"""
Interative plot, showcasing the FitzHugh_Nagumo Model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import  Button, Slider
import time as timec
#==============================================================================#

def FitzHugh_Nagumo(s_T,s_dt,_I=0.5, a=0.7, b=0.8, tau=1/0.08):

    ######### Experimental Setup
    # TIME
    T       =       s_T                       # total simulation length
    dt      =       s_dt                      # step size
    time    =       np.arange(0, T+dt, dt)    # step values

    # CURRENT
    I = np.zeros(len(time))
    I[int(0.1*T/dt):int((T-0.1*T)/dt)] = _I

    # Memory
    V = np.empty(len(time))
    W = np.empty(len(time))

    # Initial Values
    V[0] = -0.7
    W[0] = -0.5


    for i in range(1, len(time)):
        #calculate membrane potential & resting variable
        V[i] = V[i-1] + (V[i-1]-(V[i-1]**3)/3 - W[i-1] + I[i])*dt
        W[i] = W[i-1] + ((V[i-1]+a - b*W[i-1])/tau)*dt

    return V, W

def I_values(s_T,s_dt,_I=0.5, time=None):
    I = np.zeros(len(time))
    I[int(0.1*s_T/s_dt):int((s_T-0.1*s_T)/s_dt)] = _I
    return I

#==============================================================================#

def start_FN_sim(s_T,s_dt):
    # time parameters for plotting
    T       =       s_T                       # total simulation length
    dt      =      s_dt                    # step size
    time    =       np.arange(0, T+dt, dt)    # step values

    # initial parameters
    a       = 0.7
    b       = 0.8
    tau     = 1/0.12
    I_init  = 0.5
    # update functions for lines
    tc1=timec.time()
    V, W = FitzHugh_Nagumo(T,dt,_I=I_init, a=a, b=b, tau=tau)
    I = I_values(T,dt,_I=I_init, time=time)
    tc2=timec.time()
    ######### Plotting
    #axis_color = 'lightgoldenrodyellow'

    #fig = plt.figure("FitzHugh-Nagumo Neuron", figsize=(14,7))
    #ax = fig.add_subplot(111)
    #plt.title("Interactive FitzHugh-Nagumo Neuron Simulation")
    #fig.subplots_adjust(left=0.1, bottom=0.32)
    
    # plot lines
    #line = plt.plot(time, 30*V,label="Membrane Potential")[0]
    #line3 = plt.plot(time, 10*I, '-' ,color='gray',label="Applied Current")[0]
    #line.set_color("purple")
    #line.set_color("k")
    #line2 = plt.plot(time, W, lw=0.3, label="Recovery Variable")[0]
    return tc2-tc1,[time,10*I,30*V]
    '''
    # add legend
    plt.legend(loc = "upper right")

    # add axis labels
    plt.ylabel("Potential [V]/ Current [A]")
    plt.xlabel("Time [s]")
    '''
    
    '''
    # define sliders (position, color, inital value, parameter, etc...)
    I_slider_axis = plt.axes([0.1, 0.17, 0.65, 0.03], facecolor=axis_color)
    I_slider = Slider(I_slider_axis, '$I_{ext}$', 0.0, 1.0, valinit=I_init)

    # update functions
    def update(val):
        V, W = FitzHugh_Nagumo(_I = I_slider.val)
        line.set_ydata(V)
        line2.set_ydata(W)
        line3.set_ydata(I_values(I_slider.val, time=time))

    # update, if any slider is moved
    I_slider.on_changed(update)

    # Add a button for resetting the parameters
    reset_button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')

    # event of resert button being clicked
    def reset_button_was_clicked(event):
        I_slider.reset()

    reset_button.on_clicked(reset_button_was_clicked)
    
    plt.show()
    '''
#==============================================================================#

if (__name__=='__main__'):
    tc=start_FN_sim(200,0.01)
    print(tc)