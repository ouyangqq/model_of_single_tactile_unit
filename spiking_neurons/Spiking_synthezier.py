# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:38:45 2019

@author: qiangqiang ouyang
"""


import numpy as np
import matplotlib.pyplot as plt
import time as Timestamp
from scipy import signal


def spiking_synthesizer(s_T,s_dt,_V=0.5):
    T       =   s_T/1000                     # total simulation length [s]
    dt      =   s_dt/1000                     # step size [s]
    t    =   np.arange(0, T+dt, dt)      # step values [s]
    Ta=0.005
    Kf=200
    steps=int(Ta/dt)
    Vin = np.zeros(len(t))
    Vs = np.zeros(len(t))
    Vf = np.zeros(len(t))
    Vr = np.zeros(len(t))
    Va=np.random.uniform(-70,-70,len(t))
    VN=butterworth_filter(1,np.random.uniform(-10,10,(t.size)),1000,'low',10e3) #mv
    Vin[int(0.1*T/dt):int((T-0.1*T)/dt)] = _V
    tc1=Timestamp.time()
    for ptime in range(1,t.size-1-steps):  #时间 t.size  
        Vs[ptime]=Vin[int(ptime/(Ta/dt))*int((Ta/dt))]
        fs=Vs[ptime]*Kf           
        Ts=1/(fs+1)
        Vf[ptime]=(1-fs*(ptime*dt%Ts))*(Vs[ptime]>0)
        Vr[ptime]=1*(Vf[ptime]>0.5)+0*(Vf[ptime]<=0.5)
        Va[ptime:ptime+steps]+=((Vr[ptime]-Vr[ptime-1])==1)*f_sp(t[0:steps],Ta)
        Va[ptime]+=VN[ptime]
    tc2=Timestamp.time()
    #plt.plot(t, Va[:], color='k',label="SS")[0]
    #plt.plot(t, 15*Vin,'-' , color='gray', label="Applied Current")[0]
    #line.set_color("deeppink")
    #line.set_color("k")
    #np.sin(2*np.pi*t[0:steps]/Ta)
    return tc2-tc1, [t,15*Vin,Va]



actionp_plus=(30-(-70)) # -70 reset voltage
actionp_minus=((-70)-(-100))    
'-----Define functions of typical stimulus waves-----'
def f_sp(x,Ta):
    tmp=np.sin(2*np.pi*x/Ta)
    return (actionp_plus*tmp*(tmp>=0)+actionp_minus*tmp*(tmp<0))


'-----butterworth filter function----'
def butterworth_filter(order,X,f,typ,fs):
    if(typ=='low'):
        w1=2*f/fs
        b, a = signal.butter(order, w1, typ)
    elif(typ=='high'):
        w1=2*f/fs
        b, a = signal.butter(order, w1, typ)
    elif(typ=='band'):
        w1=2*f[0]/fs
        w2=2*f[1]/fs
        b, a = signal.butter(order,[w1,w2], typ)
    return signal.filtfilt(b,a,X)  
             

