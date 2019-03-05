# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:55:59 2017
Simulation of tactile units
@author: qiangqiang ouyang
"""
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import time as Timestamp
from scipy import signal
import matplotlib.pyplot as plt

class tactile_receptors():
    def __init__(self,Ttype,simTime,sample_rate,sample_num=1):
        self.Ttype=Ttype
        self.T=float (simTime) 
        self.dt = float (1/sample_rate)   #Timestep
        self.t= np.arange(0,self.T,self.dt)  
        self.Sn=sample_num
        self.stp=0
        self.Density=135
        self.sampled_num=[9,9]
        '-----skin mechanics-----'
        self.poisson_v=0.4
        self.E_ym=50*1e3  #pa,Young modulus 
        '-----neuron electronic-----'
        self.v_reset=-65*1e-3 #v
        self.VL=15*1e-3  #v
        '-----define variable arrays for all the signlas-----'       
        self.stimulus=np.zeros((sample_num,self.t.size))
        self.X0=np.zeros((sample_num,self.t.size))
        self.XM=np.zeros((sample_num,self.t.size))   #mechanic noise
        self.X1=np.zeros((sample_num,self.t.size))
        self.X2=np.zeros((sample_num,self.t.size))
        self.Sm=np.zeros((sample_num,self.t.size))
        self.Qg=np.zeros((sample_num,self.t.size))
        self.Vg=np.zeros((sample_num,self.t.size))
        self.Vf=np.zeros((sample_num,self.t.size))
        self.Vnf=np.zeros((sample_num,self.t.size))
        self.Vs=np.zeros((sample_num,self.t.size))
        #self.Vs[:,0]=1
        self.Vr=np.zeros((sample_num,self.t.size))
        self.Va=np.random.uniform(-70,-70,(sample_num,self.t.size))
        self.VN=np.zeros((sample_num,self.t.size))   #electric noise
        self.spike_trains=[]
        '-----Set parameters for each affect type-----' 
        if(self.Ttype=="SA1"):
            self.N=1
            self.Ku=0.096
            self.Kb1=0.205
            self.Kb2=0
            self.Kb3=0
            self.wbl=2*np.pi*8
            self.wbh=2*np.pi*10
            self.wl=2*np.pi*100
            self.As=3800
            self.c_plus=0
            self.Ta=0.004
            self.Kf=180
            
        elif (self.Ttype=="RA1"):
            self.N=2
            self.Ku=0.000
            self.Kb1=0.2316
            self.Kb2=0.00312
            self.Kb3=0.0
            self.wbl=2*np.pi*60
            self.wbh=2*np.pi*80
            self.wl=2*np.pi*0
            self.As=44000
            self.c_plus=0.0151
            self.Ta=0.004
            self.Kf=200
        elif (self.Ttype=="PC"):   
            self.N=3
            self.Ku=0
            self.Kb1=0
            self.Kb2=0.12782
            self.Kb3=0.00111
            self.wbl=2*np.pi*80
            self.wbh=2*np.pi*220
            self.wl=2*np.pi*0
            self.Ta=0.004
            self.c_plus=0.11
            self.As=360  #V/m
            self.Kf=300
        self.VH=999*1e-3 #v
        self.Th=1e6*self.VL/np.abs(transfer_func(1j*(self.wbl+self.wbh)/2,[self.Kb1,self.Kb2,self.Kb3],self.Ku,self.wbl,self.wbh,self.wl,self.N))/self.As  

    def tactile_units_simulating_previous_version(self,stimulus):  # the simulation code of previous version
        wd=6   #stepsize of taking Derivative
        self.stimulus=stimulus;
        #---define the temporary 2d array to storage the Intermediate result in the TCF
        X10=np.zeros((self.Sn,self.t.size))
        X11=np.zeros((self.Sn,self.t.size))
        X12=np.zeros((self.Sn,self.t.size))
        X13=np.zeros((self.Sn,self.t.size))
        if(self.N==1): self.X1=X11
        if(self.N==2): self.X1=X12
        if(self.N==3): self.X1=X13
        #self.X1=X10
        #inititate data_buf
        self.VN[:,:]=butterworth_filter(1,np.random.uniform(-5,5,(self.t.size)),1000,'low',10e3)+self.v_reset
        self.XM[:,:]=10e-6*butterworth_filter(1,np.random.uniform(-0.00,0.00,(self.t.size)),1000,'low',10e3)
        self.Va[:,:]=self.VN[:,:]
        self.X0[:,:]=self.stimulus[:,:]+self.XM[:,:]
        steps=int(self.Ta/self.dt)
        Start_time=Timestamp.time() #Start time stamp 
        for ptime in range(1,self.t.size-3*wd-steps):  #时间 t.size
            tmp=0
            tmp+=self.Kb1*(self.X0[:,ptime+wd]-self.X0[:,ptime])
            tmp+=self.Kb2*(self.X0[:,ptime+2*wd]-2*self.X0[:,ptime+1*wd]+self.X0[:,ptime])/(self.dt*wd)
            tmp+=self.Kb3*(self.X0[:,ptime+3*wd]-3*self.X0[:,ptime+2*wd]+3*self.X0[:,ptime+1*wd]-self.X0[:,ptime])/(self.dt*wd)**2
            X10[:,ptime+1] =(1-self.wbl*self.dt)*X10[:,ptime]+ tmp/wd
            if(self.N>=1):X11[:,ptime+1] =(1-self.wbh*self.dt)*X11[:,ptime]+self.wbh*(X10[:,ptime])*self.dt 
            if(self.N>=2):X12[:,ptime+1] =(1-self.wbh*self.dt)*X12[:,ptime]+self.wbh*(X11[:,ptime])*self.dt
            if(self.N>=3):X13[:,ptime+1] =(1-self.wbh*self.dt)*X13[:,ptime]+self.wbh*(X12[:,ptime])*self.dt  
            self.X2[:,ptime+1] = (1-self.wl*self.dt)*self.X2[:,ptime]+ self.Ku*self.wl*(self.X0[:,ptime])*self.dt
            self.Sm[:,ptime]=self.X1[:,ptime]+self.X2[:,ptime]
            self.Qg[:,ptime]=self.c_plus*np.abs(self.Sm[:,ptime])*(self.Sm[:,ptime]<0)+self.Sm[:,ptime]*(self.Sm[:,ptime]>=0)
            self.Vg[:,ptime]=self.As*self.Qg[:,ptime];  
            self.Vnf[:,ptime]=(self.Vg[:,ptime]-self.VL)*((self.Vg[:,ptime]>self.VL)&(self.Vg[:,ptime]<self.VH))+self.VH*(self.Vg[:,ptime]>=self.VH);
            self.Vs[:,ptime]=self.Vnf[:,int(ptime/(self.Ta/self.dt))*int((self.Ta/self.dt))] #smapling the singal of Vnf
            fs=self.Vs[:,ptime]*self.Kf
            Ts=1/(fs+1)
            self.Vf[:,ptime]=(1-fs*(ptime*self.dt%Ts))*(self.Vs[:,ptime]>0)
            self.Vr[:,ptime]=1*(self.Vf[:,ptime]>0.5)+0*(self.Vf[:,ptime]<=0.5) 
            for j in range(self.Sn):
                if(self.Vr[j,ptime]-self.Vr[j,ptime-1])==1:
                    self.Va[j,ptime:ptime+int(self.Ta/self.dt)]+=f_sp(self.t[0:int(self.Ta/self.dt)],self.Ta)#+self.VN[j,ptime:ptime+int(self.Ta/self.dt)]
        End_time=Timestamp.time()  #end time stamp 
        for ch in range(0,self.Sn): 
            spikes=[]
            for i in range(1,self.t.size): 
                if((self.Vr[ch,i]-self.Vr[ch,i-1])==1):spikes.append(i*self.dt)
            self.spike_trains.append(spikes)
        return End_time-Start_time # return the consuming time

    def tactile_units_simulating(self,stimulus):  # the simulation code of current version
        wd=10   #stepsize of taking Derivative
        self.stimulus=stimulus;
        #Define the temporary 2d array to storage the Intermediate result in the TCF---
        X10=np.zeros((self.Sn,self.t.size))
        X11=np.zeros((self.Sn,self.t.size))
        X12=np.zeros((self.Sn,self.t.size))
        X13=np.zeros((self.Sn,self.t.size))
        #inititate data_buf
        self.VN[:,:]=butterworth_filter(1,np.random.uniform(-10e-3,10e-3,(self.t.size)),1000,'low',10e3)+self.v_reset #mv
        self.XM[:,:]=1e-6*butterworth_filter(1,np.random.uniform(-0.1,0.1,(self.t.size)),1000,'low',10e3)
        self.Va[:,:]=self.VN[:,:]
        steps=int(self.Ta/self.dt)
        Start_time=Timestamp.time() #Start time stamp 
        self.X0[:,:]=self.stimulus[:,:]+self.XM[:,:]
        for ptime in range(1,self.t.size-3*wd-steps):  #t is the time stamp array with interval of dt 
            tmp=0    #wd is the time duration for computing derivation, dt is the sampling period 
            #----Two channel filter model----
            # BPF
            tmp+=self.Kb1*(self.X0[:,ptime+wd]-self.X0[:,ptime])  # Computing 1st derivation  
            tmp+=self.Kb2*(self.X0[:,ptime+2*wd]-2*self.X0[:,ptime+1*wd]+self.X0[:,ptime])/(self.dt*wd)  # Computing 2nd derivation  
            tmp+=self.Kb3*(self.X0[:,ptime+3*wd]-3*self.X0[:,ptime+2*wd]+3*self.X0[:,ptime+1*wd]-self.X0[:,ptime])/(self.dt*wd)**2  # Computing 3rd derivation
            X10[:,ptime+1] =(1-self.wbl*self.dt)*X10[:,ptime]+ tmp/wd  
            # LPF    #wbl=2*π*fbl    X10, X11, X12 and X13 are temporary variables
            if(self.N>=1):self.X1[:,ptime]=X11[:,ptime] =(1-self.wbh*self.dt)*X11[:,ptime-1]+self.wbh*(X10[:,ptime])*self.dt 
            if(self.N>=2):self.X1[:,ptime]=X12[:,ptime] =(1-self.wbh*self.dt)*X12[:,ptime-1]+self.wbh*(X11[:,ptime])*self.dt
            if(self.N>=3):self.X1[:,ptime]=X13[:,ptime] =(1-self.wbh*self.dt)*X13[:,ptime-1]+self.wbh*(X12[:,ptime])*self.dt
            self.X2[:,ptime] = (1-self.wl*self.dt)*self.X2[:,ptime-1]+ self.Ku*self.wl*(self.X0[:,ptime])*self.dt
            # Transducer 
            self.Sm[:,ptime]=self.X1[:,ptime]+self.X2[:,ptime]
            # Rectifier
            self.Qg[:,ptime]=self.c_plus*np.abs(self.Sm[:,ptime])*(self.Sm[:,ptime]<0)+self.Sm[:,ptime]*(self.Sm[:,ptime]>=0)
            self.Vg[:,ptime]=self.As*self.Qg[:,ptime];    
            # Normalizer  
            self.Vnf[:,ptime]=(self.Vg[:,ptime]-self.VL)*((self.Vg[:,ptime]>self.VL)&(self.Vg[:,ptime]<self.VH))+self.VH*(self.Vg[:,ptime]>=self.VH); 
            #----Spiking synthesizer----'
            self.Vs[:,ptime]=self.Vnf[:,int(ptime/(self.Ta/self.dt))*int((self.Ta/self.dt))] #smapling the singal of Vnf
            fc=self.Vs[:,ptime]*self.Kf; Tc=1/(fc+0.0001) #Computing frequency (fc) and period (Tc) of the carrier wave 0.0001 was added to avoid overfiting 
            self.Vf[:,ptime]=(1-fc*(ptime*self.dt%Tc))*(self.Vs[:,ptime]>0)  #Vs is modulated into the frequency-adjustable triangular wave
            self.Vr[:,ptime]=1*(self.Vf[:,ptime]>0.5)+0*(self.Vf[:,ptime]<=0.5)   #Comparing signal of Vnf  with the constant of  0.5
            #Generating action potential in Va by superposing the base wave from rising edge of the Vr
            self.Va[:,ptime:ptime+steps]+=np.mat((self.Vr[:,ptime]-self.Vr[:,ptime-1])==1).T*np.mat(f_sp(self.t[0:steps],self.Ta))
        End_time=Timestamp.time()  #End time stamp 
        self.spike_trains=[]
        '----Acquiring spikes from the signal of Vr-----'
        for ch in range(0,self.Sn): 
            spikes=[]
            for i in range(1,self.t.size): 
                if((self.Vr[ch,i]-self.Vr[ch,i-1])==1):spikes.append(i*self.dt)
            self.spike_trains.append(spikes)
        return End_time-Start_time #return the consuming time
    

actionp_plus=(30-(-70))*1e-3 # -70 reset voltage
actionp_minus=((-70)-(-100))*1e-3  
'-----Define functions of typical stimulus waves-----'
def f_sp(x,Ta):
    tmp=np.sin(2*np.pi*x/Ta)
    return (actionp_plus*tmp*(tmp>=0)+actionp_minus*tmp*(tmp<0))
def sin_wave(X,w,intentation):
    y=intentation*np.sin(w*X)
    return y
def triangular_wave(X,rate,intentation):
    T=2*intentation/(rate*(1e-3))
    y1=2*intentation*1e6*((2*X/T)-0.5)
    y2=2*intentation*1e6*np.floor(2*X/T)
    y3=(y1-y2)*(y1>y2)+(y2-y1)*(y1<y2)
    return y3*1e-6
def square_wave(X,rate,intentation):
    T=2*intentation/(rate*(1e-3))
    y1=intentation*((2*X/T)-0.5)
    y2=intentation*np.floor(2*X/T)
    y3=(rate*(1e-3))*(y1>y2)+(-rate*(1e-3))*(y1<y2)
    return y3
def step_wave(X,Tstart,Tend,rate,rate1,intentation):
    state1=rate*(1e-3)*(X-Tstart)*(X>Tstart)
    state2=state1+(intentation-state1)*(state1>intentation)
    state3=state2+(intentation+rate1*1e-3*(X-Tend)-state2)*(X>Tend)
    state4=state3+(0-state3)*(state3<0)
    return state4

'-----transfer_function of TCF ----'
def transfer_func(S,Kb,Ku,WBL,WBH,WL,Num):
    summ=0
    for i in range(Num):
       summ+=Kb[i]*S**(i+1) 
    return (summ/(S+WBL))*((WBH/(S+WBH))**(Num+1))+Ku*WL/(S+WL)    

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
             

