#frequency reponse
import pyspike as spk
from sys import path
import Receptors as receptorlib
path.append(r'..//common/') 
import ultils as alt
import Receptors as receptorlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
condition=[1,2,3,4,5]

#A=list(filter(lambda x: x >= 6, sine_spike[:,0]))
#conn= MySQLdb.connect(host='localhost', user='root',  passwd='7800125', db1='db1')
#data_s_SA1=alt.read_data('../data/tspike_SA1.txt',condition)
data_s_SA1=np.loadtxt('data/tspike_SA1.txt')
data_s_RA1=np.loadtxt('data/tspike_RA1.txt')
data_s_PC=np.loadtxt('data/tspike_PC.txt')
data_s_SA1_avg=np.zeros([20,2])
data_s_RA1_avg=np.zeros([20,2])
data_s_PC_avg=np.zeros([19,2])
for i in range(20):
    data_s_SA1_avg[i,0]=np.average([data_s_SA1[i+0,0],data_s_SA1[i+20,0],data_s_SA1[i+40,0],data_s_SA1[i+60,0],data_s_SA1[i+80,0]])
    data_s_SA1_avg[i,1]=np.average([data_s_SA1[i+0,1],data_s_SA1[i+20,1],data_s_SA1[i+40,1],data_s_SA1[i+60,1],data_s_SA1[i+80,1]])
for i in range(20):
    data_s_RA1_avg[i,0]=np.average([data_s_RA1[i+0,0],data_s_RA1[i+20,0],data_s_RA1[i+40,0],data_s_RA1[i+60,0],data_s_RA1[i+80,0]])
    data_s_RA1_avg[i,1]=np.average([data_s_RA1[i+0,1],data_s_RA1[i+20,1],data_s_RA1[i+40,1],data_s_RA1[i+60,1],data_s_RA1[i+80,1]])
space=19
for i in range(19):
    data_s_PC_avg[i,0]=np.average([data_s_PC[i+0,0],data_s_PC[i+space,0],data_s_PC[i+space*2,0],data_s_PC[i+space*3,0],data_s_PC[i+space*4,0]])
    data_s_PC_avg[i,1]=np.average([data_s_PC[i+0,1],data_s_PC[i+space,1],data_s_PC[i+space*2,1],data_s_PC[i+space*3,1],data_s_PC[i+space*4,1]])

def wave_double_set(tsensor,st1,st2,st3,x1,x2,x3):
    global rate
    global rate1
    global Tstart
    global Tend
    rate=30  #mm/s
    rate1=-10 
    intentation=x1*1e-6 #um
    Tstart=st1
    Tend=Tstart+0.25
    w=2*np.pi*20
    tsensor.stimulus[tsensor.stp,int(Tstart/tsensor.dt):int(Tend/tsensor.dt)]+=receptorlib.sin_wave(tsensor.t,w,intentation)[0:int(Tend/tsensor.dt)-int(Tstart/tsensor.dt)]

    intentation=x2*1e-6 #um
    Tstart=st2
    Tend=Tstart+0.25
    tsensor.stimulus[tsensor.stp,int(Tstart/tsensor.dt):int(Tend/tsensor.dt)]+=receptorlib.sin_wave(tsensor.t,w,intentation)[0:int(Tend/tsensor.dt)-int(Tstart/tsensor.dt)]
   
    intentation=x3*1e-6 #um
    Tstart=st3
    Tend=Tstart+0.25
    tsensor.stimulus[tsensor.stp,int(Tstart/tsensor.dt):int(Tend/tsensor.dt)]+=receptorlib.sin_wave(tsensor.t,w,intentation)[0:int(Tend/tsensor.dt)-int(Tstart/tsensor.dt)]
    
#butterworth_filter(order,x,fl,typ,fs):
    
sine_spike=np.loadtxt('data/sinuous_spike.txt')
sine_spike_PC=np.zeros((2000,2))
sine_spike_SA1=np.zeros((2000,6))
sine_spike_RA1=np.zeros((2000,2))
#sine_stimulus=np.zeros((2000,2))

#sine_stimulus=np.loadtxt('sine_stimulus.txt')
#t=np.linspace(0,1,1000)

#sine_stimulus[:,1]=receptorlib.butterworth_filter(1,sine_stimulus[:,1],2000,'low',10e3)

for i in range(1221):
    if (sine_spike[i,1]>7000)&(sine_spike[i,1]<10000):
        #if (sine_spike[i,0]>-10)&(sine_spike[i,0]<300):
        sine_spike_SA1[i,0]=sine_spike[i,0]
        sine_spike_SA1[i,1]=sine_spike[i,1]

    if (sine_spike[i,1]>2000)&(sine_spike[i,1]<4000):
        sine_spike_PC[i,0]=sine_spike[i,0]
        sine_spike_PC[i,1]=sine_spike[i,1]
    
    if (sine_spike[i,1]>4000)&(sine_spike[i,1]<5000):
        sine_spike_RA1[i,0]=sine_spike[i,0]
        sine_spike_RA1[i,1]=sine_spike[i,1]
    '''
    if (sine_spike[i,1]>-1000)&(sine_spike[i,1]<1000):
        sine_stimulus[i,0]=sine_spike[i,0]
        sine_stimulus[i,1]=sine_spike[i,1]  
        '''


tsp=np.arange(0.1,10,0.5) #ms
ty=np.arange(-0.1,0.2,0.01) #ms



def get_spike_trains():
    sim_spike_buf=[]
    measure_spike_buf=[]
    spike_stimulus=[]
    tsensor=receptorlib.tactile_receptors(Ttype='SA1',simTime=1,sample_rate=20000,sample_num=1)
    wave_double_set(tsensor,0,0.370,0.730,35,130,250) 
    for i in range(5):    
        tsensor.tactile_units_simulating(tsensor.stimulus)
        tmp=tsensor.spike_trains[tsensor.stp][:]
        sim_spike_buf.append(tmp*1000)
        
    tsensor=receptorlib.tactile_receptors(Ttype='RA1',simTime=1,sample_rate=20000,sample_num=1)
    wave_double_set(tsensor,0,0.370,0.730,35,130,250) 
    for i in range(5):    
        tsensor.tactile_units_simulating(tsensor.stimulus)
        tmp=tsensor.spike_trains[tsensor.stp][:]
        sim_spike_buf.append(tmp*1000)
        
    tsensor=receptorlib.tactile_receptors(Ttype='PC',simTime=1,sample_rate=20000,sample_num=1)
    wave_double_set(tsensor,0,0.370,0.730,35,130,250) 
    for i in range(5):    
        tsensor.tactile_units_simulating(tsensor.stimulus)
        tmp=tsensor.spike_trains[tsensor.stp][:]
        sim_spike_buf.append(tmp*1000)   
        
    spike_stimulus.append(1e3*tsensor.t)
    spike_stimulus.append(1e6*tsensor.stimulus[tsensor.stp,:])
    np.save('data/spike_stimulus.npy',spike_stimulus)  
    np.save('data/sim_spike_buf.npy',sim_spike_buf)  
    alt.text_save(sim_spike_buf,'data/sim_spike_buf.txt','w')
    
    x=sine_spike_SA1[0:88,0]
    y=sine_spike_SA1[0:88,1]
    y=(y-np.min(y))/(np.max(y)-np.min(y))
    A=np.vstack((x,y)).T   
    B=A[(A[:,1]>0.8)&(A[:,1]<1.1)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.6)&(A[:,1]<0.8)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.4)&(A[:,1]<0.6)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.2)&(A[:,1]<0.4)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>-0.1)&(A[:,1]<0.2)][:,0]
    measure_spike_buf.append(B)
    
    
    x=sine_spike_RA1[89:260,0]
    y=sine_spike_RA1[89:260,1]
    y=(y-np.min(y))/(np.max(y)-np.min(y))
    A=np.vstack((x,y)).T   
    B=A[(A[:,1]>0.8)&(A[:,1]<1.1)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.6)&(A[:,1]<0.8)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.4)&(A[:,1]<0.6)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.2)&(A[:,1]<0.4)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>-0.1)&(A[:,1]<0.2)][:,0]
    measure_spike_buf.append(B)
    
    
    x=sine_spike_PC[261:744,0]
    y=sine_spike_PC[261:744,1]
    y=(y-np.min(y))/(np.max(y)-np.min(y))
    A=np.vstack((x,y)).T   
    B=A[(A[:,1]>0.8)&(A[:,1]<1.1)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.6)&(A[:,1]<0.8)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.4)&(A[:,1]<0.6)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>0.2)&(A[:,1]<0.4)][:,0]
    measure_spike_buf.append(B)
    B=A[(A[:,1]>-0.1)&(A[:,1]<0.2)][:,0]
    measure_spike_buf.append(B)
    
    
    np.save('data/measure_spike_buf.npy',[measure_spike_buf[0:5],measure_spike_buf[5:10],measure_spike_buf[10:15]])  
    alt.text_save(measure_spike_buf,'data/measure_spike_buf.txt','w')
    #np.savetxt('../data/sim_spike_buf.txt',sim_spike_buf)



get_spike_trains()
sim_spike_buf=np.load('sim_spike_buf.npy') 
measure_spike_buf=np.load('measure_spike_buf.npy') 
#spike_stimulus_buf=[1000*tsensor.t,1e6*tsensor.stimulus[tsensor.stp,:]]
#np.save('spike_stimulus.npy',spike_stimulus_buf) 
spike_stimulus=np.load('spike_stimulus.npy') 
sim_spiketrains = spk.load_spike_trains_from_txt("data/sim_spike_buf.txt",edges=(0, 1000))
measure_spiketrains = spk.load_spike_trains_from_txt("data/measure_spike_buf.txt",edges=(0, 1000))

     
def plot_f():
    plt.figure(figsize=(8,5))
    #plt.plot(tsensor.t,tsensor.Vnf[tsensor.stp,:])
    #plt.plot(tsensor.t,1000*tsensor.stimulus[tsensor.stp,:])
    size=0.3
    #----PC---#   
    ax1=plt.subplot(711) 
    ax1.spines['top'].set_color('None')
    ax1.spines['right'].set_color('None') 
    for i in range(5): 
        x=measure_spike_buf[0][i][:]
        plt.scatter(x,(1+0.8-i*0.15)*np.ones(len(x)),color='gray',marker='o',s=size)
        x=sim_spike_buf[i+10][:]
        plt.scatter(x,(0.8-i*0.15)*np.ones(len(x)),color='r',marker='o',s=size)
    plt.yticks([0,2],color='none')
    plt.xticks([])
    plt.text(-120,1,"PC",fontsize=10)
    
    #----RA1---#
    ax2=plt.subplot(712,sharex=ax1) 
    ax2.spines['top'].set_color('None')
    ax2.spines['right'].set_color('None')
    for i in range(5): 
        x=measure_spike_buf[1][i][:]
        plt.scatter(x,(1+0.8-i*0.15)*np.ones(len(x)),color='gray',marker='o',s=size)
        x=sim_spike_buf[i+5][:]
        plt.scatter(x,(0.8-i*0.15)*np.ones(len(x)),color='b',marker='o',s=size)
    plt.yticks([0,2],color='none')
    plt.xticks([])
    plt.text(-120,1,"RA1",fontsize=10)
    
    #----PC---#
    ax3=plt.subplot(713,sharex=ax1) 
    ax3.spines['top'].set_color('None')
    ax3.spines['right'].set_color('None')
    for i in range(5): 
        x=measure_spike_buf[2][i][:]
        plt.scatter(x,(1+0.8-i*0.15)*np.ones(len(x)),color='gray',marker='o',s=size)
        x=sim_spike_buf[i][:]
        plt.scatter(x,(0.8-i*0.15)*np.ones(len(x)),color='g',marker='o',s=size)
    plt.yticks([0,2],color='none')
    plt.xticks([])
    plt.text(-120,1,"SA1",fontsize=10)

    spk.isi_profile(sim_spiketrains[0],measure_spiketrains[0])


    axes=plt.subplot(714,sharex=ax1) 
    axes.spines['top'].set_color('None')
    axes.spines['right'].set_color('None')
    plt.plot(spike_stimulus[0],spike_stimulus[1],'k',linewidth=1)
    plt.ylabel("Stimulus (um)",fontsize=8)
    plt.xticks([0,200,400,600,800,1000])
    plt.yticks([-150,0,150],fontsize=6)
    plt.xlabel("Time (ms)")     
    
   
    axes=plt.subplot(4,3,10)
    plt.title('SA1')
    plt.text(-2.5,0.4,"(b)",fontsize=12)
    axes.set_ylabel("Norm. dist. diff",fontsize=10)
    axes.spines['top'].set_color('None')
    axes.spines['right'].set_color('None')
    plt.plot(tsp,np.zeros(tsp.size),'k--',linewidth=1)
    space=20
    #plt.errorbar(tsp,spike_dis_SA1,yerr=spike_dis_SA1_err,label=u'SA1',fmt='g.-',linewidth=0.7,capsize=1.5)
    plt.plot(data_s_SA1[0:space,0],data_s_SA1[0:space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_SA1[1*space:2*space,0],data_s_SA1[1*space:2*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_SA1[2*space:3*space,0],data_s_SA1[2*space:3*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_SA1[3*space:4*space,0],data_s_SA1[3*space:4*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_SA1[4*space:5*space,0],data_s_SA1[4*space:5*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_SA1_avg[0:space,0],data_s_SA1_avg[0:space,1],'k-',linewidth=1.5)
    precision=6.98
    plt.plot(precision*np.ones(ty.size),ty,'k--',linewidth=1)
    plt.xticks([0,2,4,6,8,10],fontsize=8)
    plt.yticks([-0.1,0,0.1,0.2,0.3],fontsize=8,color='k')
    plt.text(precision,0.2,str(precision))

    
    axes=plt.subplot(4,3,11)
    plt.title('RA1')
    axes.spines['top'].set_color('None')
    axes.spines['right'].set_color('None')
    axes.set_xlabel("Jitter SD (ms)")
    plt.plot(tsp,np.zeros(tsp.size),'k--',linewidth=1)
    #plt.errorbar(tsp,spike_dis_RA1,yerr=spike_dis_RA1_err,label=u'RA1',fmt='b.-',linewidth=0.7,capsize=1.5)  
    plt.plot(data_s_RA1[0:space,0],data_s_RA1[0:space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_RA1[1*space:2*space,0],data_s_RA1[1*space:2*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_RA1[2*space:3*space,0],data_s_RA1[2*space:3*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_RA1[3*space:4*space,0],data_s_RA1[3*space:4*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_RA1[4*space:5*space,0],data_s_RA1[4*space:5*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_RA1_avg[0:space,0],data_s_RA1_avg[0:space,1],'k-',linewidth=1.5)
    precision=4.89
    plt.plot(precision*np.ones(ty.size),ty,'k--',linewidth=1)
    plt.xticks([0,2,4,6,8,10],fontsize=8)
    plt.yticks([-0.1,0,0.1,0.2,0.3],fontsize=8)
    plt.text(precision,0.2,str(precision))

    axes=plt.subplot(4,3,12) 
    plt.title('PC')
    axes.spines['top'].set_color('None')
    axes.spines['right'].set_color('None')
    plt.plot(tsp,np.zeros(tsp.size),'k--',linewidth=1)
    #plt.errorbar(tsp,spike_dis_PC,yerr=spike_dis_PC_err,label=u'PC',fmt='r.-',linewidth=0.7,capsize=1.5)
    space=19
    plt.plot(data_s_PC[0:space,0],data_s_PC[0:space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_PC[1*space:2*space,0],data_s_PC[1*space:2*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_PC[2*space:3*space,0],data_s_PC[2*space:3*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_PC[3*space:4*space,0],data_s_PC[3*space:4*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_PC[4*space:5*space,0],data_s_PC[4*space:5*space,1],'lightgray',linewidth=1.5)
    plt.plot(data_s_PC_avg[0:space,0],data_s_PC_avg[0:space,1],'k-',linewidth=1.5)
    precision=2.53
    plt.plot(precision*np.ones(ty.size),ty,'k--',linewidth=1)
    plt.yticks([-0.1,0,0.1,0.2,0.3],fontsize=8)
    plt.xticks([0,2,4,6,8,10],fontsize=8)
    plt.text(precision,0.2,str(precision))
    
    plt.text(7,0.021,'↑Worse')
    plt.text(7,-0.06,'↓Better')
    #plt.legend(['SA1','RA1','PC','Stimulus wave'],loc = 'Best',ncol=4,bbox_to_anchor=(0.15,1))
    #plt.plot(1000*t[0:250],35*np.sin(2*np.pi*20*t[0:250]),color='k')
    filepath='/home/justin/share/figures_materials/single_receptor/'
    filepath='saved_figs/'
    plt.savefig(filepath+'tspike.png',bbox_inches='tight', dpi=600)   


plot_f()
