#frequency reponse
from sys import path
path.append(r'..//common/') 
import ultils as alt

import Receptors as receptorlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
filepath='/home/justin/share/figures_materials/single_receptor/'
filepath='saved_figs/'
color_bf=['g','b','r','k']
#filepath=''
'''均方误差根'''  
def rmse(y_test, y):  
    return np.sqrt(np.mean((y_test - y) ** 2))  
  
'''与均值相比的优秀程度，介于[0~1]。0表示不如均值。1表示完美预测.这个版本的实现是参考scikit-learn官网文档'''  
def R2(y_test, y_true):  
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()  
   
'''这是Conway&White《机器学习使用案例解析》里的版本'''  
def R22(y_test, y_true):  
    y_mean = np.array(y_true)  
    y_mean[:] = y_mean.mean()  
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)  
'''
ftSA1=pd.read_csv('../data/frequency_threshold_SA1.csv', sep=',', header=1) 
#ftSA2=pd.read_csv('frequency_threshold_SA2.csv', sep=',', header=1) 
ftRA1=pd.read_csv('../data/frequency_threshold_RA1.csv', sep=',', header=1) 
ftPC=pd.read_csv('../data/frequency_threshold_PC.csv', sep=',', header=1) 
ftHM=pd.read_csv('../data/frequency_threshold_HM.csv', sep=',', header=1) 
'''
C=np.hstack([alt.read_data('data/all_fr.txt',[1,2,3,4]),np.loadtxt('data/all_fr.txt')])
ftSA1=C[C[:,0]==3,1:3]
ftRA1=C[C[:,0]==2,1:3]
ftPC=C[C[:,0]==4,1:3]
ftHM=C[C[:,0]==1,1:3]
#obsevetaion
plt.figure(figsize=(7.5,4)) 
plt.subplot(1,2,1)




plt.xscale('log')
plt.yscale('log')
plt.plot(ftSA1[:,0],ftSA1[:,1],'k^',label=u'SA1',markersize=4)
plt.plot(ftRA1[:,0],ftRA1[:,1],'ks',label=u'RA1',markersize=4)
#plt.plot(ftSA2[:,0],ftSA2[:,1],'kd-',label=u'SA2')
plt.plot(ftPC[:,0],ftPC[:,1],'ko',label=u'PC',markersize=4)
plt.plot(ftHM[:,0],ftHM[:,1],'k+',markersize=6,label=u'Human threshold (HM)')
plt.title(u"(a)") 
plt.xticks([1,10,100,1000]) 
plt.yticks([0.01,0.1,1,10,100])  
plt.xlabel(u"Frequency(Hz)", fontproperties='')
plt.ylabel(u"Indentation(um)", fontproperties='')
plt.legend(prop={'family':'simSun','size':8}) 
#plt.savefig('fr_observation.png',bbox_inches='tight', dpi=600)


#prediction of our model
plt.subplot(1,2,2)
freq=np.linspace(1,1000,1000)
th=np.zeros((5,freq.size))
thall=np.zeros((5,freq.size))
fm=0
BW=0

def data_cal(x,tpye):
    global theta_buf
    global VL
    global N
    global Kb1
    global Kb2
    global Kb3
    global wbl
    global wbh
    global wl
    global As
    global Ku
    '''
    if (tpye=="SA1"):
        N=1
        Kb,Ku,wbl,wbh,wl,As=theta_buf[0,0:6]
    if (tpye=="RA1"):
        N=2
        Kb=theta_buf[5,0:N]
        Ku,wbl,wbh,wl,As=theta_buf[5,N:7]
    if (tpye=="PC"):
        N=3
        #Kb=theta_buf[5,0:N]
        #Ku,wbl,wbh,wl,As=theta_buf[10,N:8]
    #Kb[0]=0.9
    #Kb[1]=0.000
    #N=1
    '''
    #type_sel(tpye)
    tsensor=receptorlib.tactile_receptors(Ttype=tpye,simTime=0.2,sample_rate=100000,sample_num=1)
    #for i in range(int(freq.size)):
    #    thall[x,i]= 1e6* tsensor.VL/np.abs(receptorlib.transfer_func(1j*2*np.pi*freq[i],[tsensor.Kb1,tsensor.Kb2,tsensor.Kb3],tsensor.Ku,tsensor.wbl,tsensor.wbh,tsensor.wl,tsensor.N))/tsensor.As
    
    if(tpye=="SA1"):
        freq1=ftSA1[:,0]
    elif(tpye=="RA1"):
        freq1=ftRA1[:,0]
    elif(tpye=="PC"):
        freq1=ftPC[:,0]
    for i in range(int(freq1.size)):
        th[x,i]= 1e6* tsensor.VL/np.abs(receptorlib.transfer_func(1j*2*np.pi*freq1[i],[tsensor.Kb1,tsensor.Kb2,tsensor.Kb3],tsensor.Ku,tsensor.wbl,tsensor.wbh,tsensor.wl,tsensor.N))/tsensor.As
    
   
data_cal(0,"SA1")
#data_cal(1,"SA2")
data_cal(1,"RA1")
data_cal(2,"PC")

tpye="HM"
if(tpye=="HM"):
    freq1=ftHM[:,0]
    tsensor=receptorlib.tactile_receptors(Ttype="SA1",simTime=0.2,sample_rate=100000,sample_num=1)
    for i in range(int(freq1.size)):
        thall[0,i]= 1e6* tsensor.VL/np.abs(receptorlib.transfer_func(1j*2*np.pi*freq1[i],[tsensor.Kb1,tsensor.Kb2,tsensor.Kb3],tsensor.Ku,tsensor.wbl,tsensor.wbh,tsensor.wl,tsensor.N))/tsensor.As
        
    tsensor=receptorlib.tactile_receptors(Ttype="RA1",simTime=0.2,sample_rate=100000,sample_num=1)
    for i in range(int(freq1.size)):
        thall[1,i]= 1e6* tsensor.VL/np.abs(receptorlib.transfer_func(1j*2*np.pi*freq1[i],[tsensor.Kb1,tsensor.Kb2,tsensor.Kb3],tsensor.Ku,tsensor.wbl,tsensor.wbh,tsensor.wl,tsensor.N))/tsensor.As
       
    tsensor=receptorlib.tactile_receptors(Ttype="PC",simTime=0.2,sample_rate=100000,sample_num=1)
    for i in range(int(freq1.size)):
        thall[2,i]= 1e6* tsensor.VL/np.abs(receptorlib.transfer_func(1j*2*np.pi*freq1[i],[tsensor.Kb1,tsensor.Kb2,tsensor.Kb3],tsensor.Ku,tsensor.wbl,tsensor.wbh,tsensor.wl,tsensor.N))/tsensor.As  
    for i in range(int(freq1.size)):
            th[3,i]=np.min(thall[0:3,i])   
 
#for i in range(int(freq.size)):
#    thall[3,i]=np.min(thall[0:3,i])
            
plt.xscale('log')
plt.yscale('log')
R2_SA1=R2(np.log10(th[0,0:ftSA1[:,0].size]),np.log10( ftSA1[:,1]))
R2_RA1=R2(np.log10(th[1,0:ftRA1[:,0].size]), np.log10(ftRA1[:,1]))
R2_PC=R2(np.log10(th[2,0:ftPC[:,0].size]), np.log10(ftPC[:,1]))
R2_HM=R2(np.log10(th[3,0:ftHM[:,0].size]),np.log10( ftHM[:,1]))
plt.plot(ftSA1[:,0],th[0,0:ftSA1[:,0].size],color_bf[0]+'^-',markersize=4,
         label='SA1, '+"$\mathrm{R}^{2}$="+str(round(R2_SA1,2)))
plt.plot(ftRA1[:,0],th[1,0:ftRA1[:,0].size],color_bf[1]+'s-',markersize=4,
         label='RA1, '+"$\mathrm{R}^{2}$="+str(round(R2_RA1,2)))
plt.plot(ftPC[:,0],th[2,0:ftPC[:,0].size],color_bf[2]+'o-',markersize=4,
         label='PC , '+"$\mathrm{R}^{2}$="+str(round(R2_PC,2)))
plt.plot(ftHM[:,0],th[3,0:ftHM[:,0].size],color='gray',markersize=4,linewidth=2.5,
         label='HM , '+"$\mathrm{R}^{2}$="+str(round(R2_HM,2)))

print(R2_SA1,R2_RA1,R2_PC,R2_HM)
'''
plt.text(10,0.052,"$\mathrm{R}^{2}$="+str(round(R2_SA1,2)),color='k',fontsize=10)
plt.text(10,0.032,"$\mathrm{R}^{2}$="+str(round(R2_RA1,2)),color='k',fontsize=10)
plt.text(10,0.020,"$\mathrm{R}^{2}$="+str(round(R2_PC,2)),color='k',fontsize=10)
plt.text(10,0.012,"$\mathrm{R}^{2}$="+str(round(R2_HM,2)),color='k',fontsize=10)
'''
#plt.plot(freq[0:100],th[1,0:100],'c+-',label=u'SA2')
#plt.plot(freq[15:300],th[1,15:300],'b.-',label=u'RA1')
#plt.plot(freq[9:600],th[2,9:600],'r+-',label=u'PC')
#plt.plot(freq[0:600],th[3,0:600],'k--',linewidth=3,label=u'Human threshold')
plt.title(u"(b)") 
plt.xticks([1,10,100,1000]) 
plt.yticks([0.01,0.1,1,10,100])   
plt.xlabel(u"Frequency(Hz)", fontproperties='')
#plt.ylabel(u"Indentation(um)", fontproperties='')
plt.legend(prop={'family':'simSun','size':8}) 

plt.savefig(filepath+'frequency_response.png',bbox_inches='tight', dpi=600)
