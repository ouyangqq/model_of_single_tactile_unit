import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from sys import path
import ultils as alt

color_bf=['g','b','r','k']
numIterations=1000
space=200

def f_2(x,a,b,c): 
    return  a*((1/x)**b)+c 

def f_2(x,a,b,c): 
    return  a*((1/x)**b)+c 

def plot_cpy():
    #plt.figure(figsize=(2.2,1.8)) 
    order=[1,2,3,4]
    TSA1=[12,13,11,13]
    TRA1=[40,8,7.6,8.3]
    TPC=[50,20,5,3]
    plt.errorbar(order,TSA1,yerr=5.2,label=u'SA1',fmt=color_bf[0]+'^-',capsize=3,linewidth=1.2)
    plt.errorbar(order,TRA1,yerr=4.6,label=u'RA1',fmt=color_bf[1]+'s-',capsize=3,linewidth=1.2)
    plt.errorbar(order,TPC,yerr=3.4,label=u'PC',fmt=color_bf[2]+'o-',capsize=3,linewidth=1.2)
    plt.yticks([0,20,40,60],fontsize=10) 
    plt.xticks([1,2,3,4],fontsize=10)  
    plt.xlabel(u"Highest order of BPF (n)",fontsize=10)
    plt.ylabel(u"Fitting precision (ips) ",fontsize=12)
    plt.text(0,60,"(b)",fontsize=16)
    plt.legend(loc=0,prop={'family':'simSun','size':10}) 
 
def ploy_f(x,P): 
    res=0
    for i in range(0,len(P)):
        res=res+P[i]*x**(len(P)-i-1)
    return res


pbuf_SA1=[[0.8,0.8,0.05],[0.8,0.15,0.05],[0.8,0.1,0.05],[0.8,0.1,0.05]]
pbuf_RA1=[[3.2,0.62,0.05],[3.2,0.62,0.05],[3.2,0.62,0.05],[3.2,0.62,0.05]]
pbuf_PC=[[3.2,0.62,0.05],[3.2,0.62,0.05],[3.2,0.62,0.05],[3.2,0.62,0.05]]
def plot_f(ttype,cos,no,color):
    data_x=cos[cos[:,0]==no,1]
    data_y=cos[cos[:,0]==no,2]
    A=np.polyfit(data_x,data_y,6)
    x = np.arange(10,1000,50)   
    y=np.polyval(A,x)
    plt.plot(x,y,color,label=u'n='+str(no),markersize=4, linewidth=0.7)
    plt.title(ttype, fontsize=10)
    if (ttype=="SA1"):
        plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=10) 
        plt.ylabel(u"Training loss", fontsize=12)
    else:plt.yticks([0,0.5,1],color="none") 
    if(ttype=='SA1'):plt.text(-500,1,"(a)",fontsize=16)
    plt.xlabel(u"Iterations", fontsize=12)  
    plt.legend(loc=0,prop={'family':'simSun','size':8}) 
    #plt.savefig('Trianing.png',bbox_inches='tight', dpi=300)    
    
plt.figure(figsize=(14,3)) 

A=alt.read_data('data/training_loss_SA1.txt',[1,2,3,4])
cos_buf=np.loadtxt('data/training_loss_SA1.txt')
cos_buf=np.hstack([A,cos_buf])
plt.subplot(1,5,1)
for i in range(0,4):
    if(i==0):
        color=color_bf[0]+".-"
    elif(i==1):
        color=color_bf[0]+"^-"
    elif(i==2):
        color=color_bf[0]+"s-"
    elif(i==3):
        color=color_bf[0]+"o-"
    plot_f("SA1",cos_buf,i+1,color)

A=alt.read_data('data/training_loss_RA1.txt',[1,2,3,4])
cos_buf=np.loadtxt('data/training_loss_RA1.txt')
cos_buf=np.hstack([A,cos_buf])
plt.subplot(1,5,2)
for i in range(0,4):
    if(i==0):
        color=color_bf[1]+".-"
    elif(i==1):
        color=color_bf[1]+"^-"
    elif(i==2):
        color=color_bf[1]+"s-"
    elif(i==3):
        color=color_bf[1]+"o-"
    plot_f("RA1",cos_buf,i+1,color)

A=alt.read_data('data/training_loss_PC.txt',[1,2,3,4])
cos_buf=np.loadtxt('data/training_loss_PC.txt')
cos_buf=np.hstack([A,cos_buf])    
plt.subplot(1,5,3)
for i in range(0,4):
    if(i==0):
        color=color_bf[2]+".-"
    elif(i==1):
        color=color_bf[2]+"^-"
    elif(i==2):
        color=color_bf[2]+"s-"
    elif(i==3):
        color=color_bf[2]+"o-"
    plot_f("PC",cos_buf,i+1,color)
  
plt.subplot(1,3,3)   
plot_cpy()



filepath='saved_figs/'
plt.savefig(filepath+'fitting_error.jpg',bbox_inches='tight', dpi=600)



