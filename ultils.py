import numpy as np
filepath='/home/justin/share/figures_materials/single_receptor/'
def read_data(filename,condition):  
    file=open(filename,'r')  
    buf=[]
    buf.append(-1)
    strbuf=file.readlines()  #
    for i in range(0,len(strbuf)): 
        if len(strbuf[i])==1:  
            buf.append(i)
    file.close()
    buf.append(len(strbuf))
    databuf=np.ones([buf[1]-buf[0]-1,1])*condition[0]
    for i in range(1,len(buf)-1): 
        databuf=np.vstack([databuf,np.ones([buf[i+1]-buf[i]-1,1])*condition[i]])
    return databuf
    #fnew.close()  
    
    
def read_data1(filename):  
    file=open(filename,'r')  
    buf=[]
    buf.append(-1)
    strbuf=file.readlines()  #
    for i in range(0,len(strbuf)): 
        if len(strbuf[i])==1:  
            buf.append(i)
    file.close()
    buf.append(len(strbuf))
    databuf=np.ones([buf[1]-buf[0]-1,1])
    return databuf
    #fnew.close()  
    
 
def read_data2(filename,condition):  
    file=open(filename,'r')  
    buf=[]
    buf.append(-1)
    strbuf=file.readlines()  #
    for i in range(0,len(strbuf)): 
        if len(strbuf[i])==1:  
            buf.append(i)
    file.close()
    buf.append(len(strbuf))
    databuf=np.ones([buf[1]-buf[0]-1,1])*condition[0]
    for i in range(1,len(buf)-1): 
        databuf=np.vstack([databuf,np.ones([buf[i+1]-buf[i]-1,1])*condition[i]])
    return databuf
    #fnew.close()     
    
def text_save(content,filename,mode):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])[1:len(str(content[i]))-1].replace('\n',' ')+'\n')
    file.close()
    
    
def text_read(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def loadData(path,split=' '):
    #data=list()
    res=list()
    with  open(path,'r') as fileReader:
        lines = fileReader.readlines()  # 读取全部内容
        for line in lines:
            data=[]
            line = line.strip()
            line = line.split(split)#根据数据间的分隔符切割行数据
            #data.append(line[:])
            for i in range(len(line)):
                if(line[i]!=''):
                    data.append(line[i])
            res.append(np.array(data).astype(float))  
    #data = data.astype(float)
    #np.random.shuffle(data)
    #label=data[:,0]
    #features=data[:,1:]
    #print("data loaded!")
    return res#features,label-1