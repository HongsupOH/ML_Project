from os import listdir
from os.path import isfile, join
import numpy as np
from collections import defaultdict
import pandas as pd
from pyevtk.hl import pointsToVTK
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# transform from txt file to numpy
def txt_to_data(filename):
    data = {}
    f = open(filename)
    lines = f.readlines()
    for ind,line in enumerate(lines):
        if '\n' in line:
            line = line.replace('\n','')
    
        row = line.split(" ")
        row = [x for x in row if x!='']
        row = row[1:]
        
        if ind == 0:
            label = row[:]
            for ind in label:
                data[ind] = []
        else:
            for ind,ele in enumerate(row):
                if ele == 'Void':
                    data[label[ind]].append(0)
                else:
                    data[label[ind]].append(float(ele))
    return data

def genGDATA(files):
    gdata = {}
    for ind,file in enumerate(files):
        curFile = txt_to_data('All_info/{}'.format(file))
        h = np.array(curFile['h'])
        gdata[h[0]] = txt_to_data('All_info/{}'.format(file))
    return gdata

# Generate global data
def genData(files):
    gdata = genGDATA(files)
    h_vals = list(gdata)
    for ind,ele in enumerate(h_vals):
        data = gdata[ele]
        c1 = np.array(data['c_1'])
        c2 = np.array(data['c_2'])
        h = np.array(data['h'])
        RB = np.array(data['RB'])
        
        left = np.append(c1.reshape(len(c1),1),c2.reshape(len(c2),1),axis=1)
        
        CurX = np.append(left,h.reshape(len(h),1),axis=1)
        CurY = RB.reshape(len(RB),1)
        
        if ind == 0:
            X = CurX
            Y = CurY
        else:
            X = np.append(X,CurX,axis = 0)
            Y = np.append(Y,CurY,axis = 0)

    return X,Y

def genDdata(files):
    gdata = genGDATA(files)
    h_vals = list(gdata)
    for ind,ele in enumerate(h_vals):
        data = gdata[ele]
        #c1_d
        c1 = np.array(data['c_1'])
        c1_d = []
        val_c1 = min(c1)
        while val_c1<0.30:
            c1_d+=[val_c1 for x in range(10)]
            val_c1+= 0.01
        c1_d+=[max(c1) for x in range(10)]
        c1_d = np.array(c1_d)
        #c2_d
        c2 = np.array(data['c_2'])
        c2_part = np.linspace(min(c2),max(c2),int(len(c1_d)/10))
        c2_part = c2_part.tolist()
        c2_d = []
        for _ in range(10):
            c2_d+=c2_part
        c2_d = np.array(c2_d)
        #h_d
        h = np.array(data['h'])
        h_d = np.array([h[0] for x in range(len(c1_d))])
        
        RB = np.array(data['RB'])
        
        left = np.append(c1_d.reshape(len(c1_d),1),c2_d.reshape(len(c2_d),1),axis=1)
        
        CurX = np.append(left,h_d.reshape(len(h_d),1),axis=1)
        
        if ind == 0:
            X = CurX
        else:
            X = np.append(X,CurX,axis = 0)

    return X

def Datamain():
    mypath = 'All_info'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    X,Y = genData(files)
    X_d = genDdata(files)
    return X,Y,X_d


def GenDenseData(c1,c2,h,agraph):
    c1_d = []
    val_c1 = min(c1)
    while val_c1<0.30:
        c1_d+=[val_c1 for x in range(100)]
        val_c1+= 0.01
    c1_d+=[max(c1) for x in range(100)]
    c1_d = np.array(c1_d)

    c2_part = np.linspace(min(c2),max(c2),int(len(c1_d)/100))
    c2_part = c2_part.tolist()
    c2_d = []
    for _ in range(100):
        c2_d+=c2_part
    c2_d = np.array(c2_d)
    h_d = np.array([h[0] for x in range(len(c1_d))])

    left_d = np.append(c1_d.reshape(len(c1_d),1),c2_d.reshape(len(c2_d),1),axis=1)
    X_d = np.append(left_d,h_d.reshape(len(h_d),1),axis=1)

    RB_hat_d = agraph.evaluate_equation_at(X_d)
    RB_hat_d = RB_hat_d.reshape(RB_hat_d.shape[0])
    return c1_d,c2_d,h_d,X_d,RB_hat_d


def GenCSVData(agraph):
    mypath = 'All_info'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    gdata = genGDATA(files)
    h_vals = list(gdata)
    for ind,h_val in enumerate(h_vals):
        data = gdata[h_val]
        # 1. call x,y and z data
        c1 = np.array(data['c_1'])
        c2 = np.array(data['c_2'])
        RB = np.array(data['RB'])
        h = np.array(data['h'])
        # 2. Make X
        left = np.append(c1.reshape(len(c1),1),c2.reshape(len(c2),1),axis=1)
        X = np.append(left,h.reshape(len(h),1),axis=1)
        # 3. Make RB_hat
        RB_hat = agraph.evaluate_equation_at(X).reshape(len(RB))
        c1_d,c2_d,h_d,X_d,RB_hat_d = GenDenseData(c1,c2,h,agraph)
        data_d = np.append(X_d,RB_hat_d.reshape((len(c1_d),1)),axis=1)
        # 4. Save data (c1,c2,h,RB_hat)
        left_data = np.append(X,RB.reshape((len(c1),1)),axis=1)
        data = np.append(left_data,RB_hat.reshape((len(c1),1)),axis=1)
        pd.DataFrame(data,columns=["c1","c2","h","RB","RB_hat"]).to_csv("results_csv/{}_h={}.csv".format(ind,h[0]))
        pd.DataFrame(data_d,columns=["c1","c2","h","RB_hat"]).to_csv("results_csv/dense_data/{}_h={}.csv".format(ind,h[0]))
        pointsToVTK("results_VTK/RB/data_ind_{}".format(ind),c1,c2,RB,data={'RB':RB})
        pointsToVTK("results_VTK/RB_hat/data_ind_{}".format(ind),c1,c2,RB_hat,data={'RB_hat':RB_hat})
        pointsToVTK("results_VTK/RB_hat_dense/data_ind_{}".format(ind),c1_d,c2_d,RB_hat_d,data={'RB_hat_d':RB_hat_d})
        
def helpPlot(agraph):
    mypath = 'All_info'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    gdata = genGDATA(files)
    h_vals = list(gdata)
    for ind,h_val in enumerate(h_vals):
        data = gdata[h_val]
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection='3d')

        c1 = np.array(data['c_1'])
        c2 = np.array(data['c_2'])
        LB = data['LB']
        UB = data['UB']
        h = np.array(data['h'])
        RB = np.array(data['RB'])
       
        left = np.append(c1.reshape(len(c1),1),c2.reshape(len(c2),1),axis=1)

        X = np.append(left,h.reshape(len(h),1),axis=1)
        Y = RB
        
        RBhat = agraph.evaluate_equation_at(X)
        RBhat = RBhat.reshape(RBhat.shape[0])

        c1_d,c2_d,h_d,X_d,RBhat_d = GenDenseData(c1,c2,h,agraph)
        
        surf1 = ax.plot_trisurf(c2,c1,RB,color = 'blue',label = 'TRUE',alpha = 0.5)
        scat1 = ax.scatter(c2,c1,RB,color = 'blue',s = 100,edgecolors = "black",linewidths = 1,alpha = 0.5)
        surf1._facecolors2d = surf1._facecolor3d 
        surf1._edgecolors2d = surf1._edgecolor3d
        
        surf2 = ax.plot_trisurf(c2,c1,RBhat,color = 'red',edgecolor="black",linewidths = 1, label = 'BINGO',alpha = 0.5)
        scat2 = ax.scatter(c2,c1,RBhat,color = 'red',s = 100,edgecolors = "black",linewidths = 1)
        surf2._facecolors2d = surf2._facecolor3d 
        surf2._edgecolors2d = surf2._edgecolor3d

        surf3 = ax.plot_trisurf(c2_d,c1_d,RBhat_d,color='red', label = 'BINGO_Dense',alpha=0.5)
        surf3._facecolors2d = surf3._facecolor3d
        surf3._edgecolors2d = surf3._edgecolor3d
        
        ax.set_xlabel('c2')
        ax.set_ylabel('c1')
        ax.set_zlabel('RB')
        ax.set_zlim(0, 1)
        plt.legend()
        plt.title('h = {} and L0 = {}'.format(data['h'][0],data['L_0'][0]))
        plt.grid()
        plt.savefig("results_plots/{}_h={}.pdf".format(ind, h_val))

print(Datamain())
