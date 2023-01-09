"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
ColorMapFEM.py

Date:    
28.12.2022

Developmed by:
-Ph.D. Candidate German Solorzano
Supervised by:
-Dr. Vagelis Plevris

Sponsored by:
Oslo Metropolitan University, Oslo, Norway.
Department of Civil Engineering and Energy Technology 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Header:
# This script contains various function to visualize the results of the FEM analysis   
# The analysis results are recorded in text files and then this routines can be used to plot stress and strain fields
# the visuallization functions in this file are mostly adaptted for 2D RC shear walls. However, it can be easiliy modified for other elements (even 3D solid models) 

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# plot the color map of a given quantity
# file -> the results as stored by the OpeenSees recorder (each row is one analysis step that contains the information for all the elements/nodes)
# comp -> the component of the stress/strain (for example for stress, sx=1, sy=2)
# scale -> the map is plotted on the deformed shape, so a scale for the deformations must be given

def colorMap(ops,nv,nh,file,comp, scale=100, title="",row=-1):
    
    # get the average at the nodes (the numering is the same as it is in the opensees model)
    avg = getNodalAverage(ops,nv,nh,file,comp,row)
  
    # x coordinates
    x = []
    # y coordinates
    y = []
    # z coordinates
    z = [] 
    # triangles indices
    t = []
    # color value
    v = []
    
    nodes = ops.getNodeTags()         
    for n in nodes:
        coords = ops.nodeCoord(n)
        disp = ops.nodeDisp(n)
        coords[0] = coords[0] + disp[0]*scale
        coords[1] = coords[1] + disp[1]*scale
        coords[2] = coords[2] + disp[2]*scale
        x.append(coords[0])
        y.append(coords[1]) 
        z.append(coords[2])
        v.append(avg[n-1])
      
    hLines=nh+1
    for k in range(nh):
        for i in range(nv):
            n1 = hLines*(i)+k
            n2 = hLines*(i)+k+1
            n3 = hLines*(i+1)+k+1
            n4 = hLines*(i+1)+k
            triangle1 = [n1,n2,n3]  
            triangle2 = [n1,n4,n3]
            t.append(triangle1)
            t.append(triangle2)
        
    t = np.asarray(t)     
    minValue = min(v)
    maxValue = max(v)
    p = (maxValue-minValue) * 0.15
    
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.tripcolor(x,y,t,v,edgecolors='black',shading='gouraud',cmap=cm.jet,vmin=minValue-p, vmax=maxValue+p)
    plt.colorbar()
    plt.title(title)

    plt.show()
    
    
  

def getForceAverage(ops,nv,nh, dof):
    
    lastIndex = nh*nv;
    hLines = nh+1
    vLines = nv+1
    
    elements = ops.getEleTags()[0:lastIndex] 
    
    # wich indices from the element vector force should be used for averaging?
    indices = []
    for i in range(4):
        indices.append(i*6+dof)
    nodalValues = np.zeros((hLines*vLines,2));
 
    # do the averaging first
    for e in elements:
        forces = ops.eleForce(e)
        nodes = ops.eleNodes(e)
        # average at node
        for i in range(4):
            nodeIndex = nodes[i]-1
            nodalValues[nodeIndex,0] = nodalValues[nodeIndex,0] + forces[indices[i]];
            nodalValues[nodeIndex,1] = nodalValues[nodeIndex,1] + 1
    
    avg = np.zeros(hLines*vLines); 
    for i in range(len(avg)):
        avg[i] = nodalValues[i,0]/nodalValues[i,1]
    
    return avg;
 
def getForceAverage2(ops,nv,nh, dof):
    
    lastIndex = nh*nv;
    hLines = nh+1
    vLines = nv+1
    
    elements = ops.getEleTags()[0:lastIndex] 
    
    # wich indices from the element vector force should be used for averaging?
    indices = []
    for i in range(4):
        indices.append(i*6+dof)
    nodalValues = np.zeros((hLines*vLines));
 
    # do the averaging first
    for e in elements:
        forces = ops.eleForce(e)
        nodes = ops.eleNodes(e)
        # average at node
        for i in range(4):
            nodeIndex = nodes[i]-1
            nodalValues[nodeIndex] = nodalValues[nodeIndex] + forces[indices[i]];

    return nodalValues;     
 
    
# perform the averaging of the stress/strain at the nodes
def getNodalAverage(ops,nv,nh,file,comp,row=-1):
    
    nEles = nh*nv;
    hLines = nh+1
    vLines = nv+1
    
    test = np.loadtxt(file, delimiter=' ', unpack="False")


    #ncols = len(test[0])
    #print("ncols",ncols)
    
    test = test[:,row]
    #print(test)
    
    skip = 32
    stress = np.zeros((nEles,skip))
    for i in range(nEles):
        c = skip*i
        stress[i,:] = test[c:c+skip]
  
    # wich indices from the element vector force should be used for averaging?
    indices = []
    for i in range(4):
        indices.append(i*8+ comp)

    elements = ops.getEleTags()[0:nEles] 
    nodalValues = np.zeros((hLines*vLines,2));

    # do the averaging first
    for e in elements:
        #forces = ops.eleForce(e)
        nodes = ops.eleNodes(e)
        # average at node
        for i in range(4):
            nodeIndex = nodes[i]-1
            nodalValues[nodeIndex,0] = nodalValues[nodeIndex,0] + stress[e-1,indices[i]];
            nodalValues[nodeIndex,1] = nodalValues[nodeIndex,1] + 1
    
    avg = np.zeros(hLines*vLines); 
    for i in range(len(avg)):
        avg[i] = nodalValues[i,0]/nodalValues[i,1]
        
    return avg    


# plot the progression of the stress/strain field in multi-plot figure
# the routines requires the output file of the nodal displacements (must contain only dof 1 and 2), and the strain/stress field and its corresponding component to be mapped..
def colorMapVarious(ops,nv,nh,fileStrain,fileDisp,comp,scale=100, title=""):
    
    # change the text size and style
    plt.rcParams.update({'font.size': 14})
    plt.rc('font', family='TimesNewRomman')
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=10) #controls default text size
    plt.rc('axes', titlesize=10) #fontsize of the title
    plt.rc('axes', labelsize=10) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
    plt.rc('legend', fontsize=10) #fontsize of the legend
    
    # create the subplots
    figure, ax = plt.subplots(2, 3)
    
    ## add a title to the figure
    figure.suptitle(title, fontsize=16)
    
    dataStrain = np.loadtxt(fileStrain, delimiter=' ', unpack="False")
    dataDisp = np.loadtxt(fileDisp, delimiter=' ', unpack="False")
    
    # total number of values
    ncols = len(dataStrain[0])
    
    # define some stations at which the field is going to be plotted
    # the size of this array must correspond to the number of subplots
    stations = [0, 
                0.05,
                0.1,
                0.2,
                0.50,
                1]

    # store the maximum horizontal displacemt
    maxDx = 0
    count = 0
    for jr in range(2):
        for jc in range(3):
            
            index = int(stations[count]*ncols)
            if index>0:
                index = index-1
            
            avg = getNodalAverage(ops,nv,nh,fileStrain,comp,index)
            dispRow = dataDisp[:,index]
            
            skip = 2
            nNodes = (nv+1)*(nh+1)
            nodeDisp = np.zeros((nNodes,skip))
            for t in range(nNodes):
                c = skip*t
                nodeDisp[t,:] = dispRow[c:c+skip]
             
            # x coordinates
            x = []
            # y coordinates
            y = []
            # z coordinates
            z = [] 
            # triangles indices
            t = []
            # color value
            v = []
            nodes = ops.getNodeTags()         
            for n in nodes:
                coords = ops.nodeCoord(n)
                disp = ops.nodeDisp(n)
                dx = nodeDisp[n-1,0]
                dy = nodeDisp[n-1,1]
                dz = 0
                coords[0] = coords[0] + dx*scale
                coords[1] = coords[1] + dy*scale
                coords[2] = coords[2] + dz*scale
                
                if dx>maxDx:
                    maxDx = dx
                
                x.append(coords[0])
                y.append(coords[1]) 
                z.append(coords[2])
                v.append(avg[n-1])
              
            hLines=nh+1
            for k in range(nh):
                for i in range(nv):
                    n1 = hLines*(i)+k
                    n2 = hLines*(i)+k+1
                    n3 = hLines*(i+1)+k+1
                    n4 = hLines*(i+1)+k
                    triangle1 = [n1,n2,n3]  
                    triangle2 = [n3,n4,n1]
                    t.append(triangle1)
                    t.append(triangle2)
                
    
            t = np.asarray(t)     
            minValue = min(v)
            maxValue = max(v)
            p = (maxValue-minValue) * 0.15
             
            plot = ax[jr,jc].tripcolor(x,y,t,v,edgecolors='black',shading='gouraud',cmap=cm.jet,vmin=minValue-p, vmax=maxValue+p)
            ax[jr,jc].set_title("dx = "+ str( round(stations[count]*maxDx*1000,3)) +" (" +str(stations[count]*100)+"%)" )
            ax[jr,jc].axis('equal')
            plt.colorbar(plot,ax=ax[jr,jc])
            plt.tight_layout()   
            
            
            count = count+1
        

def colorMapOne(ops,nv,nh,axes,fileStrain,fileDisp,comp,station,scale=100, title=""):
    
    dataStrain = np.loadtxt(fileStrain, delimiter=' ', unpack="False")
    dataDisp = np.loadtxt(fileDisp, delimiter=' ', unpack="False")
    
    # total number of values
    ncols = len(dataStrain[0])
    
    # define some stations at which the field is going to be plotted
    # the size of this array must correspond to the number of subplots

    # store the maximum horizontal displacemt
    maxDx = 0
    count = 0
    
            
    index = int(station*ncols)
    if index>0:
        index = index-1
    
    avg = getNodalAverage(ops,nv,nh,fileStrain,comp,index)
    dispRow = dataDisp[:,index]
    
    skip = 2
    nNodes = (nv+1)*(nh+1)
    nodeDisp = np.zeros((nNodes,skip))
    for t in range(nNodes):
        c = skip*t
        nodeDisp[t,:] = dispRow[c:c+skip]
     
    # x coordinates
    x = []
    # y coordinates
    y = []
    # z coordinates
    z = [] 
    # triangles indices
    t = []
    # color value
    v = []
    nodes = ops.getNodeTags()         
    for n in nodes:
        coords = ops.nodeCoord(n)
        disp = ops.nodeDisp(n)
        dx = nodeDisp[n-1,0]
        dy = nodeDisp[n-1,1]
        dz = 0
        coords[0] = coords[0] + dx*scale
        coords[1] = coords[1] + dy*scale
        coords[2] = coords[2] + dz*scale
        
        if dx>maxDx:
            maxDx = dx
        
        x.append(coords[0])
        y.append(coords[1]) 
        z.append(coords[2])
        v.append(avg[n-1])
      
    hLines=nh+1
    for k in range(nh):
        for i in range(nv):
            n1 = hLines*(i)+k
            n2 = hLines*(i)+k+1
            n3 = hLines*(i+1)+k+1
            n4 = hLines*(i+1)+k
            triangle1 = [n1,n2,n3]  
            triangle2 = [n3,n4,n1]
            t.append(triangle1)
            t.append(triangle2)
        

    # fig, ax, = plt.subplots()
    
    t = np.asarray(t)     
    minValue = min(v)
    maxValue = max(v)
    p = (maxValue-minValue) * 0.15
     
    plot = axes.tripcolor(x,y,t,v,edgecolors='black',shading='gouraud',cmap=cm.jet,vmin=minValue-p, vmax=maxValue+p)
    axes.set_title("dx = "+ str( round(station*maxDx*1000,3)) +" (" +str( round(station*100,3))  +"%)" )
    axes.axis('equal')
    # plt.colorbar(plot,ax=axes)
    
    # plt.tight_layout()   
    
    # plot = ax.tripcolor(x,y,t,v,edgecolors='black',shading='gouraud',cmap=cm.jet,vmin=minValue-p, vmax=maxValue+p)
    # ax.set_title("dx = "+ str( round(station*maxDx*1000,3)) +" (" +str(station*100)+"%)" )
    # ax.axis('equal')
    # plt.colorbar(plot,ax=ax)
    # # plt.tight_layout()   
    
    return plot
  


        

       
# =============================================================================
# x=[0,1,0,1,0,1]
# y=[0,0,1,1,2,2]
# z=[0, 0, 0.5, 0.5, 1,1]
# 
# tridat=[[0,1,3],[3,2,0],[2,3,5],[5,4,2]] # 2 triangles in counterclockwise order
# 
# 
# triang=tri.Triangulation(x,y,tridat) #this has changed
# refiner=tri.UniformTriRefiner(triang)
# interp=tri.LinearTriInterpolator(triang,z) #linear interpolator
# new,new_z=refiner.refine_field(z,interp,subdiv=6) #refined mesh
# 
# fig = plt.figure()
# plt.axis('equal')
# plt.tripcolor(new.x,new.y,new_z,cmap=cm.jet)
# =============================================================================
