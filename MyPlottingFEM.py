"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
MyPlottingFEM.py

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
# This script contains functions for visualization purposes, it is used by the OpenSees script to visualize the model


import math
import matplotlib.pyplot as plt
import random as rnd
import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

# FUNTION TO PLOT A CIRCLE
def plotCircle(ax,centerX,centerY,radius,n,color):
    x = [0]*n
    y = [0]*n
    #divide the 360 degrees with the number of points
    angle = 360/(n-1)
    for i in range(n): 
        theta = math.radians(angle)*i
        x[i] = radius * math.cos(theta)+centerX
        y[i] = radius * math.sin(theta)+centerY
    ax.plot(x, y,color=color)  
#------------------------------------------------------------------------------    


# A class to store a figure and draw shapes in it. 
# it contains functions to create shapes to draw RC shear walls
class canvas():
    
    def __init__(self):
        figure,axes = plt.subplots(ncols=1, nrows=1)
        self.fig = figure
        self.ax = axes
        
    def drawRectangle(self, initX,initY,w,h,colorFill):
        self.ax.add_patch(Rectangle((initX, initY), w, h,color=colorFill))
        
    def drawRectangleWithContour(self, initX,initY,w,h,colorFill,colorLine,lineWidth=1):
        self.ax.add_patch(Rectangle((initX, initY), w, h,color=colorFill))
        botLeft = [initX, initY]
        botRight = [initX+w, initY]
        topLeft = [initX, initY+h]
        topRight = [initX+w, initY+h] 
        self.drawRectanglePolygon(botLeft,botRight,topRight,topLeft,colorFill,colorLine,lineWidth=1);
 
    # draw a polygon using points
    def drawPolygon(self, p1,p2,p3,p4,colorFill="white",colorLine="black",lineWidth=1):
        points = []
        points.append(p1)
        points.append(p2)
        points.append(p3)
        points.append(p4)   
        self.ax.add_patch(Polygon(points, closed=True,color=colorFill))
        self.drawLine(p1,p2, lineWidth,colorLine)
        self.drawLine(p2,p3, lineWidth,colorLine)
        self.drawLine(p3,p4, lineWidth,colorLine)
        self.drawLine(p4,p1, lineWidth,colorLine) 

    # draw a polygon using coords
    def drawPolygon2(self, coords,colorFill="white",colorLine="black",lineWidth=1):
        self.ax.add_patch(Polygon(coords, closed=True,color=colorFill))
        self.drawLine(coords[0],coords[1], lineWidth,colorLine)
        self.drawLine(coords[1],coords[2], lineWidth,colorLine)
        self.drawLine(coords[2],coords[3], lineWidth,colorLine)
        self.drawLine(coords[3],coords[0], lineWidth,colorLine)


    # add text to the plot
    def addText(self,x,y,txt,fontFamily="serif",color="black",size=8):
        font = {'family': fontFamily,
        'color':  color,
        'size': size
        }
        self.ax.text(x,y, txt, fontdict=font)

    
    # draw the RC shear wall, using the information continaed in the OpeenSees Model
    def drawRCwall(self,ops,nv,nh,nbe,includeLabels=False,title=""):
        lastIndex = nh*nv;
        be1 = ops.getEleTags()[0:nv*nbe] 
        wall = ops.getEleTags()[nv*nbe:lastIndex-nv*nbe] 
        be2 = ops.getEleTags()[lastIndex-nv*nbe:lastIndex]
        trusses = ops.getEleTags()[lastIndex:len(ops.getEleTags())]
        
        self.ax.set_title(title)
        self.drawElements(ops,be1,color="grey",displayText=includeLabels)   
        self.drawElements(ops,wall,color="lightgray",displayText=includeLabels)        
        self.drawElements(ops,be2,color="grey",displayText=includeLabels)    
        self.drawTrusses(ops,trusses,color="blue",displayText=includeLabels)  
        self.drawNodes(ops,color="black",displayText=includeLabels)   

    # draw the RC shear wall in deformed shape, using the information continaed in the OpeenSees Model
    def drawRCwallDeformed(self,ops,nv,nh,nbe,includeLabels="True",scale=1,title=""):
        lastIndex = nh*nv;
        be1 = ops.getEleTags()[0:nv*nbe] 
        wall = ops.getEleTags()[nv*nbe:lastIndex-nv*nbe] 
        be2 = ops.getEleTags()[lastIndex-nv*nbe:lastIndex]
        trusses = ops.getEleTags()[lastIndex:len(ops.getEleTags())]
        self.ax.set_title(title)
        self.drawElementsDeformed(ops,be1,color="grey",displayText=includeLabels,scale=scale)   
        self.drawElementsDeformed(ops,wall,color="lightgray",displayText=includeLabels,scale=scale)        
        self.drawElementsDeformed(ops,be2,color="grey",displayText=includeLabels,scale=scale)     
 
    
    def drawElements(self,ops,elements,color="white",displayText=True):       
        for e in elements:
            nodalCoords = [];
            nodes = ops.eleNodes(e)
            for n in nodes:
                coords = ops.nodeCoord(n)[0:2]
                nodalCoords.append(coords)
            self.drawPolygon2(nodalCoords,colorFill=color) 
            if displayText:
                pi = nodalCoords[0]
                pj = nodalCoords[2]
                cx = (pi[0]+pj[0])/2
                cy = (pi[1]+pj[1])/2
                self.addText(cx-0.015, cy-0.015, e,color="black",size=16)

    # draw a deformed element using the polygon routine and the nodal displacement
    def drawElementsDeformed(self,ops,elements,color="white",scale=1,displayText=False):       
        for e in elements:
            nodalCoords = [];
            nodes = ops.eleNodes(e)
            for n in nodes:
                coords = ops.nodeCoord(n)[0:2]
                disp = ops.nodeDisp(n)[0:2]
                coords[0] = coords[0] + disp[0]*scale
                coords[1] = coords[1] + disp[1]*scale
                nodalCoords.append(coords)       
            self.drawPolygon2(nodalCoords,colorFill=color) 


    def drawTrusses(self,ops,elements,lineWidth=8.0,color="orange",displayText=True):       
        for e in elements:
            nodalCoords = [];
            nodes = ops.eleNodes(e)
            for n in nodes:
                coords = ops.nodeCoord(n)[0:2]
                nodalCoords.append(coords)
            self.drawLine(nodalCoords[0],nodalCoords[1], lineWidth,color)
            if displayText:
                pi = nodalCoords[0]
                pj = nodalCoords[1]
                cx = (pi[0]+pj[0])/2
                cy = (pi[1]+pj[1])/2
                self.addText(cx, cy, e,color=color,size=8)

    # draw a node
    def drawNodes(self,ops,color="black",displayText=True):
        nodes = ops.getNodeTags()
        for n in nodes:
            coords = ops.nodeCoord(n)
            x = coords[0]
            y = coords[1] 
            r = 0.0085
            plotCircle(self.ax,coords[0],coords[1],r,32,color=color) 
            if displayText:
                self.addText(x+r, y+r, n,color=color,size=16)
                
    def drawNode(self,x,y,color):                
        self.ax.plot(self.x,self.y,marker='o',color="red") 

    def drawLine(self, p1,p2,width,colorLine):
        self.ax.plot([p1[0],p2[0]], [p1[1],p2[1]],color=colorLine,lw=width)        

   
    def equalScale(self):
        self.ax.axis('equal')

# DEFINE A POINT CLASS:
class Point2D:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.inside = False
    
    def randomize(self, minVal,maxVal):
        self.x = rnd.uniform(minVal,maxVal)
        self.y = rnd.uniform(minVal,maxVal)
        
    def printToConsole(self):
        print("point:",self.x,",",self.y)
        
    def plot(self):    
        if self.inside:
             plt.plot(self.x,self.y,marker='o',color='green')  
        else:
             plt.plot(self.x,self.y,marker='o',color='red')  
#------------------------------------------------------------------------------    

# DEFINE A CIRCLE CLASS:
class Circle:
    def __init__(self, cx,cy,radius):  
        self.cx = cx
        self.cy = cy
        self.r = radius
               
    def getArea(self):   
        return math.pi*self.r**2
    
    #Note: you can use the object methods inside other object methods/functions
    def printArea(self):
        print("Area of the circle is: ",self.getArea())

    def isPointInside(self,p):
        distance = math.sqrt((p.x-self.cx)**2 + (p.y-self.cy)**2)  
        if distance>self.r:
            p.inside = False
            return False
        else:
            p.inside = True
            return True
        
    def plot(self):
        plotCircle(self.cx, self.cy, self.r, 50)
#------------------------------------------------------------------------------     


def newFigure():
    plt.figure()