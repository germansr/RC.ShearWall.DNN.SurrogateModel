"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
AppGUI.py

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
# This script builds a Graphical User Interface (GUI) to test and visualize the results of the DNN surrogate model  

import ColorMapFEM as colorMap
import tkinter as tk
from tkinter import ttk
import CanvasFunctions as draw
import TrainedNNprediction as testNN
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import ShearWallParametrizedAsFunction as shearWall
from threading import Thread
import sys
import InputVariableBounds as inputBounds
import Normalization as normalization
import random as r

########### FUNCTIONS AND CLASSES ###########################################################################      

# THREAD THAT IS KILLABLE (so that the analysis can be canceled)
class thread_with_trace(Thread):
     
      def __init__(self, *args, **keywords):
        Thread.__init__(self, *args, **keywords)
        self.killed = False
     
      def start(self):
        self.__run_backup = self.run
        self.run = self.__run     
        Thread.start(self)
     
      def __run(self):
        global ops;  
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup
        
     
      def globaltrace(self, frame, event, arg):
        if event == 'call':
          return self.localtrace
        else:
          return None
     
      def localtrace(self, frame, event, arg):
        if self.killed:
          if event == 'line':
            raise SystemExit()
        return self.localtrace
     
      def kill(self):
        self.killed = True

# canvas where the wall section is drawn
class myCanvas(tk.Frame):
    def __init__(self, root, width, height):
        self.w = width
        self.h = height
        self.canvas = tk.Canvas(root, width=self.w, height=self.h,bd=0,bg="white")
        self.canvas.pack( fill=tk.BOTH, expand=True)
        root.bind('<Configure>', self.resize)
    
    # call this everytime the canvas is resized    
    def resize(self, event):
        self.w = event.width
        self.h = event.height
        update_values(event);
        plotCurrentPushoverCurve()
        

# helper class to store the information of each input variable
class myVariable():  
    def __init__(self, name, minvalue, maxvalue, unitString="-", unitFactor = 1):
        self.unitFactor = unitFactor
        self.unitString = str(unitString)
        self.slider = None
        self.name=name
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.string_var = tk.StringVar()
        
    def setSlider(self,slider):    
        self.slider=slider
        
    def getSliderValue(self):
        return self.slider.get()

    def getRealValue(self):
        length = self.maxvalue - self.minvalue
        return self.minvalue + (self.slider.get()/100) * length

    def update(self):  
        self.string_var.set(self.name+" = "+str(round(self.getRealValue()*self.unitFactor,3)) + " ["+self.unitString+"]")


# upadte the drawings and the graphics everytime a variable is changed
def update_values(event):
    
    if not init:
        return
    
    #canvas 1 for the wall cross section
    canvas1 = myCanvas1.canvas
    
    #canvas 2 for the wall elevation
    canvas2 = myCanvas2.canvas
    
    canvas1.delete("all")
    canvas2.delete("all")
    
    w = myCanvas1.w
    h = myCanvas1.h
    mToPx = 140
    moveX = 0
    moveY = -0
    elevationScale = 0.3
    
    xA = w/2 - (var_length.getRealValue()/2)*mToPx + moveX
    xB = w/2 + (var_length.getRealValue()/2)*mToPx + moveX
    
    yA = h/2 - (var_thickness.getRealValue()/2)*mToPx + moveY
    yB = h/2 + (var_thickness.getRealValue()/2)*mToPx + moveY
    
    # DRAW THE WALL RECTANGLE
    canvas1.create_polygon(xA, yA, xA, yB, xB,yB,xB,yA,outline='black', fill='lightgrey')
    canvas1.create_polygon(xA, yA, xA, yB, xB,yB,xB,yA,outline='black', fill='lightgrey')
    
    # DRAW THE WALL LENGTH AND THE THICKNESS DIMENSION LINES
    offset = 12
    draw.dimensionHorizontal(canvas1, xA, xB, yA-offset,var_length.getRealValue())
    draw.dimensionVertical(canvas1, yA, yB, xB+offset, var_thickness.getRealValue());
    
    # draw the web reinforcemenet
    draw.web_reinf(canvas1,var_BElength.getRealValue(), var_length.getRealValue(),
                  xA, xB, yA, yB,
                  var_WEBlongReinf.getRealValue(),
                  var_WEBtransvReinf.getRealValue(),
                  var_thickness.getRealValue())
    
    # DRAW THE BOUNDARY ELEMENTS
    draw.BE_left (canvas1,var_BElength.getRealValue(), var_length.getRealValue(),
                  xA, xB, yA, yB,
                  var_BElongReinf.getRealValue(),
                  var_BEtransvReinf.getRealValue(),
                  var_thickness.getRealValue())
    
    
    draw.BE_right(canvas1,var_BElength.getRealValue(), var_length.getRealValue(),
                  xA, xB, yA, yB,
                  var_BElongReinf.getRealValue(),
                  var_BEtransvReinf.getRealValue(),
                  var_thickness.getRealValue())
    
    # define the scaling
    lengthScaled_elevation = var_length.getRealValue() * mToPx * elevationScale
    heightScaled_elevation = var_Height.getRealValue() * mToPx * elevationScale
    
    # draw the wall elevation
    draw.wallElevation(myCanvas2, lengthScaled_elevation, 
                                  var_length.getRealValue(), 
                                  heightScaled_elevation, 
                                  var_Height.getRealValue(),
                                  var_BElength.getRealValue())
    
    # draw the axial load
    draw.axialLoad(myCanvas2, lengthScaled_elevation, 
                                var_length.getRealValue(), 
                                heightScaled_elevation, 
                                var_Height.getRealValue(),
                                var_AxialLoad.getRealValue(),
                                var_CompStrength.getRealValue(),
                                var_thickness.getRealValue())

    # plot the pushover curve   
    plotCurrentPushoverCurve()
    # update the variables values and text
    updateVariables()

# update the variables values and text 
# this is mainly to update the text on the sliders after the use modifies them
def updateVariables():
    for v in variables:
        v.update();

# clear all the curves and plot the current curve afterwards
def clearGraphic():
    global state
    if state != "Locked":
        for l in axes.get_lines():
                l.remove()            
        plotCurrentPushoverCurve()        
               
# open the results panel
def showResults():
    openResultsPanel();

# plot the pushover curve using the current values
def plotCurrentPushoverCurve(doted=False):
    if not init:
        return
    
    global newLine, currentOutput
    
    # PREDICT THE VALUES USING THE STORED NEURAL NETWORK
    inputValues, predOut = testNN.predict(nnet,
                                        normalizer,
                                        var_thickness.getRealValue(), 
                                        var_length.getRealValue(), 
                                        var_BElength.getRealValue(), 
                                        var_BElongReinf.getRealValue(), 
                                        var_BEtransvReinf.getRealValue(), 
                                        var_WEBlongReinf.getRealValue(), 
                                        var_WEBtransvReinf.getRealValue(), 
                                        var_AxialLoad.getRealValue(), 
                                        var_Height.getRealValue(), 
                                        var_CompStrength.getRealValue(), 
                                        var_YieldStrength.getRealValue()
                                        )
    
    
    xdata = [0,0.5,1.0,2.5,5,10,20]
    ydata = []
    ydata.append(0)
    for val in predOut[0]:
        ydata.append(val)
        
    # change the preivous "curve" color to grey and width to 1    
    if newLine is not None:
         newLine.set_color("lightgrey") 
         newLine.set_linewidth(1)
         newLine.set_label(None)
    
    # plot the new line   
    if doted:
       newLine, = axes.plot(xdata,ydata, linewidth=1, color='black', marker = 'x', markersize = 5, label="NN prediction")
       newLine.set_color("black") 
       newLine.set_linewidth(1)
    else:
       newLine, = axes.plot(xdata,ydata, label="NN prediction")
       newLine.set_color("red") 
       newLine.set_linewidth(2)
        

    currentOutput = ydata

    # Need both of these in order to rescale
    axes.relim()
    axes.autoscale_view()
    axes.legend(loc='lower right')
    # #We need to draw *and* flush
    figure.canvas.draw()
    figure.canvas.flush_events()



# function to run the static pushover analysis      
def runStaticPushoverAnalysis():
   global loadingBar, analysisCount, ops, maxConvergedDispX
   
   targetDisp = 0.02
   increment = targetDisp/steps

   performPushOver = True
   plotValidation = False
   plotDeformedGravity = False
   plotPushOverResults = False

   analysisCount = analysisCount + 1
   
   #---------  CREATE THE MODEL AND RUN THE ANALYSIS WITH THE RANDOM VECTOR ---------
   [x,y],ops = shearWall.run(  var_thickness.getRealValue(),
                           var_length.getRealValue(),
                           var_BElength.getRealValue(),
                           var_BElongReinf.getRealValue(),
                           var_BEtransvReinf.getRealValue(),
                           var_WEBlongReinf.getRealValue(),
                           var_WEBtransvReinf.getRealValue(),
                           var_AxialLoad.getRealValue(),
                           var_Height.getRealValue(),       
                           var_CompStrength.getRealValue(),
                           var_YieldStrength.getRealValue(),
                           8,
                           2,
                           10,
                           targetDisp,
                           increment,
                           performPushOver,
                           plotValidation,
                           plotDeformedGravity,
                           plotPushOverResults,
                           progressBar=loadingBar,
                           recordResults=True)  
   #----------------------------------------------------------------------------------
   
   maxConvergedDispX = max(x)
   
   clearGraphic()
   
   realPushOverCurve, = axes.plot(x,y,label="Pushover Analysis "+str(analysisCount))
   realPushOverCurve.set_color("orange") 
   realPushOverCurve.set_linewidth(3)
   axes.legend(loc='lower right')

   # #We need to draw *and* flush
   figure.canvas.draw()
   figure.canvas.flush_events()
   
   plotCurrentPushoverCurve(True)
   
   if loadingWindow is not None:
       loadingWindow.destroy()
       addTextToConsole("Analysis Finished... ID="+str(analysisCount))
       printInput("Analysis Input:");
       addTextToConsole("");
       lock_variables()
       # btnShowResults.configure(bg='palegreen')      


def addTextToConsole(txt):
    text_box.config(state="normal")
    text_box.insert(tk.END,txt+"\n")
    # tosses txt into textarea on a new line after the end
    # text_box.delete(0,tk.END) # deletes your textbox text
    text_box.yview_moveto( 1 )
    text_box.config(state="disabled")
     
# close the results window    
def closeResultsPanel():
    global resultsWindow
    
    if resultsWindow != None:
        resultsWindow.destroy()
        resultsWindow = None
        

# UPDATE THE FEM RESULTS PLOT    
def updateFEMresults():
    global ops,resultsWindow, axesResults, figureResults,combobox1,slider, colorbar,comboboxa,slider3

    axesResults.clear() 

    station = slider.get()/100
    scale = (slider3.get()/100)*50
    
    if ops != None and axesResults!= None:
        if colorbar != None:
            colorbar.remove()
          
        component = int(comboboxa.get()) 
        
        if combobox1.get()=="Stress":   
            plot = colorMap.colorMapOne(ops, 10, 8, axesResults,"RunTimeNodalResults/stress_pushover.txt", "RunTimeNodalResults/disp_pushover.txt", component,station,scale=scale)
        else:
            plot = colorMap.colorMapOne(ops, 10, 8, axesResults,"RunTimeNodalResults/strain_pushover.txt", "RunTimeNodalResults/disp_pushover.txt", component,station,scale=scale)
            
        colorbar = figureResults.colorbar(plot)
        figureResults.canvas.draw()
        figureResults.canvas.flush_events()
        
    else:
        addTextToConsole("Run the FEM analysis to visualize the results")


# open the results window
def openResultsPanel():

    global resultsWindow, axesResults,figureResults,combobox1,slider,comboboxa,slider3,colorbar

    # only open this window if the model is analyzed
    if ops == None:
        addTextToConsole("Run the FEM analysis to visualize the results")
        return

    # this is a trick so that only one window can be opened at all times 
    # ( avoid creating a new windows everytime the button is clicked)
    if resultsWindow != None:
        return
    

    resultsWindow = tk.Toplevel(root)
    resultsWindow.title("FEM Analysis Results")
    resultsWindow.protocol("WM_DELETE_WINDOW", closeResultsPanel)
    
    ww = 300
    wh = 600
    posx = root.winfo_x() + root.winfo_width()/2 -ww/2;
    posy = root.winfo_y() + root.winfo_height()/2 -wh/2;
    sizeText = str(ww)+"x"+str(wh)+"+"+str(int(posx))+"+"+str(int(posy))
    
    resultsWindow.geometry(sizeText)
  
    
    # COMBOBOX FOR THE "FIELD" 
    box0 = tk.Frame(resultsWindow)
    box0.pack(anchor="w",fill=tk.X)
    
    label0 = tk.Label(box0, text = "Results Options", font='TkDefaultFont 11 bold')
    label0.pack(padx=5,pady=2,anchor="w", side = tk.LEFT )
    
    btnShowColorMap = tk.Button(box0, text='Update Results', width=12,height=1, bd='1', command=updateFEMresults,bg='lightgray')
    btnShowColorMap.pack(padx=[45,5],pady=[5,5],anchor="w", side = tk.RIGHT)
    
    
    # COMBOBOX FOR THE "FIELD" 
    box1 = tk.Frame(resultsWindow)
    box1.pack(anchor="w",fill=tk.X)
    

    label1 = tk.Label(box1, width =6, text = "Field: ",anchor="w")
    label1.pack(padx=5,pady=2,anchor="w", side = tk.LEFT )
    
    combobox1 = ttk.Combobox(box1,state='readonly')
    combobox1['values'] = ('Stress', 'Strain')
    combobox1.pack(fill=tk.X, padx=2, pady=5)
    combobox1.set("Strain")
    # combobox1.bind('<<ComboboxSelected>>', callback)
    
    
    # COMBOBOX FOR THE "FIELD" 
    boxa = tk.Frame(resultsWindow)
    boxa.pack(anchor="w",fill=tk.X)

    labela = tk.Label(boxa, width =6, text = "DoF: ",anchor="w")
    labela.pack(padx=5,pady=2,anchor="w", side = tk.LEFT )
    
    comboboxa = ttk.Combobox(boxa,state='readonly')
    comboboxa['values'] = ("0",'1', '2', '3', "4","5")
    comboboxa.pack(fill=tk.X, padx=2, pady=5)
    comboboxa.set("1")

    # SLIDER FOR THE "STEP"
    box2 = tk.Frame(resultsWindow)
    box2.pack(anchor="w",fill=tk.X)
    
    label2 = tk.Label(box2, width = 6, text = "Step: ",anchor="w")
    label2.pack(padx=5,pady=[5,0], side = tk.LEFT )
    
    slider = tk.Scale(box2,from_=0,to=100,orient='horizontal',showvalue=False)
    slider.set(100)
    slider.pack(fill = tk.X,pady=[5,5])
    
    
    # COMBOBOX FOR THE "FIELD" 
    box3 = tk.Frame(resultsWindow)
    box3.pack(anchor="w",fill=tk.X)

    label3 = tk.Label(box3, width =6, text = "Scale: ",anchor="w")
    label3.pack(padx=5,pady=2,anchor="w", side = tk.LEFT )
    
    slider3 = tk.Scale(box3,from_=0,to=100,orient='horizontal',showvalue=False)
    slider3.set(50)
    slider3.pack(fill = tk.X,pady=[5,5])
    
    
    # CREATE A MATPLOT FIGURE FOR THE PUSHOVER GRAPHIC
    figureResults = Figure(figsize=(2.5, 3), dpi=100)
    axesResults = figureResults.add_subplot()

    # INITIALIZE THE PLOT INSIDE THE GUI
    figure_canvas_2 = FigureCanvasTkAgg(figureResults, resultsWindow)
    NavigationToolbar2Tk(figure_canvas, resultsWindow)
    figure_canvas_2.get_tk_widget().pack(fill=tk.BOTH, expand = True)
    
    # plot the strain field
    plot = colorMap.colorMapOne(ops, 10, 8, axesResults,"RunTimeNodalResults/strain_pushover.txt", "RunTimeNodalResults/disp_pushover.txt", 1,1,scale=20)
    colorbar = figureResults.colorbar(plot)
    
    
# run analysis BUTTON
def runAnalysis():

    global loadingWindow,t1,loadingBar, analysisCount

    loadingWindow = tk.Toplevel(root)
    loadingWindow.title("Analysis Status...")
    
    ww = 300
    wh = 100
    posx = root.winfo_x() + root.winfo_width()/2 -ww/2;
    posy = root.winfo_y() + root.winfo_height()/2 -wh/2;
    sizeText = str(ww)+"x"+str(wh)+"+"+str(int(posx))+"+"+str(int(posy))
    
    loadingWindow.geometry(sizeText)
    tk.Label(loadingWindow,text ="Please wait... \n Running static pushover analysis \n using OpenSeesPy in the background").pack()
    
    # add the even "on closing" to the window. When the window is closed, the function "call_analysis" will be executed
    loadingWindow.protocol("WM_DELETE_WINDOW", cancel_analysis)
    
    loadingBar = ttk.Progressbar(loadingWindow,orient=tk.HORIZONTAL,length=200,mode="determinate",takefocus=True,maximum=steps+5)
    loadingBar.pack()    
    
    tk.Label(loadingWindow,text ="Close this window to cancel").pack()
    
    # start the analysis in a new thread
    t1 = thread_with_trace(target = runStaticPushoverAnalysis)
    t1.start()


def lock_variables():
   global state
   btnLock.configure(bg='red')
   btnLock.configure(text='Locked') 
   
   #btnUnlock.configure(bg='green') 
   state="Locked"
   for v in variables:
       v.slider.config(state="disabled")
   

def unlock_variables():
   global ops, state,resultsWindow
   ops = None
   clearGraphic()
   btnLock.configure(bg='lightgray') 
   btnLock.configure(text='Lock') 
   btnUnlock.configure(bg='lightgray') 
   btnShowResults.configure(bg='lightgray') 
   state = "Normal" 
   
   if resultsWindow != None:
       resultsWindow.destroy()
       resultsWindow = None
       
   for v in variables:
       v.slider.config(state="normal")    
    

def updateLoadingBar():
   global loadingBar
   loadingBar.step()                 

       
def cancel_analysis():
   global t1, ops
   
   if loadingWindow is not None:
       loadingWindow.destroy()
       
   if t1 is not None:
       ops.exit()
       t1.kill()
       t1.join() 
       ops = None
       
   

    
# print the input and the surrogate output to the console    
def printResults():
    if not init:
        return
    
    inputValues = [ var_thickness.getRealValue(),
                    var_length.getRealValue(),
                    var_BElength.getRealValue(),
                    var_BElongReinf.getRealValue(),
                    var_BEtransvReinf.getRealValue(),
                    var_WEBlongReinf.getRealValue(),
                    var_WEBtransvReinf.getRealValue(),
                    var_AxialLoad.getRealValue(),
                    var_Height.getRealValue(),       
                    var_CompStrength.getRealValue(),
                    var_YieldStrength.getRealValue()]
    

    
    roundedList = [ round(elem, 4) for elem in inputValues ]
    inputString = str(roundedList)
    inputString = inputString[1:-1]
    addTextToConsole("Input Values:")
    addTextToConsole(inputString)
    
    roundedList = [ round(elem, 4) for elem in currentOutput ]
    inputString = str(roundedList)
    inputString = inputString[1:-1]
    addTextToConsole("Output Values:")
    addTextToConsole(inputString)

# print the input 
def printInput(txtInput):
    if not init:
        return
    
    inputValues = [ var_thickness.getRealValue(),
                    var_length.getRealValue(),
                    var_BElength.getRealValue(),
                    var_BElongReinf.getRealValue(),
                    var_BEtransvReinf.getRealValue(),
                    var_WEBlongReinf.getRealValue(),
                    var_WEBtransvReinf.getRealValue(),
                    var_AxialLoad.getRealValue(),
                    var_Height.getRealValue(),       
                    var_CompStrength.getRealValue(),
                    var_YieldStrength.getRealValue()]
      
    roundedList = [ round(elem, 4) for elem in inputValues ]
    inputString = str(roundedList)
    inputString = inputString[1:-1]
    addTextToConsole(txtInput)
    addTextToConsole(inputString)

    
# randomize all the sliders
def randomizeVariables():  
    for i in range(len(variables)):
        y = r.randint(0, 100)
        variables[i].slider.set(y)
                  
                  
##################### CODE BEGINS ###########################################################################        
state = "Normal"
analysisCount = 0
init = False
newLine = None
loadingWindow = None
steps = 200
t1 = None
t2 = None
currentOutput = None
programVersion = "Beta 0.1"
ops = None
resultsWindow = None
maxConvergedDispX = 20
axesResults = None
cbValue = "Strain-Y"
station = 0
colorbar = None

# ROOT GUI ELEMENT
root=tk.Tk()
root.title("RC-Shear Walls Analysis with Neural Networks")
root.geometry("900x700")

# MAIN FRAME
mainFrame = tk.PanedWindow(root, orient=tk.HORIZONTAL)
mainFrame.pack(fill = tk.BOTH, expand = True)

# LEFT PANEL
leftframe = tk.PanedWindow(mainFrame,width=200)
mainFrame.add(leftframe)

# RIGHT PANEL
rightframe = tk.PanedWindow(mainFrame, orient=tk.VERTICAL)
mainFrame.add(rightframe)

# TOOLBAR 
toolBarPane = tk.Frame(rightframe)
rightframe.add(toolBarPane)




btnRunAnalysis = tk.Button(toolBarPane, text='Run FEM Analysis', width=14,height=1, bd='1', command=runAnalysis, anchor="w",bg='lightgray')
btnRunAnalysis.pack(anchor="w",padx=2,pady=2, side = tk.LEFT )

btnShowResults = tk.Button(toolBarPane, text='FEM Results', width=10,height=1, bd='1', command=showResults, anchor="w",bg='lightgray')
btnShowResults.pack(anchor="w",padx=2,pady=2, side = tk.LEFT )

btnPrintValues = tk.Button(toolBarPane, text='Print Values', width=10,height=1, bd='1', command=printResults, anchor="w",bg='lightgray')
btnPrintValues.pack(anchor="w",padx=5,pady=2, side = tk.RIGHT )




# button to clear all the graphic lines and re-render the current curve only
btnClearGraph = tk.Button(toolBarPane, text='Clear Graph', width=10,height=1, bd='1', command=clearGraphic, anchor="w",bg='lightgray')
btnClearGraph.pack(anchor="w",padx=(0,0),pady=2, side = tk.RIGHT )


# RIGHT-TOP PANEL (INFO AND PUSHOVER GRAPHIC)
rightTopPane = tk.PanedWindow(rightframe, orient=tk.HORIZONTAL)
rightframe.add(rightTopPane)

#  RIGHT-TOP-LEFT - INFO PANE
rightTopPane_left = tk.PanedWindow(rightTopPane)
rightTopPane.add(rightTopPane_left)
# RIGHT-TOP-RIGHT - PUSHOVER GRAPHIC
rightTopPane_right = tk.PanedWindow(rightTopPane)
rightTopPane.add(rightTopPane_right)


# RIGHT-BOTTOM PANEL (WALL CROSS SECTION AND ELEVATION)
rightBotPane = tk.PanedWindow(rightframe)
rightframe.add(rightBotPane)

# CREATE A MATPLOT FIGURE FOR THE PUSHOVER GRAPHIC
figure = Figure(figsize=(3, 3), dpi=100)


# INITIALIZE THE PLOT INSIDE THE GUI
figure_canvas = FigureCanvasTkAgg(figure, rightTopPane_right)


# bar = NavigationToolbar2Tk(figure_canvas, toolBarPane)
# bar.pack(anchor="w",padx=35,pady=2, )

figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand = True)




# CREATE CANVAS TO DRAW THE WALL CROSS SECTION AND ELEVATION
myCanvas1 = myCanvas(rightBotPane, 600,200)
myCanvas2 = myCanvas(rightTopPane_left, 250,200)

consoleLabel = tk.Label(root, text="Console", font='TkDefaultFont 11 bold')
consoleLabel.pack(anchor="w",pady=5,padx=(5,0))

# THE TEXTBOX IN THE BOTTOM
text_box = tk.Text(root,height=6)
text_box.pack(fill = "x", pady=(0,5),padx=(5,5))

minValues = inputBounds.minValues
maxValues = inputBounds.maxValues

# create the input variables and the sliders 
var_thickness = myVariable("Thickness", minValues[0], maxValues[0], unitString="cm", unitFactor = 100)
var_Height = myVariable("Height", minValues[8], maxValues[8], unitString="cm", unitFactor = 100)
var_length = myVariable("Wall Length", minValues[1], maxValues[1], unitString="cm", unitFactor = 100)
var_BElength = myVariable("BE Length", minValues[2], maxValues[2], unitString="%", unitFactor = 100)
var_CompStrength = myVariable("Comp. Strength f'c", minValues[9], maxValues[9], unitString="MPa", unitFactor = 1/1e6)
var_YieldStrength = myVariable("Yield Strength fy", minValues[10], maxValues[10], unitString="MPa", unitFactor = 1/1e6)
var_BElongReinf = myVariable("BE long Reinf", minValues[3], maxValues[3], unitString="%", unitFactor = 100)
var_BEtransvReinf = myVariable("BE transv Reinf", minValues[4], maxValues[4], unitString="%", unitFactor = 100)
var_WEBlongReinf = myVariable("WEB long Reinf", minValues[5], maxValues[5], unitString="%", unitFactor = 100)
var_WEBtransvReinf = myVariable("WEB transv Reinf", minValues[6], maxValues[6], unitString="%", unitFactor = 100)
var_AxialLoad = myVariable("Axial Load", minValues[7], maxValues[7], unitString="-", unitFactor = 1)

# out all the input variables into a list
variables = []
variables.append(var_CompStrength)
variables.append(var_YieldStrength)
variables.append(var_Height)
variables.append(var_thickness)
variables.append(var_length)
variables.append(var_BElength)
variables.append(var_BElongReinf)
variables.append(var_BEtransvReinf)
variables.append(var_WEBlongReinf)
variables.append(var_WEBtransvReinf)
variables.append(var_AxialLoad)


# Create a box that contains a label (title) and a slider
box = tk.Frame(leftframe)
leftframe.add(box)

titleLabel = tk.Label(box, text="Input Variables", font='TkDefaultFont 11 bold')
titleLabel.pack(anchor="w",padx=5,pady=5)

toolBar2 = tk.Frame(box)
toolBar2.pack()

btnLock = tk.Button(toolBar2, text='Lock', width=7, height=1, bd='1', command=lock_variables, anchor="c", bg="lightgrey")
btnLock.pack(anchor="c",padx=2,pady=2,  side = tk.LEFT )

btnUnlock = tk.Button(toolBar2, text='Unlock', width=7,height=1, bd='1', command=unlock_variables, anchor="c", bg="lightgrey")
btnUnlock.pack(anchor="c",padx=2,pady=2,  side = tk.LEFT )

btnRandom = tk.Button(toolBar2, text='Random', width=7,height=1, bd='1', command=randomizeVariables, anchor="c", bg="lightgrey")
btnRandom.pack(anchor="c",padx=2,pady=2,  side = tk.RIGHT )

# create the sliders that modify the input variables
for i in range(len(variables)):
    
    variables[i].string_var.set(variables[i].name)
    
    sliderLabel = tk.Label(box,anchor="w",textvariable = variables[i].string_var)
    slider = tk.Scale(box,from_=0,to=100,orient='horizontal',command=update_values,showvalue=False)
    slider.set(50)
    
    sliderLabel.pack(fill = tk.BOTH, expand = True)
    slider.pack(fill = tk.BOTH, expand = True)
    
    variables[i].setSlider(slider)

# line below the title for asthethics
separator = ttk.Separator(box, orient='horizontal')
separator.place(x=5, y=30, relwidth=0.95, height=1)


# # LOAD THE PREVIOUSLY SAVED NEURAL NETWORK MODEL
pathToTheNN='NeuralNetworkWeights/dnn_surrogate_model.h5'
nnet=load_model(pathToTheNN)
nnet.summary()

# # CREATE THE MIN-MAX NORMALIZER
normalizer = normalization.getNormalizerForSurrogateModel()


plt.rcParams.update({'font.size': 14})
plt.rc('font', family='TimesNewRomman')
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=10) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the legend

# DEFINE THE AXES OBJECT
axes = figure.add_subplot()

# STYLE FOR THE PLOT
axes.set_title('Pushover curve')
axes.set_xlabel('Displacement [mm]')
axes.set_ylabel('Base Shear [kN]')
axes.set_autoscaley_on(True)
axes.set_xlim(-1, 21)
axes.grid(linestyle='--')

figure.set_tight_layout(True)

# THIS FUNTIONS PLOTS THE CURRENT PUSHOVER CURVE INTO THE PLOT
newLine = None
plotCurrentPushoverCurve()


# # FINISH INITIALIZATION
init = True

addTextToConsole("...Program Initialized...  ["+programVersion+", Dec 2022]")
addTextToConsole("Program Goal: Predicting the NL-response of RC Shear Walls using Artificial Neural Networks")
addTextToConsole("Developed by: Ph.D. Candidate German Solorzano Ramirez, and Dr. Vagelis Plevris")
addTextToConsole("Sponsored by: Oslo Metropolitan University (OsloMet), Department of Civil Engineering and Energy Technology ")
addTextToConsole("")

text_box.config(state="disabled")

root.mainloop()







