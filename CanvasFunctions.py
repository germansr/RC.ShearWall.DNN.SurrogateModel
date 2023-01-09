"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
CanvasFunctions.py

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
# functions that are used to draw the cross section, and the elevation of the RC shear wall

from tkinter import LabelFrame, Label

def dimensionHorizontal(canvas, x1, x2, y, length, offsetLabel=0):

    label_frame = LabelFrame(canvas, text="",bd=0)
    label = Label(label_frame, text=str( round(length*100) ) + " cm",bg="white")
    label.pack()
    canvas.create_window(x1+(x2-x1)/2, y-15+offsetLabel, window=label_frame, anchor="c")              
        
    canvas.create_line(x1,y,x2,y, fill=dimLineColor, width=dimLineWidth)
    canvas.create_line(x1,y-dimLinePointerLength,x1,y+dimLinePointerLength, fill=dimLineColor, width=dimLineWidth)
    canvas.create_line(x2,y-dimLinePointerLength,x2,y+dimLinePointerLength, fill=dimLineColor, width=dimLineWidth)
    
    
def dimensionVertical(canvas, y1, y2, x, length):
    
    label_frame = LabelFrame(canvas, text="",bd=0)
    label = Label(label_frame, text=str( round(length*100) ) + " cm",bg="white")
    label.pack()
    canvas.create_window(x+25 ,y1+(y2-y1)/2, window=label_frame, anchor="c") 
    
    canvas.create_line(x,y1,x,y2, fill=dimLineColor, width=dimLineWidth)
    canvas.create_line(x-dimLinePointerLength,y1,x+dimLinePointerLength,y1, fill=dimLineColor, width=dimLineWidth)
    canvas.create_line(x-dimLinePointerLength,y2,x+dimLinePointerLength,y2, fill=dimLineColor, width=dimLineWidth)

def web_reinf(canvas, EBlength,wallLength,xA,xB,yA,yB, webLongReinf, webTransvReinf, thickness):
    
    ebLengthScreen = (xB-xA)*EBlength
    lengthScreen = (xB-xA)-ebLengthScreen*2
    
    concArea = thickness * (wallLength-(EBlength*wallLength*2))
    # concrete cover
    r = concCoverGraph * mToPx
    # THE DIAMETER USED FOR THE REBAR IS SCALED SO THAT IT IS VISIBLE IN THE DRAWING
    # THE TRUE NUMERICAL VALUE IS d6 = 19.05mm
    rad = (30/1000) * mToPx
    # rebar area in meters2
    areaVar = pow((19.05/1000),2)/4 * 3.1416
    # required rebar area in meters2
    areaReq = webLongReinf * concArea
    # quantity of rebar to cover the required area
    qty = areaReq/areaVar
    # quantity rounded to the neares even value
    qtyRounded = myRound(qty)
    # quantity of rebar per face (two faces, so divide by 2)
    quantityPerSide = int(qtyRounded/2)
    
    rmin = 0.0025 
    rmax = 0.0085
    width = linInterpolation(rmin,1,rmax,3,webTransvReinf)
    cr = concCoverGraphHoop*mToPx
    canvas.create_polygon(xA+cr+ebLengthScreen*0.5, yA+cr, 
                          xA+cr+ebLengthScreen*0.5, yB-cr, 
                          xB-cr-ebLengthScreen*0.5, yB-cr, 
                          xB-cr-ebLengthScreen*0.5, yA+cr, 
                          fill='', 
                          outline=WEBtransvReinfColor,
                          width=width)
    
    
    if quantityPerSide-1==0:
        quantityPerSide=2
        
    step = (lengthScreen-(2*r)-rad) / (quantityPerSide-1) 
    
    for i in range(quantityPerSide):
        s = step*i
        x1 = xA+s+r +ebLengthScreen;
        x2 = xA+s+rad+r +ebLengthScreen;
        y1 = yA+r;
        y2 = yA+rad+r;
        canvas.create_oval(x1,y1,x2,y2,fill=rebarWebFill)    
        x1 = xA+s+r +ebLengthScreen;
        x2 = xA+s+rad+r +ebLengthScreen;
        y1 = yB-r;
        y2 = yB-rad-r;
        canvas.create_oval(x1,y1,x2,y2,fill=rebarWebFill)

    

def BE_left(canvas, EBlength,wallLength,xA,xB,yA,yB, longReinf, transvReinf, thickness):
    ebLengthScreen = (xB-xA)*EBlength
    dimensionHorizontal(canvas,xA,xA+ebLengthScreen,yB+15, wallLength*EBlength,offsetLabel=30)
    xB = xA+ ebLengthScreen
    canvas.create_polygon(xA, yA, xA, yB, xB, yB, xB, yA, outline='black', fill='darkgrey')
    
    rmin = 0.0075 
    rmax = 0.015
    width = linInterpolation(rmin,1,rmax,3,transvReinf)
    cr = concCoverGraphHoop*mToPx
    canvas.create_polygon(xA+cr, yA+cr, xA+cr, yB-cr, xB-cr, yB-cr, xB-cr, yA+cr, 
                          fill='', 
                          outline=BEhoopColor,
                          width=width)
    
    concArea = EBlength*wallLength*thickness
    # concrete cover
    r = concCoverGraph * mToPx
    # THE DIAMETER USED FOR THE REBAR IS SCALED SO THAT IT IS VISIBLE IN THE DRAWING
    # THE TRUE NUMERICAL VALUE IS d6 = 19.05mm
    rad = (30/1000) * mToPx
    # rebar area in meters2
    areaVar = pow((19.05/1000),2)/4 * 3.1416
    # required rebar area in meters2
    areaReq = longReinf * concArea
    # quantity of rebar to cover the required area
    qty = areaReq/areaVar
    # quantity rounded to the neares even value
    qtyRounded = myRound(qty)
    # quantity of rebar per face (two faces, so divide by 2)
    quantityPerSide = int(qtyRounded/2)
    if quantityPerSide-1==0:
        quantityPerSide=2
    step = (ebLengthScreen-(2*r)-rad) / (quantityPerSide-1) 
    
    for i in range(quantityPerSide):
        s = step*i
        x1 = xA+s+r;
        x2 = xA+s+rad+r;
        y1 = yA+r;
        y2 = yA+rad+r;
        canvas.create_oval(x1,y1,x2,y2,fill=rebarFill)
        x1 = xA+s+r;
        x2 = xA+s+rad+r;
        y1 = yB-r;
        y2 = yB-rad-r;
        canvas.create_oval(x1,y1,x2,y2,fill=rebarFill)
    
    
    
def BE_right(canvas, EBlength,wallLength,xA,xB,yA,yB, longReinf, transvReinf, thickness):   
    ebLengthScreen = (xB-xA)*EBlength
    canvas.create_line(xB-ebLengthScreen,yA,xB-ebLengthScreen,yB, fill=ebLineColor, width=ebLineWidth)
    dimensionHorizontal(canvas,xB,xB-ebLengthScreen,yB+15, wallLength*EBlength,offsetLabel=30)
    
    x1 = xB - ebLengthScreen
    x2 = xB
    y1 = yA
    y2 = yB
    canvas.create_polygon(x1, y1, x1, y2, x2, y2, x2, y1, outline='black', fill='darkgrey')
    
    rmin = 0.0075 
    rmax = 0.015
    width = linInterpolation(rmin,1,rmax,3,transvReinf)
    cr = concCoverGraphHoop*mToPx
    canvas.create_polygon(x1+cr, y1+cr, x1+cr, y2-cr, x2-cr, y2-cr, x2-cr, y1+cr, 
                          fill='', 
                          outline=BEhoopColor,
                          width=width)
    
    concArea = EBlength*wallLength*thickness
    # concrete cover
    r = concCoverGraph * mToPx
    # THE DIAMETER USED FOR THE REBAR IS SCALED SO THAT IT IS VISIBLE IN THE DRAWING
    # THE TRUE NUMERICAL VALUE IS d6 = 19.05mm
    rad = (30/1000) * mToPx
    # rebar area in meters2
    areaVar = pow((19.05/1000),2)/4 * 3.1416
    # required rebar area in meters2
    areaReq = longReinf * concArea
    # quantity of rebar to cover the required area
    qty = areaReq/areaVar
    # quantity rounded to the neares even value
    qtyRounded = myRound(qty)
    # quantity of rebar per face (two faces, so divide by 2)
    quantityPerSide = int(qtyRounded/2)
    if quantityPerSide-1==0:
        quantityPerSide=2
    step = (ebLengthScreen-(2*r)-rad) / (quantityPerSide-1) 
    for i in range(quantityPerSide):
        s = step*i
        x1 = xB+s+r -ebLengthScreen;
        x2 = xB+s+rad+r -ebLengthScreen;
        y1 = yA+r;
        y2 = yA+rad+r;
        canvas.create_oval(x1,y1,x2,y2,fill=rebarFill)
        x1 = xB+s+r -ebLengthScreen;
        x2 = xB+s+rad+r -ebLengthScreen;
        y1 = yB-r;
        y2 = yB-rad-r;
        canvas.create_oval(x1,y1,x2,y2,fill=rebarFill)
        
        
 
def wallElevation(myCanvas, lengthScaled, lengthReal, heightScaled, heightReal,EBlength):
    
    w = myCanvas.w
    h = myCanvas.h
    canvas = myCanvas.canvas
    
    y1 = wallOffsetY + (h/2) - (heightScaled/2)
    y2 = wallOffsetY + (h/2) + (heightScaled/2)
    
    x1 = wallOffsetX + (w/2) - (lengthScaled/2)
    x2 = wallOffsetX + (w/2) + (lengthScaled/2)

    dimensionHorizontal(canvas,x1,x2,y2+15, lengthReal,offsetLabel=30)
    dimensionVertical(canvas,y1,y2,x2+15, heightReal)
    # wall rectangle
    canvas.create_polygon(x1, y1, x1, y2, x2, y2, x2, y1, outline='black', fill='lightgrey')

    # BE left rectangle
    x2 = x1+ ((x2-x1)*EBlength)
    canvas.create_polygon(x1, y1, x1, y2, x2, y2, x2, y1, outline='black', fill='darkgrey')
    
    # BE right rectangle
    x2 = wallOffsetX + (w/2) + (lengthScaled/2)
    x1 = x2 - ((x2-x1)*EBlength)
    x2 = wallOffsetX + (w/2) + (lengthScaled/2)
    canvas.create_polygon(x1, y1, x1, y2, x2, y2, x2, y1, outline='black', fill='darkgrey')
 
    y1 = wallOffsetY + (h/2) - (heightScaled/2)
    y2 = wallOffsetY + (h/2) + (heightScaled/2)
    x1 = wallOffsetX + (w/2) - (lengthScaled/2)
    x2 = wallOffsetX + (w/2) + (lengthScaled/2)
    # black line in the bottom to point out where is the "ground"
    canvas.create_line(x1-10,y2,x2+10,y2, fill="black", width=5)

    
    
def axialLoad(myCanvas, lengthScaled, lengthReal, heightScaled, heightReal,paxial,fc,t):
    w = myCanvas.w
    h = myCanvas.h
    canvas = myCanvas.canvas
    
    Paxial = 0.85*fc*lengthReal*t*paxial  
    offsetFromWall = 2
    headLength = 12
    y1 = wallOffsetY + (h/2) - (heightScaled/2)
    x1 = wallOffsetX + (w/2) - (lengthScaled/2)
    x2 = wallOffsetX + (w/2) + (lengthScaled/2)
    
    label_frame = LabelFrame(canvas, text="",bd=0)
    label = Label(label_frame, text=str( round(Paxial/1000) ) + " [kN]",bg="white")
    label.pack()
    canvas.create_window(x1+lengthScaled/2, 
                         y1-arrowLoadLength, 
                         window=label_frame, anchor="c")
    
    canvas.create_line(x1+lengthScaled/2,
                       y1-offsetFromWall,
                       x1+lengthScaled/2,
                       y1-arrowLoadLength, 
                       fill=arrowLoadColor, 
                       width=arrowLoadWidth)
    
    canvas.create_line(x1+lengthScaled/2,
                       y1-offsetFromWall,
                       x1+lengthScaled/2-headLength/2,
                       y1-headLength, 
                       fill=arrowLoadColor, 
                       width=arrowLoadWidth)
    
    canvas.create_line(x1+lengthScaled/2,
                       y1-offsetFromWall,
                       x1+lengthScaled/2+headLength/2,
                       y1-headLength, 
                       fill=arrowLoadColor, 
                       width=arrowLoadWidth)


def linInterpolation(x1,y1,x2,y2,x):
    return y1 + (x-x1) * ((y2-y1)/(x2-x1))
    
    
def myRound(n):
    answer = round(n)
    if not answer%2:
        return answer
    if abs(answer+1-n) < abs(answer-1-n):
        return answer + 1
    else:
        return answer - 1
  

# some drawing parameters
  

concCoverGraphHoop = 0.030
concCoverGraph = 0.035
    
BEhoopColor = "royalblue"  
WEBtransvReinfColor = "dark sea green"  

arrowLoadLength = 50  
arrowLoadWidth = 2 
arrowLoadColor = "magenta"
  
# wall elevation
wallOffsetX = -20
wallOffsetY = 0

# web rebar    
rebarWebFill = "green"  
  
# EB rebar   
rebarFill = "blue"

ebLineColor = "black"
ebLineWidth = 1

# drawing scale
mToPx = 180

dimLinePointerLength = 5  
dimLineColor = "black"
dimLineWidth = 1