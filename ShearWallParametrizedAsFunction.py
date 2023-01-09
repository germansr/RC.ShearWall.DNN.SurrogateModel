"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Project: 
An Open-Source Framework for Modeling RC Shear Walls using Deep Neural Networks

File:    
ShearWallParametrizesAsFunction.py

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
# A file that contains a function called "run" that will take the geometry and parameters of a RC shear wall as input values and
# will create the corresponding FEM model and run the static pushover analysis   

import openseespy.opensees as ops
import numpy as np
import MyPlottingFEM as plotFEM
import matplotlib.pyplot as plt
import ColorMapFEM as colorMap

def getCurrentNode():
    if len(ops.getNodeTags()) == 0:
        return 0     
    else:
        return ops.getNodeTags()[-1]            
    
def getCurrentElement():
    if len(ops.getEleTags()) == 0:
        return 0     
    else:
        return ops.getEleTags()[-1]  

def getRebarAreaMeters(imperialNumber):
    d = imperialNumber/8
    r = (d/2) * 0.0254 # convert to metric (m)
    a = 3.1416*r**2 # compute area
    return a

def getRebarAreaMetersMetric(metricDiameter):
    metricDiameter = metricDiameter*0.001;
    r = (metricDiameter/2)
    a = 3.1416*r**2 # compute area
    return a

# this function is used to create the MULTI-LAYERED-SHELL section 
def defineWallSecion_ratios_LongAndTransv(ops,secID,mconc,mreinfV,mreinfH,length,height,thick,cover, ratioLong, ratioTransv):
    
    # wall properties
    t = thick
    r = cover
    concMaterial = mconc
    reinfMaterialV = mreinfV
    reinfMaterialH = mreinfH
    
    # vertical reinforcement 
    totalSreqLong = length*t*ratioLong;
    reqPerSideLong = totalSreqLong*0.5 
    layerThickV = reqPerSideLong/length
    layerThickV2 = layerThickV/2
    
    # horizontal reinforcement
    totalSreqTransv = height*t*ratioTransv;
    reqPerSideTransv = totalSreqTransv*0.5 
    layerThickH = reqPerSideTransv/height
    layerThickH2 = layerThickH/2;

    middleConcreteThickWall = t - r*2 - layerThickV*2 - layerThickH*2
    conct = middleConcreteThickWall/4;
    
    # each row, starting from the second, is a layer
    # each row contains the thickness value and the reference to the corresponding material model
    ops.section('LayeredShell',secID,14,
                concMaterial,r,                      # conc cover top
                reinfMaterialV,layerThickV2,         # vertical Reinf layer top 1
                reinfMaterialV,layerThickV2,         # vertical Reinf layer top 2
                reinfMaterialH,layerThickH2,         # horizontal Reinf layer top 1
                reinfMaterialH,layerThickH2,         # horizontal Reinf layer top 2   
                concMaterial,conct,                  # cconcrete core layer 1
                concMaterial,conct,                  # cconcrete core layer 2
                concMaterial,conct,                  # cconcrete core layer 3
                concMaterial,conct,                  # cconcrete core layer 4
                reinfMaterialH,layerThickH2,         # horizontal Reinf layer bot 1
                reinfMaterialH,layerThickH2,         # horizontal Reinf layer bot 2  
                reinfMaterialV,layerThickV2,         # vertical Reinf layer bot 1
                reinfMaterialV,layerThickV2,         # vertical Reinf layer bot 2
                concMaterial,r)                      # conc cover bottom
    
# #####################################################################################################
# GENERATION OF THE PARAMETRIC MODEL
# THE ROUTINE IS SEMI-AUTOMATED FROM THIS POINT FORWARD, SHOULD NOT BE MODIFIED UNLESS NECCESARY
# Units in Newton[N] and Meters[m]
# ####################################################################################################    
    
# the first 11 parameters are the input values specified in the paper
# the remainder parameters are used to discretize the model, specify the number of iterations, and to indicate wheter or not to print some graphics    
def run(t,lw,plbe,pl,pt,webpl,webpt,paxial,wallHeight,compStrength,yieldStrength, 
            meshH=8,
            meshBE=2,
            meshV=10,
            targetDisp=0.02,
            increment=0.0001,
            performPushOver=True,
            plotValidation=False,
            plotDeformedGravity=False,
            plotPushOverResults=False,
            progressBar=None,
            printProgression=True,
            concMatParams=None,
            recordResults=False):


    if plotPushOverResults:
        recordResults=True
        
    if plotDeformedGravity:
        recordResults=True
        
    # Initialize OpenSees model
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)  

    # material parameters      
    fc = compStrength
   
    # axial force in N according to ACI318-19 (not considering the reinforced steel at this point for simplicity)
    Pforce = 0.85*fc*t*lw*paxial
    
    # wall dimensions
    EB_length = lw*plbe;
    wallLength = lw -2*EB_length;
    wallHeight = wallHeight
    wallThick = t;
    
    # DISCRETIZATION
    # number of elements to discretize each boundary element
    discBE = meshBE
    
    # number of elements in the vertical direction
    vSpaces=meshV
    
    # number of elements in the horizontal direction
    hSpaces=meshH #must be even number so that there is a node in the middle
    
    nNodes = (hSpaces+1) * (vSpaces+1)
    
    
    # concrete cover used
    cover = 0.0125;
    
    # shell type
    ShellType = "ShellNLDKGQ"
    
    # shell sections
    sectionID_BE = 1
    sectionID_web = 2
    
    # AUTOMATICALLY COMPUTED VARIABLES
    # number of shells horizontally to discretize the wall segment
    discWall = hSpaces-discBE*2
    
    # shell element horizontal dimensions
    vSpacing = wallHeight/vSpaces;
    hSpacingEB = EB_length/discBE;
    hSpacingWall = wallLength/discWall;
    
    nShellElements = hSpaces*vSpaces;
    hLines = hSpaces+1;
    vLines = vSpaces+1
    
    # array with all the required vertical coordinates
    vSpacing = np.ones(vSpaces+1)*vSpacing;
    vSpacing = np.concatenate((np.zeros(1),vSpacing),axis=0) 
    
    # array with all the required vertical coordinates
    hSpacing = np.zeros(hLines)
    hSpacing[0:discBE] = np.ones(discBE)*hSpacingEB;
    hSpacing[discBE:hSpaces-discBE] = np.ones(hSpaces-2*discBE)*hSpacingWall
    hSpacing[hSpaces-discBE:hSpaces] = np.ones(discBE)*hSpacingEB;
    hSpacing = np.concatenate((np.zeros(1),hSpacing),axis=0) 
    
    # index of the top middle node
    ControlNode= int((vLines-1) * hLines + 1 + hSpaces/2)
    
    elevation = 0
    for i in range(vLines):
        posX=0
        elevation = elevation+vSpacing[i]
        for j in range(hLines):
            nodeIndex = getCurrentNode()+1
            posX = posX+hSpacing[j]
            #print(str(nodeIndex)+":",posX,elevation)
            ops.node(nodeIndex,posX,elevation,0)
            # fix ground nodes
            if i==0:    
                # Fix supports at base of columns
                #   tag, DX, DY, RZ
                ops.fix(nodeIndex, 1, 1, 1, 1, 1, 1)
    
    ops.timeSeries("Linear", 1)					# create TimeSeries for gravity analysis
    ops.pattern('Plain',1,1)
    
    # GRAVITY LOAD AT THE TOP MIDDLE NODE !
    midNode = (hSpaces+1)*(vSpaces+1) - hSpaces/2 
    ops.load(midNode,  0, -Pforce,0.0,0.0,0.0,0.0)	# apply vertical load


    # NON LINEAR CONCRETE MATERIAL MODEL             
    # the fracture strength ft is 10% of fc
    ft = 0.10*fc
    # the crushing strength fcu is 20% of fc
    fcu = -0.20*fc
    # the strain at maxium compressive strength eco is -0.002
    eco = -0.002
    # the strain at the crushing strength ecu is -0.005
    ecu = -0.005
    # the ultimate tensile strain is etu is 0.001
    etu = 0.001
    # shear retention factor is 0.3
    srf = 0.3
    
    # if the mateirla paremters for the concrete are input, then use them
    if concMatParams is not None:
        ops.nDMaterial('PlaneStressUserMaterial',1,40,7, concMatParams[0], 
                                                         concMatParams[1], 
                                                         concMatParams[2], 
                                                         concMatParams[3], 
                                                         concMatParams[4], 
                                                         concMatParams[5], 
                                                         concMatParams[6])
    
    # if not, use the default based on the FC                                        
    else:
        ops.nDMaterial('PlaneStressUserMaterial',1,40,7, fc, ft, fcu, eco, ecu, etu, srf)
            
        
    
    #figure, ax = plt.subplots(2, 3)
    
    
    # out of plane behaviour incorporated to the plane stress material
    ops.nDMaterial('PlateFromPlaneStress',4,1,1.283e10)            
    
    # NON LINEAR STEEL
    # elastic moduli for common reinforcement steel
    steelElasticMod = 202.7e9
    # strain hardening ratio for renforcement steel
    strainHardeningRatio = 0.01
    # yield strength is a user parameter
    fy = yieldStrength
    ops.uniaxialMaterial('Steel02',8,fy,steelElasticMod,strainHardeningRatio,20.0,0.925,0.15)
    
    # Convert rebar material to plane stress/plate rebar 
    #angle=90 longitudinal reinforced steel
    #angle=0 transverse reinforced steel
    ops.nDMaterial('PlateRebar',10,8,90.0) #vertical
    ops.nDMaterial('PlateRebar',11,8,0.0) #horizontal
    
    # Define LayeredShell sections 
    # shell with smeared rebar layer in both directions
    # this section is used for the WEB
    defineWallSecion_ratios_LongAndTransv(ops,            #openseesObj,
                                          sectionID_web,  #sectionID,
                                          4, 10, 11,      #materialConc, #steelReinfLong(vert), #matSteelReinfTransv(horz)
                                          wallLength,     #length (of the wall)
                                          wallHeight,     #height
                                          wallThick,      #thick
                                          cover,          #cover
                                          webpl,          #long reinf ratio (minimum from code)
                                          webpt)          #transv reinf ratio (minimum from code)
    
    #  shell with smeared rebar layer only in the horizontal direction (vertical rebar is defined with truss elements)
    # this section is used for the Boundary Elements
    defineWallSecion_ratios_LongAndTransv(ops,            #openseesObj
                                          sectionID_BE,   #sectionID,
                                          4, 10, 11,      #materialConc, #steelReinfLong(vert), #matSteelReinfTransv(horz)
                                          EB_length,      #length (of the special boundary element)
                                          wallHeight,     #height
                                          wallThick,      #thick
                                          cover,          #cover
                                          pl,             #long reinf ratio  
                                          pt)             #transv reinf ratio 
    
    shellSections = np.zeros(hSpaces)
    shellSections[0:discBE] = np.ones(discBE)*sectionID_BE
    shellSections[discBE:hSpaces-discBE] = np.ones(hSpaces-2*discBE)*sectionID_web
    shellSections[hSpaces-discBE:hSpaces] = np.ones(discBE)*sectionID_BE
    
    
    
    
    # create the elements
    # the numbering starts from the bottom left corner and goes up 
    # this numbering facilitates creationg of the border elements
    for k in range(hSpaces):
        section = shellSections[k]
        for i in range(vSpaces):
            eIndex = getCurrentElement()+1
            n1 = 1 + hLines*(i)+k
            n2 = 1 + hLines*(i)+k+1
            n3 = 1 + hLines*(i+1)+k+1
            n4 = 1 + hLines*(i+1)+k
            ops.element(ShellType,eIndex,n1,n2,n3,n4,int(section))
    
    # BEAM IN THE TOP TO STABILIZE DEFORMATION IN TOP NODES
    a = t*10
    b = t*10
    E = 35e9
    Iz = ((a*b*b*b)/12) 
    Iy = ((a*b*b*b)/12) 
    Jxx=0.141*a*b*b*b
    G=E/(2*(1+0.2))
    A = a*b
    ops.geomTransf('Linear', 1, 0,1,0)
    nodei = midNode = (hSpaces+1)*(vSpaces)+1
    for j in range (hSpaces):            
        ops.element('elasticBeamColumn', getCurrentElement()+1, nodei+j, nodei+j+1, A, E, G,Jxx,Iy,Iz, 1)
    

    # STORE DISPLACEMENTS FOR CONTROL NODE
    if recordResults:
        ops.recorder('Node', 
                     '-file', 'RunTimeNodalResults/Disp.txt', 
                     '-closeOnWrite', 
                     '-node', ControlNode, 
                     '-dof',2, 
                     'disp')
        
        ops.recorder('Element',
                     '-file','RunTimeNodalResults/strain_gravity.txt',
                     '-closeOnWrite',
                     '-eleRange', 1,nShellElements,
                     'strains')
    

    
    # GRAVITY ANALYSIS
    steps = 10
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr',1.0e-4,200)
    ops.algorithm('BFGS','-count',100)
    ops.integrator('LoadControl',1/steps)
    ops.analysis('Static')
    ops.analyze(steps)
    
    if plotDeformedGravity:
        
        # PLOT MODEL MESH
        canvas = plotFEM.canvas()
        canvas.equalScale()
        canvas.drawRCwall(ops,vSpaces,hSpaces,discBE,includeLabels=False,title="Model Mesh")
        
        # PLOT DEFORMED SHAPE DUE TO GRAVITY LOADS
        # canvas2 = plotFEM.canvas()
        # canvas2.equalScale()
        # canvas2.drawRCwallDeformed(ops,vSpaces,hSpaces,discBE,includeLabels=False,scale=1000)

    
    # END OF GRAVITY LOADING ANALYSIS
    # Keep the gravity loads for further analysis
    ops.loadConst('-time',0.0)					
    ops.wipeAnalysis()
    
    dataPush = []
    # PUSHOVER ANALYSIS
    if(performPushOver):
        
        if recordResults:
            ops.recorder('Node', 
                         '-file', 'RunTimeNodalResults/disp_pushover.txt', 
                         '-closeOnWrite', 
                         '-nodeRange', 1,nNodes, 
                         '-dof',1,2,
                         'disp')
            
            ops.recorder('Element',
                     '-file','RunTimeNodalResults/strain_pushover.txt',
                     '-closeOnWrite',
                     '-eleRange', 1,nShellElements,
                     'strains')
            
            ops.recorder('Element',
                     '-file','RunTimeNodalResults/stress_pushover.txt',
                     '-closeOnWrite',
                     '-eleRange', 1,nShellElements,
                     'stresses')
            
        ops.record()    
        
    	# create a plain load pattern for pushover analysis
        ops.pattern("Plain", 2, 1)
        MaxDisp= targetDisp
        DispIncr = increment
        NstepsPush=int(MaxDisp/DispIncr)
        
        if printProgression:
            print("Starting pushover analysis...")
            print("   total steps: ",NstepsPush)
            
        ops.load(ControlNode, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0)	# Apply a unit reference load in DOF=1
        
        ops.system("BandGeneral")
        ops.numberer("RCM")
        referenceDOF = 1
        ops.integrator("DisplacementControl", ControlNode, referenceDOF, DispIncr)
        ops.algorithm('NewtonLineSearch')
        ops.test('NormDispIncr',1e-05, 100, 0)
        ops.analysis("Static")
        	     
        maxUnconvergedSteps = 10
        unconvergeSteps = 0
        finishedSteps = 0 
        dataPush = np.zeros((NstepsPush+1,2))
        
        # Perform pushover analysis
        for j in range(NstepsPush):
            if unconvergeSteps>maxUnconvergedSteps:
                break;
                
            result = ops.analyze(1)
            
            if result<0:
                unconvergeSteps=unconvergeSteps+1
                
            finishedSteps = j   
            disp = ops.nodeDisp(ControlNode,1)*1000		# Convert to mm
            baseShear = -ops.getLoadFactor(2)*0.001
            dataPush[j+1,0] = disp
            dataPush[j+1,1] = baseShear
            
            if progressBar is not None:
                progressBar.step()
            
            if printProgression:
                print("step",j+1,"/", NstepsPush,"   ","disp","=",str(round(disp,2)))
    
            
        if plotPushOverResults:
            plt.rcParams.update({'font.size': 14})
            plt.rc('font', family='TimesNewRomman')
            plt.rcParams["font.family"] = "Times New Roman"
            
            plt.figure(figsize=(4,3), dpi=100)
            plt.plot(dataPush[0:finishedSteps,0], -dataPush[0:finishedSteps,1], color="red", linewidth=1.2, linestyle="-", label='Pushover Analysis')
            plt.axhline(0, color='black', linewidth=0.4)
            plt.axvline(0, color='black', linewidth=0.4)
            plt.grid(linestyle='dotted') 
            plt.xlabel('Displacement (mm)')
            plt.ylabel('Base Shear (kN)')
            
            
            if plotValidation:
                 # Read test output data to plot 
                 Test = np.loadtxt("RunTimeNodalResults/experimental_data.txt", delimiter="\t", unpack="False")
                 plt.plot(Test[0,:], Test[1,:], color="black", linewidth=0.8, linestyle="--", label='Experimental Data')
                 plt.xlim(-1, 25)
                 plt.xticks(np.linspace(-20,20,11,endpoint=True)) 
                 
            plt.tight_layout()     
            plt.legend() 
            plt.show()
            
            
            canvas3 = plotFEM.canvas()
            canvas3.equalScale()
            canvas3.drawRCwallDeformed(ops,vSpaces,hSpaces,discBE,includeLabels=False,scale=20, title="Deformed Shape")
            # colorMap.colorMap(ops, vSpaces, hSpaces, "RunTimeNodalResults/strain_pushover.txt", "RunTimeNodalResults/disp_pushover.txt", 1,scale=20,title="Strain (Y) ")

            
            colorMap.colorMapVarious(ops, vSpaces, hSpaces, "RunTimeNodalResults/strain_pushover.txt", "RunTimeNodalResults/disp_pushover.txt", 1,scale=20, title="Strain (Y) progression")
            
            
        return [dataPush[0:finishedSteps,0], -dataPush[0:finishedSteps,1]], ops    

    return [0,0],[0,0],ops
        
   