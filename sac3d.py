import neuron
from neuron import h # hoc interpreter

import numpy as np # arrays
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt
import quickspikes as qs # quick cython spike detection algorithm
from pylab import * # for drawing
import os.path # make folders for file output
import copy as cp # copying objects

h('load_file("/home/mouse/Desktop/NEURONstuff/SAC/sac3dSecondEdit.hoc")')
#h('load_file("/home/geoff/Desktop/NEURONstuff/SAC/sac3dSecondEdit.hoc")')
basest = '/home/mouse/Desktop/NEURONoutput/'
#basest = '/home/geoff/Desktop/NEURONoutput/'

# ------------ MODEL RUN PARAMETERS -----------------------------
h.tstop = 500#250#200 # (ms)
h.steps_per_ms = 10#2 # [10 = 10kHz]
h.dt = .1#.5 # (ms) [.1 = 10kHz]
h.v_init = -70
h.celsius = 36.9
# -----------------------------------------------------------

h('objref SAC3D')
h('SAC3D = new SAC(0,0)')
soma = h.SAC3D.soma
dends = h.SAC3D.dends
h('progress = 0.0')
h('currDir = 0')
############# active properties ######################
activeSOMA = 1
activeDEND = 1 
vcPas = 0 # set passive properties for better clamp
if vcPas:
	activeSOMA = 0
	activeDEND = 0
	activeSYN = 0
	dendPas = 1
	soma_gleak_pas = 0#.0001667 # (S/cm2)
	dend_gleak_pas = 0#.0001667 # (S/cm2)
	prim_gleak_pas = 0#.0001667 # (S/cm2)
# membrane noise
dend_nzFactor = 0 # default NF_HHst = 1 (try with .5)
soma_nzFactor = .1#.25
# soma active properties (SAC model in Ding uses .003 K for soma, down to .002 for dist dend)
somaNa = 0.0 # (S/cm2) 
somaK = .005#.01#.005#.1#.035 # (S/cm2)
somaKm = 0
somaKA = .01#.005
somaCaT = .0003
somaCaL = .0003
soma_gleak_hh = .00033#.0015#.001#.00033#.0001667 # (S/cm2) 
soma_eleak_hh = -70.0 # (mV)
soma_gleak_pas = .0001667 # (S/cm2) 
soma_eleak_pas = -60 # (mV)
# dend compartment active properties
dendNa = 0.0 # (S/cm2) 
dendK = .003#.01#.05 # (S/cm2) 
dendKmax = .004
dendKAmax = .009#.004
dendKm = 0
dendCaT = .0003
dendCaL = .0003
dend_gleak_hh = .00033#0.0001667 # (S/cm2) 
dend_eleak_hh = -70.0 # (mV)
dend_gleak_pas =  .0001667 # (S/cm2)  
dend_eleak_pas = -60 # (mV)
######################################################
# time constants
sustTau1 = 30#30
sustTau2 = 60#100
transTau1 = 5
transTau2 = 15#5#10
# synapse weights
sustWeight = .0002
transWeight = .0004
# synapse placement
sustMaxProb = .6#.3#.4
transMaxProb = .3#.1#.2
# starting seeds
seed = 10
nzSeed = 0

# ------------------VISUAL INPUT ---------------------------								
lightstart=0 # start time of the stimulus bar(ms)
speed= .5#1#1.6 # speed of the stimulus bar (um/ms)
lightwidth=300	# width of the stimulus bar(um)
rotateMode = 1
xMotion = 0 # move bar in x, if not, move bar in y #default 0: y motion
lightXstart=-10#0#-60#0  # start location (X axis)of the stimulus bar (um)
lightXend=300  # end location (X axis)of the stimulus bar (um)
lightYstart= -10  # start location (Y axis) of the stimulus bar (um)
lightYend=300  # end location (Y axis) of the stimulus bar (um)
lightReverse=0	# direction	of light sweep (left>right;right<left)
# angles
dirLabel = [225, 270, 315, 0, 45, 90, 135, 180] # for labelling
inds = np.array(dirLabel).argsort() # for sorting responses later
circle = [0, 45, 90, 135, 180, 225, 270, 315, 0] # for polarplot
circle = np.deg2rad(circle)
dirs =     [135, 90, 45, 0, 45, 90, 135, 180] # for reference 
revDirs =  [45, 90, 135, 180, 135, 90, 45, 0] # for scaling (flipped)
# parameters
dirTrials = 1
jitter = 3
# -----------------------------------------------------------

##################### TERMINAL FINDER ################################
def sacCrawler():
	global orderList, terminals, nonTerms, primGrps
	# sort dendrite sections by branch order and if they are terminal
	soma.push() # start from soma
	orderPos = [0] # last item indicates which # child to check next
	orderList = [[]] # list of lists, dends sorted by branch order 
	terminals = [] # terminal branches (no children)
	nonTerms = [] # non-terminal branches (with children)
	primGrps = [] # dendrites grouped by their primaries
	primCount = 0
	keepGoing = 1
	while keepGoing:
		sref = h.SectionRef() # reference to current section
		if orderPos[-1] < sref.nchild(): # more children to check
			if len(orderPos) > 1: # exclude primes from non-terminals
				nonTerms.append(h.cas()) # add this parent dend to list ###### THIS LIST IS BUGGED (OVER REPORTING)
			sref.child[orderPos[-1]].push() # access child of current
			if len(orderPos) > len(orderList):
				orderList.append([]) 
			orderList[len(orderPos)-1].append(h.cas()) # order new child
			if h.cas() is not h.SAC3D.soma and len(orderPos) == 1:
				primGrps.append([h.cas()]) # start new prime group
				primCount += 1
			if(len(orderPos)) > 1:
				primGrps[primCount-1].append(h.cas()) # add new child to prime group
			orderPos.append(0) # extend to next order
		else: # ran out of children for current parent
			if len(orderPos) == 1: # exceeded number of prime dends
				keepGoing = 0 # entire tree is mapped, exit loop
			else: # current part of tree is exhausted, walk back an order
				if not sref.nchild(): # no children at all
					terminals.append(h.cas()) # add childless dend to list
				del orderPos[-1] # step back an order
				orderPos[-1] += 1 # go to next child of previous order
				h.pop_section() # go back to parent
	print "Number of terminal dendrites: " + str(len(terminals))
#####################################################################
sacCrawler()

def setSoma():
	soma.Ra = 75 #100
	
	if activeSOMA:
		soma.insert('HHst')
		soma.gnabar_HHst = somaNa
		soma.gkbar_HHst = somaK
		soma.gkmbar_HHst = somaKm #.002 HH default
		soma.gleak_HHst = soma_gleak_hh # (S/cm2)
		soma.eleak_HHst = soma_eleak_hh
		soma.NF_HHst = soma_nzFactor
		soma.gtbar_HHst = somaCaT
		soma.glbar_HHst = somaCaL
		#soma.insert('Kv3_3')
		#soma.gbar_Kv3_3 = somaKA
	else:
		soma.insert('pas')
		soma.g_pas = soma_gleak_pas # (S/cm2)
		soma.e_pas = soma_eleak_pas
setSoma()

def basicMembSetup():
	# default all dendrites are same
	for dend in dends:
		dend.Ra = 75 #100
		if activeDEND:
			dend.insert('HHst')
			dend.gnabar_HHst = dendNa
			dend.gkbar_HHst = dendK
			dend.gkmbar_HHst = dendKm #.002 HH default
			dend.gleak_HHst = dend_gleak_hh # (S/cm2)
			dend.eleak_HHst = dend_eleak_hh
			dend.NF_HHst = dend_nzFactor
			dend.gtbar_HHst = dendCaT # default
			dend.glbar_HHst = dendCaL # default
			#dend.insert('Kv3_3')
			#dend.gbar_Kv3_3 = dendKAmax			
basicMembSetup()

def complexMembSetup():
	for group in primGrps:
		for i in xrange(len(group)):
			group[i].push()
			#if primary, set referance point to 0 end
			if not i:
				h.distance(0,0) 
			somaDist = h.distance(0) # dist to start of current section
			
			#group[i].gkbar_HHst = dendKmax - dendKmax*.666*(1.0 - 1/(1.0 + np.exp(-(somaDist-20.0)/5.0)))
			#group[i].gbar_Kv3_3 = dendKAmax - dendKAmax*.666*(1.0 - 1/(1.0 + np.exp(-(somaDist-20.0)/5.0)))
			
			h.pop_section()
complexMembSetup()	

# add option to:
# calculate center XY point of soma
# do all calculations for probability based on XY distance, rather than cable
def placeSyns():
	global sustLocs, transLocs, sustDists, transDists, seed
	global sustX, sustY, transX, transY, sustStim, sustCon
	global transStim, transCon, transBPs, sustBPs
	
	stepSize = 9.0 #(um)
	seg = 1.0/(2*soma.nseg) # portion of section segments break it in to (nseg=5)
	xyMode = 1 # 1: distance by XY; 0: distance by cable
	# XY ref point of soma  
	soma.push()
	pts = int(h.n3d()) # number of 3d points of section
	if(pts % 2): #odd number of points
		somaX = h.x3d((pts-1)/2)
		somaY = h.y3d((pts-1)/2)
	else:
		somaX = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
		somaY = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
	h.pop_section()
	# synapse objects
	h('objref sustBPs[400], transBPs[400]') #don't know how many there will be
	sustBPs = h.sustBPs
	sustStim = []
	sustCon = []
	transBPs = h.transBPs
	transStim = []
	transCon = []
	# coordinates
	sustLocs = []
	sustDists = []
	transLocs = []
	transDists = []
	segXlocs = []
	segYlocs = []
	# counters
	sustCount = 0
	transCount = 0
	dendex = [0,0]
	for group in primGrps:
		for i in xrange(len(group)):
			group[i].push()
			#if primary, set referance point to 0 end
			if not i:
				h.distance(0,0)
			# figure out approx how many 3d points per segment
			# need to know for xy position of placed synapses
			pts = int(h.n3d())
			ptsPerSeg = np.round(pts/(2.0*group[i].nseg))
			if ptsPerSeg < 1.0: ptsPerSeg = 1.0
			somaDist = h.distance(0) # dist to start of section
			# set diam of section
			group[i].diam = .23 + .20 *(exp(-.08*somaDist))
			if stepSize < group[i].L:
				segLen = group[i].L*seg
				segsPerStep = np.round(stepSize/segLen)#np.round(stepSize/segSize,decimals=1)
				if segsPerStep < 1.0: segsPerStep = 1.0
				#print segsPerStep
				for loc in frange(0, 1.0, segsPerStep*seg): #orig: start from segsPerStep*seg
					if loc > 1.0: break
					if (loc/seg)*ptsPerSeg < pts:
						xpt = h.x3d((loc/seg)*ptsPerSeg)
						ypt = h.y3d((loc/seg)*ptsPerSeg)
					else:
						xpt = h.x3d(pts-1)
						ypt = h.y3d(pts-1)
					if not xyMode: totalDist = somaDist+(loc*group[i].L)
					else:
						totalDist = sqrt((somaX-xpt)**2 + (somaY-ypt)**2)
					segXlocs.append(xpt)
					segYlocs.append(ypt)
					#sustProb = sustMaxProb*(1.0 - .98/(1.0 + np.exp(-(totalDist-40.0)/10.0))) # sigmoid test
					sustProb = sustMaxProb*(1.0 - 1/(1.0 + np.exp(-(totalDist-40.0)/15.0))) # sigmoid test
					#transProb = transMaxProb*(-.032552 + 1143.6/((totalDist-47.172)**2+1083.3)) # lor fit of erlang test
					transProb = transMaxProb*(-.028928 + 910.83/((totalDist-42.426)**2+864.68)) # slightly sharper
					# sustained bipolar
					bino = h.Random(seed)
					seed+=1
					roll = bino.binomial(1,sustProb)
					if roll > 0:
						# [segment position, x location, y location]
						sustLocs.append([dendex, loc, xpt, ypt])
						sustDists.append(totalDist)
						# synapse
						sustBPs[sustCount] = h.Exp2Syn(loc) 
						sustBPs[sustCount].tau1 = sustTau1 # rise
						sustBPs[sustCount].tau2 = sustTau2 # decay
						# stimulator
						sustStim.append(h.NetStim(loc))
						sustStim[sustCount].interval = 0
						sustStim[sustCount].number = 1
						sustStim[sustCount].noise = 0
						# connection
						sustCon.append(h.NetCon(
							sustStim[sustCount], sustBPs[sustCount], 0, 0, sustWeight))
						sustCount += 1
					# transient bipolar
					bino = h.Random(seed)
					seed+=1
					roll = bino.binomial(1,transProb)
					if roll > 0:
						# [segment position, x location, y location]
						transLocs.append([dendex, loc, xpt, ypt])
						transDists.append(totalDist)
						# synapse
						transBPs[transCount] = h.Exp2Syn(loc) 
						transBPs[transCount].tau1 = transTau1 # rise
						transBPs[transCount].tau2 = transTau2 # decay
						# stimulator
						transStim.append(h.NetStim(loc))
						transStim[transCount].interval = 0
						transStim[transCount].number = 1
						transStim[transCount].noise = 0
						# connection
						transCon.append(h.NetCon(
							transStim[transCount], transBPs[transCount], 0, 0, transWeight))
						transCount += 1
			dendex[1] += 1
			h.pop_section()
		dendex = [dendex[0]+1,0] # next primary, reset count
	print "sustained BP syns: " + str(len(sustLocs))
	print "transient BP syns: " + str(len(transLocs))
	
	sustX = [sustLocs[i][2] for i in xrange(len(sustLocs))]
	sustY = [sustLocs[i][3] for i in xrange(len(sustLocs))]
	transX = [transLocs[i][2] for i in xrange(len(transLocs))]
	transY = [transLocs[i][3] for i in xrange(len(transLocs))]
	if 0:
		# cable dist rasters
		plt.figure(0)
		plt.subplot(2,1,1)
		plt.xlim(0,300)
		plt.eventplot(sustDists,'horizontal')
		plt.subplot(2,1,2)
		plt.xlim(0,300)
		plt.eventplot(transDists,'horizontal')
		
		# xy location scatter
		plt.figure(1)
		plt.scatter(segXlocs,segYlocs,color='.1',marker='o')
		plt.scatter(sustX,sustY,color='m',marker='x',s=60)		
		plt.scatter(transX,transY,color='y',marker='+',s=60)
		plt.xlim(0,300)
		plt.ylim(0,300)
		plt.show()
	if 0:
		somaXs=[];somaYs=[]
		soma.push()
		pts = int(h.n3d()) # number of 3d points of section
		for i in xrange(pts):
			somaXs.append(h.x3d(i))
			somaYs.append(h.y3d(i))
		h.pop_section()
		plt.figure(2)
		plt.scatter(segXlocs,segYlocs)
		#plt.scatter(somaXs,somaYs)
		plt.scatter(somaX,somaY)
		plt.xlim(0,300)
		plt.ylim(0,300)
		plt.show()
placeSyns()

def locTest():
	stepSize = 1.0 #(um)
	seg = 1.0/(2*soma.nseg) # portion of section segments break it in to (nseg=5)
	segXlocs = []
	segYlocs = []
	for dend in dends:
		dend.push()
		pts = int(h.n3d())
		ptsPerSeg = np.round(pts/(2.0*dend.nseg))
		if ptsPerSeg < 1.0: ptsPerSeg = 1.0
		if stepSize < dend.L:
			segLen = dend.L*seg
			segsPerStep = np.round(stepSize/segLen)
			if segsPerStep < 1.0: segsPerStep = 1.0
			for loc in frange(segsPerStep*seg, 1.0, segsPerStep*seg):
				if (loc/seg)*ptsPerSeg < pts:
					xpt = h.x3d((loc/seg)*ptsPerSeg)
					ypt = h.y3d((loc/seg)*ptsPerSeg)
				else:
					xpt = h.x3d(pts-1)
					ypt = h.y3d(pts-1)
				segXlocs.append(xpt)
				segYlocs.append(ypt)
		h.pop_section()
	plt.figure(2)
	plt.scatter(segXlocs,segYlocs)
	plt.xlim(0,300)
	plt.ylim(0,300)
	plt.show()
#locTest()

def barOnsets(seedL):
	global seed
	
	if dirRunning:
		_sustX = dirSustX
		_sustY = dirSustY
		_transX = dirTransX
		_transY = dirTransY
	else:
		_sustX = sustX
		_sustY = sustY
		_transX = transX
		_transY = transY
		
	for syn in xrange(len(_sustX)):
		# distance to synapse divided by speed
		if xMotion:
			if lightReverse:
				synT =(lightstart+(lightXend-_sustX[syn])/speed)
			else:
				synT =(lightstart+(_sustX[syn]-lightXstart)/speed)
		else: # motion in y
			if lightReverse:
				synT =(lightstart+(lightYend-_sustY[syn])/speed)
			else:
				synT =(lightstart+(_sustY[syn]-lightYstart)/speed)
			
		# variable onset time for current synapse
		mOn = h.Random(seedL)
		mOn.normal(synT, jitter)
		seedL += 1
		sustStim[syn].start = mOn.repick()
	
	for syn in xrange(len(_transX)):
		# distance to synapse divided by speed
		if xMotion:
			if lightReverse:
				synT =(lightstart+(lightXend-_transX[syn])/speed)
			else:
				synT =(lightstart+(_transX[syn]-lightXstart)/speed)
		else: # motion in y
			if lightReverse:
				synT =(lightstart+(lightYend-_transY[syn])/speed)
			else:
				synT =(lightstart+(_transY[syn]-lightYstart)/speed)
			
		# variable onset time for current synapse
		mOn = h.Random(seedL)
		mOn.normal(synT, jitter)
		seedL += 1
		transStim[syn].start = mOn.repick()
		
	seed = seedL

def ringOnsets(cf, seedL):
	global seed
	
	soma.push()
	pts = int(h.n3d()) # number of 3d points of section
	if(pts % 2): #odd number of points
		somaX = h.x3d((pts-1)/2)
		somaY = h.y3d((pts-1)/2)
	else:
		somaX = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
		somaY = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
	h.pop_section()
	
	furthest = furthestDist()
	
	for syn in xrange(len(sustX)):
		# distance to synapse divided by speed
		dist = sqrt((somaX-sustX[syn])**2 + (somaY-sustY[syn])**2)
		cfTime = lightstart + dist/speed
		if cf: # centrifugal
			synT = cfTime
		else: # centripetal
			synT = furthest/speed - cfTime + 10
			
		# variable onset time for current synapse
		mOn = h.Random(seedL)
		mOn.normal(synT, jitter)
		seedL += 1
		sustStim[syn].start = mOn.repick()
	
	for syn in xrange(len(transX)):
		# distance to synapse divided by speed
		dist = sqrt((somaX-transX[syn])**2 + (somaY-transY[syn])**2)
		cfTime = lightstart + dist/speed
		if cf: # centrifugal
			synT = cfTime
		else: # centripetal
			synT = furthest/speed - cfTime + 10
			
		# variable onset time for current synapse
		mOn = h.Random(seedL)
		mOn.normal(synT, jitter)
		seedL += 1
		transStim[syn].start = mOn.repick()
		
	seed = seedL

def ringFlash():
	global seed, nzSeed, transCon, sustCon
	
	seedL = seed
	
	width = 30
	h.tstop = 250
	synT = 20
	
	soma.push()
	pts = int(h.n3d()) # number of 3d points of section
	if(pts % 2): #odd number of points
		somaX = h.x3d((pts-1)/2)
		somaY = h.y3d((pts-1)/2)
	else:
		somaX = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
		somaY = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
	
	# setup voltage clamp 
	h('objref VC')
	h('VC = new SEClamp(.5)')
	#VC = h.SEClamp(.5)
	VC = h.VC
	VC.dur1 = 100 # (ms)
	VC.dur2 = 100
	VC.dur3 = 100
	h.tstop = VC.dur1 + VC.dur2 + VC.dur3
	VC.amp1 = -70 # (mV)
	VC.amp2 = -70
	VC.amp3 = -70
	
	# record current
	vecIlist = []
	vecI = h.Vector()
	# note: no & and backwards + _ref_ notation in python
	vecI.record(VC._ref_i)
	
	#h.pop_section()
	
	for ring in frange(0, 150, width):
		# sustained
		for syn in xrange(len(sustX)):
			# distance to synapse divided by speed
			dist = sqrt((somaX-sustX[syn])**2 + (somaY-sustY[syn])**2)
			
			if dist > ring and dist < ring+width:
				sustCon[syn].weight[0] = sustWeight
				# variable onset time for current synapse
				mOn = h.Random(seedL)
				mOn.normal(synT, jitter)
				seedL += 1
				sustStim[syn].start = mOn.repick()
			else:
				sustCon[syn].weight[0] = 0
		# transient
		for syn in xrange(len(transX)):
			# distance to synapse divided by speed
			dist = sqrt((somaX-transX[syn])**2 + (somaY-transY[syn])**2)
			
			if dist > ring and dist < ring+width:
				transCon[syn].weight[0] = transWeight
				# variable onset time for current synapse
				mOn = h.Random(seedL)
				mOn.normal(synT, jitter)
				seedL += 1
				transStim[syn].start = mOn.repick()
			else:
				transCon[syn].weight[0] = 0
		###### model run ######
		h.init()
		
		if 0:
			# set noise seeds
			soma.seed_HHst = nzSeed
			nzSeed += 1
			for dend in dends:
				dend.seed_HHst = nzSeed
				nzSeed += 1
			# soma recording		
			vecVm = h.Vector()
			vecVm.record(soma(.5)._ref_v)
		
		h.run()
		
		if 1:
			# store vectors in list between runs
			vecItemp = np.round(np.array(vecI), decimals = 3)
			vecIlist.append(cp.copy(vecItemp))
		#######################
	for i in xrange(len(vecIlist)):
		plot(vecIlist[i])
		
	xlabel('time')
	ylabel('current')
	show()
	
	seed = seedL
	
dirRunning = 0
def dirRun():
	global numReps, dirRunning, dirSustX, dirSustY, dirTransX, dirTransY
	global nzSeed
	
	dirRunning = 1
	h.progress = 0.0
	
	if rotateMode: origin = findOrigin()
	
	for j in xrange(dirTrials):
		for i in xrange(len(dirs)):
			h.currDir = dirLabel[i]
			
			if rotateMode:
				# rotate sustained BP locations
				dirSustX = range(len(sustX))
				dirSustY = range(len(sustY))
				for k in xrange(len(sustX)):
					dirSustX[k], dirSustY[k] = rotate(origin, (sustX[k], sustY[k]), math.radians(dirLabel[i]))
				# rotate transient BP locations
				dirTransX = range(len(transX))
				dirTransY = range(len(transY))
				for k in xrange(len(transX)):
					dirTransX[k], dirTransY[k] = rotate(origin, (transX[k], transY[k]), math.radians(dirLabel[i]))
			else:
				dirSustX = sustX
				dirSustY = sustY
				dirTransX = transX
				dirTransY = transY
			
			###### model run ######
			h.init()
			barOnsets(seed)
			
			# set noise seeds
			soma.seed_HHst = nzSeed
			nzSeed += 1
			for dend in dends:
				dend.seed_HHst = nzSeed
				nzSeed += 1
			# soma recording		
			vecVm = h.Vector()
			vecVm.record(soma(.5)._ref_v)
		
			h.run()
			#######################
			
		h.progress = h.progress + 100.0/dirTrials
	dirRunning = 0
	#fname = 'lastSeeds.dat' # last used seeds
	#newFile = open(basest + runLabel + fname,'w')
	#newFile.write('seed:\n'+str(seed-1)+'\nnzSeed:\n'+str(nzSeed-1))
	#newFile.close()

ringRunning = 0
def ringRun():
	global seed, nzSeed, ringRunning
	ringRunning = 1
	for j in xrange(dirTrials):
		for i in xrange(2):
			
			###### model run ######
			h.init()
			ringOnsets(i, seed)
			
			# set noise seeds
			soma.seed_HHst = nzSeed
			nzSeed += 1
			for dend in dends:
				dend.seed_HHst = nzSeed
				nzSeed += 1
			# soma recording		
			vecVm = h.Vector()
			vecVm.record(soma(.5)._ref_v)
		
			h.run()
			#######################
	ringRunning = 0

def vsteps():
	global transCon, sustCon
	
	soma.push()
	
	for syn in transCon: 
		syn.weight[0] = 0
	for syn in sustCon: 
		syn.weight[0] = 0
		
	# setup voltage clamp 
	h('objref VC')
	h('VC = new SEClamp(.5)')
	#VC = h.SEClamp(.5)
	VC = h.VC
	VC.dur1 = 25 # (ms)
	VC.dur2 = 100
	VC.dur3 = 25
	h.tstop = VC.dur1 + VC.dur2 + VC.dur3
	VC.amp1 = -70 # (mV)
	VC.amp3 = -70
	
	# record current
	vecIlist = []
	vecI = h.Vector()
	# note: no & and backwards + _ref_ notation in python
	vecI.record(VC._ref_i)
		
	for hold in frange(-70, 40, 10):
		VC.amp2 = hold # set step potential
		
		# initiate and run
		h.init()
		h.run()
		
		# store vectors in list between runs
		vecItemp = np.round(np.array(vecI), decimals = 3)
		vecIlist.append(cp.copy(vecItemp))
	
	for i in xrange(len(vecIlist)):
		plot(vecIlist[i])
		
	xlabel('time')
	ylabel('current')
	show()
			
# range() does not work with floats, this simple function generates
# ranges of values for loops with whatever step increment I want
def frange(start, stop, step):
	i = start
	while i < stop:
		# makes this work as an iterator, returns value then continues
		# loop for next call of function
		yield i 
		i += step

#single cell
def findOrigin():
	leftX = 1000
	rightX = -1000
	topY = -1000
	botY = 1000
	for dend in dends:
		dend.push()
		
		# mid point
		pts = int(h.n3d()) # number of 3d points of section
		if(pts % 2): #odd number of points
			xLoc = h.x3d((pts-1)/2)
			yLoc = h.y3d((pts-1)/2)
		else:
			xLoc = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
			yLoc = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
		if (xLoc < leftX): leftX = xLoc
		if (xLoc > rightX): rightX = xLoc
		if (yLoc < botY): botY = yLoc
		if (yLoc > topY): topY = yLoc	
		
		# terminal point
		xLoc = h.x3d(pts-1)
		yLoc = h.y3d(pts-1)
		if (xLoc < leftX): leftX = xLoc
		if (xLoc > rightX): rightX = xLoc
		if (yLoc < botY): botY = yLoc
		if (yLoc > topY): topY = yLoc
		
		h.pop_section()
	#print  'leftX: '+str(leftX)+ ', rightX: '+str(rightX)
	#print  'topY: '+str(topY)+ ', botY: '+str(botY)
	return	(leftX+(rightX-leftX)/2, botY+(topY-botY)/2)

def furthestDist():
	
	soma.push()
	pts = int(h.n3d()) # number of 3d points of section
	if(pts % 2): #odd number of points
		somaX = h.x3d((pts-1)/2)
		somaY = h.y3d((pts-1)/2)
	else:
		somaX = (h.x3d(pts/2)+h.x3d((pts/2)-1))/2.0
		somaY = (h.y3d(pts/2)+h.y3d((pts/2)-1))/2.0
	h.pop_section()

	furthest = 0
	for dend in dends:
		dend.push()	
		pts = int(h.n3d())
		# terminal point
		xLoc = h.x3d(pts-1)
		yLoc = h.y3d(pts-1)
		dist = sqrt((somaX-xLoc)**2 + (somaY-yLoc)**2)
		if dist > furthest: furthest = dist
		h.pop_section()
	
	return	furthest
	
def rotate(origin, point, angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""
	ox, oy = origin
	px, py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
	return qx, qy

soma.push()
h.xopen("starburst3D.ses") # open neuron gui session
