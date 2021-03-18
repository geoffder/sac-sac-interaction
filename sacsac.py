import neuron
from neuron import h  # hoc interpreter

import math
import numpy as np  # arrays
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt
from pylab import *  # for drawing
import os.path  # make folders for file output
import copy as cp  # copying objects

basest = '/home/mouse/Desktop/NEURONoutput/'  # output folder
#basest = '/home/geoff/Desktop/NEURONoutput/'

# ------------ MODEL RUN PARAMETERS -----------------------------
h.tstop = 2200  #4000#200 # (ms)
h.steps_per_ms = 1  #2 # [10 = 10kHz]
h.dt = 1  #.5 # (ms) [.1 = 10kHz]
h.v_init = -70
h.celsius = 36.9
# ---------------------------------------------------------------

h('numSecs = 3')
h('create sacSomaA, sacSomaB, sacDendA[numSecs], sacDendB[numSecs]')
sacSomaA = h.sacSomaA
sacSomaB = h.sacSomaB
sacDendA = h.sacDendA
sacDendB = h.sacDendB

# pt3dadd(x, y, z, diam)
sacSomaA.push()
h.pt3dclear()
h.pt3dadd(3.711, 19.531, 10, 0.878906)
h.pt3dadd(5.518, 19.897, 7.1442, 10.6233)
h.pt3dadd(4.245, 11.772, 6.541, 8.35494)
h.pt3dadd(4.368, 12.2, 6.4825, 8.31975)
h.pt3dadd(4.594, 12.607, 6.3301, 7.7364)
h.pt3dadd(5.007, 12.963, 5.9626, 6.14065)
h.pt3dadd(4.59, 13.486, 10, 0.878906)
h.pop_section()
sacSomaB.push()
h.pt3dclear()
h.pt3dadd(164.59, 14.486, 10, 0.878906)
h.pt3dadd(165.007, 13.963, 5.9626, 6.14065)
h.pt3dadd(164.594, 13.607, 6.3301, 7.7364)
h.pt3dadd(164.368, 13.2, 6.4825, 8.31975)
h.pt3dadd(164.245, 12.772, 6.541, 8.35494)
h.pt3dadd(165.518, 20.897, 7.1442, 10.6233)
h.pt3dadd(163.711, 20.531, 10, 0.878906)
h.pop_section()
# xy limits: x0=3.711, x1=165.518, y0=11.772, y1=20.897
# origin points: x=84.6145, y=16.334
#xOrigin = 84.6145
#yOrigin = 16.334

# connect up
for i in range(int(h.numSecs)):
    if not i:
        sacDendA[i].connect(sacSomaA, .5, 0)
        sacDendB[i].connect(sacSomaB, .5, 0)
    else:
        sacDendA[i].connect(sacDendA[i - 1], 1, 0)
        sacDendB[i].connect(sacDendB[i - 1], 1, 0)

nzSeed = 1

somaNa = 0
somaK = .005  #.1
somaKm = .001
soma_gleak_hh = .0001667
soma_eleak_hh = -70
soma_nzFactor = 0.25  #0

dendNa = 0
dendK = .003  #.03#.01
dendKm = .0004
dendCaT = .0003  #.0003 # default=.0003
dendCaL = .0003  #.0003 # default=.0003
dend_gleak_hh = .0001667
dend_eleak_hh = -70
dend_nzFactor = 0.25  #0

proxTau1 = 10  #5#20#5 # most proximal bipolar rise
proxTau2 = 60  #50#150#500#400 #300 # decay
midTau1 = .1  #1#5#1 # middle bipolar rise
midTau2 = 12  #20#10#10#50#75#100 #150 # decay #try 15 again?
distTau1 = .1  #1#5#1 # most distal bipolar rise
distTau2 = 12  #20#10#10#50#75#100 #150 # decay
gabaTau1 = .5  # SAC-SAC GABA rise
gabaTau2 = 60  #200#100#150 # decay      #200

# time constants for condition with all bipolars being the same
homoTau1 = 1
homoTau2 = 12  #10#50#150#50#150

proxWeight = .00025 + .00025 * .1  #.0005     #.00025
transWeight = .00045 + .00045 * .1  #.0004#.0005#.001      #.00045
h('gabaWeight = 0')
h('gabaDelay = 0')
h.gabaWeight = .001  #.00075#.0005#.001 #.00075
h.gabaDelay = .5
gabaRev = -70  #-60

lightstart = 0
lightXstart = 0
speed = .1

# direction stuff
dirLabel = [225, 270, 315, 0, 45, 90, 135, 180]
inds = np.array(dirLabel).argsort()  # for sorting responses later
circle = [0, 45, 90, 135, 180, 225, 270, 315, 0]  # for polarplot
circle = np.deg2rad(circle)


def setSomas():
    sacSomaA.nseg = 1
    sacSomaB.nseg = 1
    sacSomaA.Ra = 100
    sacSomaB.Ra = 100

    sacSomaA.insert('HHst')
    sacSomaA.gnabar_HHst = somaNa
    sacSomaA.gkbar_HHst = somaK
    sacSomaA.gkmbar_HHst = somaKm
    sacSomaA.gleak_HHst = soma_gleak_hh  # (S/cm2)
    sacSomaA.eleak_HHst = soma_eleak_hh
    sacSomaA.NF_HHst = soma_nzFactor
    sacSomaA.seed_HHst = nzSeed

    sacSomaB.insert('HHst')
    sacSomaB.gnabar_HHst = somaNa
    sacSomaB.gkbar_HHst = somaK
    sacSomaB.gkmbar_HHst = somaKm
    sacSomaB.gleak_HHst = soma_gleak_hh  # (S/cm2)
    sacSomaB.eleak_HHst = soma_eleak_hh
    sacSomaB.NF_HHst = soma_nzFactor
    sacSomaB.seed_HHst = nzSeed


setSomas()


def setDends():
    # can adjust densities/parameters along length by using sec index
    for i in range(int(h.numSecs)):
        sacDendA[i].nseg = 5
        sacDendB[i].nseg = 5
        sacDendA[i].Ra = 100
        sacDendB[i].Ra = 100
        sacDendA[i].L = 50
        sacDendB[i].L = 50
        sacDendA[i].diam = .5
        sacDendB[i].diam = .5

        sacDendA[i].insert('HHst')
        sacDendA[i].gnabar_HHst = dendNa
        sacDendA[i].gkbar_HHst = dendK
        sacDendA[i].gkmbar_HHst = dendKm
        sacDendA[i].gtbar_HHst = dendCaT
        sacDendA[i].glbar_HHst = dendCaL
        sacDendA[i].gleak_HHst = dend_gleak_hh  # (S/cm2)
        sacDendA[i].eleak_HHst = dend_eleak_hh
        sacDendA[i].NF_HHst = dend_nzFactor
        sacDendA[i].seed_HHst = nzSeed

        sacDendB[i].insert('HHst')
        sacDendB[i].gnabar_HHst = dendNa
        sacDendB[i].gkbar_HHst = dendK
        sacDendB[i].gkmbar_HHst = dendKm
        sacDendB[i].gtbar_HHst = dendCaT
        sacDendB[i].glbar_HHst = dendCaL
        sacDendB[i].gleak_HHst = dend_gleak_hh  # (S/cm2)
        sacDendB[i].eleak_HHst = dend_eleak_hh
        sacDendB[i].NF_HHst = dend_nzFactor
        sacDendB[i].seed_HHst = nzSeed
    if 1:
        for i in range(int(h.numSecs) - 1):
            sacDendA[i].gtbar_HHst = 0
            sacDendA[i].glbar_HHst = 0
            sacDendB[i].gtbar_HHst = 0
            sacDendB[i].glbar_HHst = 0
        sacDendA[int(h.numSecs) - 1].gtbar_HHst = dendCaT
        sacDendA[int(h.numSecs) - 1].glbar_HHst = dendCaL
        sacDendB[int(h.numSecs) - 1].gtbar_HHst = dendCaT
        sacDendB[int(h.numSecs) - 1].glbar_HHst = dendCaL


setDends()

#shorten dends
#sacDendA[1].L = 5
#sacDendB[1].L = 5

# bipolar and gaba synapses (netstim, exp2syn, netcon)
# bipolar netstims are triggered at times dependent on velocity
# gaba netstims are triggered by the voltage near dendritic tip

h('numBips = 4')
h('objref bipsA[numBips], bipsB[numBips]')
h('objref bipStimsA[numBips], bipStimsB[numBips]')
h('objref bipConsA[numBips], bipConsB[numBips]')
h('objref gabaSynA[1], gabaSynB[1], gabaStimA, gabaStimB')
h('objref gabaConA, gabaConB, presynA, presynB')

# bipolar objects
bipsA = h.bipsA
bipsB = h.bipsB
bipStimsA = h.bipStimsA
bipStimsB = h.bipStimsB
bipConsA = h.bipConsA
bipConsB = h.bipConsB
# gaba objects
presynA = h.presynA
presynB = h.presynB
gabaSynA = h.gabaSynA
gabaSynB = h.gabaSynB
gabaStimA = h.gabaStimA
gabaStimB = h.gabaStimB
gabaConA = h.gabaConA
gabaConB = h.gabaConB

####### sac A synapses ##########
sacDendA[0].push()
# x origin of dendrite: 14.368
# proximal
bipsA[0] = h.Exp2Syn(.1)  # 5um from soma (x ~= 19.368)
bipStimsA[0] = h.NetStim(.1)
bipConsA[0] = h.NetCon(bipStimsA[0], bipsA[0], 0, 0, proxWeight)
# medial
bipsA[1] = h.Exp2Syn(.5)  # 25um (x ~= 39.368)
#bipsA[1] = h.Exp2Syn(.6) # 35um (x ~= 49.368)
bipStimsA[1] = h.NetStim(.5)
bipConsA[1] = h.NetCon(bipStimsA[1], bipsA[1], 0, 0, transWeight)
# distal
bipsA[2] = h.Exp2Syn(.9)  # 45um (x ~= 59.368)
bipStimsA[2] = h.NetStim(.9)
bipConsA[2] = h.NetCon(bipStimsA[2], bipsA[2], 0, 0, transWeight)
# distaler
sacDendA[1].push()
bipsA[3] = h.Exp2Syn(.3)  # 65um (x ~= 79.368)
bipStimsA[3] = h.NetStim(.3)
bipConsA[3] = h.NetCon(bipStimsA[3], bipsA[3], 0, 0, transWeight)
h.pop_section()
# default biplor stim settings
for stim in bipStimsA:
    stim.number = 1
    stim.noise = 0
# gaba
gabaSynA[0] = h.Exp2Syn(.3)  # 15um (x ~= 29.368)
gabaStimA = h.NetStim(.3)
gabaStimA.number = 0
gabaStimA.start = 0
gabaStimA.noise = 0
h.pop_section()
# gaba to A is provided by tip of B
sacDendB[2].push()
# have to define connection in hoc because variable (voltage) ref is
# restricted to the currently accessed section (no dend._ref_v etc)
h('presynB = new NetCon(&v(1), gabaSynA[0], -50, gabaDelay, gabaWeight)')  #############
#h('presynB = new NetCon(&v(1), gabaSynA[0], -55, gabaDelay, gabaWeight)')
h.pop_section()
# time constants for sacA
bipsA[0].tau1 = proxTau1
bipsA[0].tau2 = proxTau2
bipsA[1].tau1 = midTau1
bipsA[1].tau2 = midTau2
bipsA[2].tau1 = distTau1
bipsA[2].tau2 = distTau2
bipsA[3].tau1 = distTau1
bipsA[3].tau2 = distTau2
gabaSynA[0].tau1 = gabaTau1
gabaSynA[0].tau2 = gabaTau2
#################################

####### sac B synapses ##########
sacDendB[0].push()
# x origin of dendrite: 174.368
# proximal
bipsB[0] = h.Exp2Syn(.1)  # 5um from soma (x ~= 169.368)
bipStimsB[0] = h.NetStim(.1)
bipConsB[0] = h.NetCon(bipStimsB[0], bipsB[0], 0, 0, proxWeight)
# medial
bipsB[1] = h.Exp2Syn(.5)  # 25um (x ~= 149.368)
bipStimsB[1] = h.NetStim(.5)
bipConsB[1] = h.NetCon(bipStimsB[1], bipsB[1], 0, 0, transWeight)
# distal
bipsB[2] = h.Exp2Syn(.9)  # 45um (x ~= 129.368)
bipStimsB[2] = h.NetStim(.9)
bipConsB[2] = h.NetCon(bipStimsB[2], bipsB[2], 0, 0, transWeight)
# distaler
sacDendB[1].push()
bipsB[3] = h.Exp2Syn(.3)  # 65um (x ~= 109.368)
bipStimsB[3] = h.NetStim(.3)
bipConsB[3] = h.NetCon(bipStimsB[3], bipsB[3], 0, 0, transWeight)
h.pop_section()
# default biplor stim settings
for stim in bipStimsB:
    stim.number = 1
    stim.noise = 0
# gaba syn (on SAC B, from SAC A)
gabaSynB[0] = h.Exp2Syn(.3)  # 15um (x ~= 159.368)
gabaStimB = h.NetStim(.3)
gabaStimB.number = 0
gabaStimB.start = 0
gabaStimB.noise = 0
h.pop_section()
# create gaba pre-synapse on SAC A
sacDendA[2].push()
h('presynA = new NetCon(&v(1), gabaSynB[0], -50, gabaDelay, gabaWeight)')  ##############
#h('presynA = new NetCon(&v(1), gabaSynB[0], -55, gabaDelay, gabaWeight)')
h.pop_section()
# time constants for sacB
bipsB[0].tau1 = proxTau1
bipsB[0].tau2 = proxTau2
bipsB[1].tau1 = midTau1
bipsB[1].tau2 = midTau2
bipsB[2].tau1 = distTau1
bipsB[2].tau2 = distTau2
bipsB[3].tau1 = distTau1
bipsB[3].tau2 = distTau2
gabaSynB[0].tau1 = gabaTau1
gabaSynB[0].tau2 = gabaTau2
#################################

# set synaptic reversals
for i in range(int(h.numBips)):
    bipsA[i].e = 0
    bipsB[i].e = 0
gabaSynA[0].e = gabaRev
gabaSynB[0].e = gabaRev

# make list of all bipolar synapses, from sacA then B
allBips = []
allBipStims = []
for i in range(int(h.numBips)):
    allBips.append(bipsA[i])
    allBipStims.append(bipStimsA[i])
for i in range(int(h.numBips)):
    allBips.append(bipsB[i])
    allBipStims.append(bipStimsB[i])
# until more sophisticated setup, just manually set x locations
# sac A from soma out, then sac B from soma out
#xLocs = [19.37, 39.37, 59.37, 169.37, 149.37, 129.37]
xLocs = [19.37, 39.37, 59.37, 79.37, 189.37, 159.37, 129.37, 109.37]
yLocs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# temp origin based on these
xOrigin = (169.37 - 19.37) / 2.0
yOrigin = 0.0

# shorter dend locations
#xLocs = [19.37, 39.37, 59.37, 124.37, 104.37, 84.37]


def bar(_speed, _xLocs):
    for i in range(len(allBips)):
        allBipStims[i].start = lightstart + (_xLocs[i] - lightXstart) / _speed
    h.run()


def dirBar(angle):
    dirXlocs = range(len(xLocs))
    dirYlocs = range(len(yLocs))
    for k in range(len(xLocs)):
        dirXlocs[k], dirYlocs[k] = rotate(
            (xOrigin, yOrigin), (xLocs[k], yLocs[k]), math.radians(angle)
        )
    for i in range(len(allBips)):
        allBipStims[i].start = lightstart + (dirXlocs[i] - lightXstart) / speed
    h.run()


gabaOn = 1


def gabaSwitch():
    global gabaOn

    if gabaOn:
        h.presynA.weight[0] = 0
        h.presynB.weight[0] = 0
        gabaOn = 0
    else:
        h.presynA.weight[0] = h.gabaWeight
        h.presynB.weight[0] = h.gabaWeight
        gabaOn = 1


homogenous = 0


def homoBips():
    global homogenous

    if not homogenous:
        for i in range(len(bipsA)):
            bipsA[i].tau1 = homoTau1
            bipsA[i].tau2 = homoTau2
            bipsB[i].tau1 = homoTau1
            bipsB[i].tau2 = homoTau2
        # flag that bipolars are now identical
        homogenous = 1
    else:  # already homogenous, switch back to original values
        bipsA[0].tau1 = proxTau1
        bipsA[0].tau2 = proxTau2
        bipsA[1].tau1 = midTau1
        bipsA[1].tau2 = midTau2
        bipsA[2].tau1 = distTau1
        bipsA[2].tau2 = distTau2
        bipsA[3].tau1 = distTau1
        bipsA[3].tau2 = distTau2
        bipsB[0].tau1 = proxTau1
        bipsB[0].tau2 = proxTau2
        bipsB[1].tau1 = midTau1
        bipsB[1].tau2 = midTau2
        bipsB[2].tau1 = distTau1
        bipsB[2].tau2 = distTau2
        bipsB[3].tau1 = distTau1
        bipsB[3].tau2 = distTau2
        # revert flag
        homogenous = 0


# range() does not work with floats, this simple function generates
# ranges of values for loops with whatever step increment I want
def frange(start, stop, step):
    i = start
    while i < stop:
        # makes this work as an iterator, returns value then continues
        # loop for next call of function
        yield i
        i += step


def tuning():
    _xLocs = xLocs
    threshold = 11  #11#13#18
    numTrials = 10

    # velocity run parameters
    startVel = .1  #.05#.025
    lastVel = 4.1
    stepVel = .1  #.05#.025
    numSteps = int((lastVel - startVel) / stepVel) + 1
    # record dendritic tips of SAC A and B
    tipRecs = [h.Vector(), h.Vector()]
    tipRecs[0].record(sacDendA[2](1)._ref_v)
    tipRecs[1].record(sacDendB[2](1)._ref_v)
    tipCa = [h.Vector(), h.Vector()]
    tipCa[0].record(sacDendA[2](1)._ref_ica)
    tipCa[1].record(sacDendB[2](1)._ref_ica)
    # outputs
    aPeaks = range(numSteps)
    bPeaks = range(numSteps)
    aThrAreas = range(numSteps)
    bThrAreas = range(numSteps)
    vAxis = range(numSteps)
    #velRecs = range(numSteps)
    velRecs = [range(numSteps) for i in range(numTrials)]
    aPeaks_iCa = np.zeros(numSteps)  #range(numSteps)
    bPeaks_iCa = np.zeros(numSteps)  #range(numSteps)
    #velRecs_iCa = range(numSteps)
    velRecs_iCa = [range(numSteps) for i in range(numTrials)]
    for k in range(numTrials):
        i = 0
        for speed in frange(startVel, lastVel, stepVel):
            setNoise()
            bar(speed, _xLocs)  # run model
            # store recordings
            #velRecs[i] = cp.deepcopy(tipRecs)
            velRecs[k][i] = cp.deepcopy(tipRecs)
            velRecs_iCa[k][i] = cp.deepcopy(tipCa)
            # make voltage relative to resting potential
            aRec = np.subtract(tipRecs[0], dend_eleak_hh)
            bRec = np.subtract(tipRecs[1], dend_eleak_hh)
            # calculate peak voltage and iCa
            aPeaks[i] = max(aRec)
            bPeaks[i] = max(bRec)
            #aPeaks_iCa[i] = abs(min(tipCa[0]))
            #bPeaks_iCa[i] = abs(min(tipCa[1]))
            aPeaks_iCa[i] += abs(min(tipCa[0]))
            bPeaks_iCa[i] += abs(min(tipCa[1]))
            # threshold voltage
            for j in range(len(tipRecs[0])):
                if aRec[j] < threshold:
                    aRec[j] = 0
                if bRec[j] < threshold:
                    bRec[j] = 0
            # calculate area above threshold
            aThrAreas[i] = sum(aRec)
            bThrAreas[i] = sum(bRec)
            # x axis for plotting
            vAxis[i] = speed * 1000
            i += 1
    # divide peak sums by num trials (average)
    aPeaks_iCa = np.divide(aPeaks_iCa, numTrials)
    bPeaks_iCa = np.divide(bPeaks_iCa, numTrials)

    # calculate preferred - null DSi
    pkPN = np.divide(np.subtract(aPeaks, bPeaks), np.add(aPeaks, bPeaks))
    arPN = np.divide(np.subtract(aThrAreas, bThrAreas), np.add(aThrAreas, bThrAreas))
    pkPN_iCa = np.divide(
        np.subtract(aPeaks_iCa, bPeaks_iCa), np.add(aPeaks_iCa, bPeaks_iCa)
    )
    # save calculated tuning results to file
    fnames = ['vTuningPk', 'vTuningThrArea']
    newFiles = range(len(fnames))
    results = [[aPeaks, bPeaks, pkPN], [aThrAreas, bThrAreas, arPN]]
    for f in range(len(newFiles)):
        newFiles[f] = open(basest + fnames[f] + '.dat', 'w')
    for k in range(numSteps):
        for i in range(len(results)):
            metric = results[i]
            for j in range(len(metric)):
                data = metric[j]
                newFiles[i].write(str(np.round(data[k], decimals=3)) + '\t')
            newFiles[i].write('\n')
    for f in newFiles:
        f.close()
    # save raw data to file
    fname = 'vTuningRecs.dat'
    newFile = open(basest + fname, 'w')
    for i in range(len(velRecs[0][0][0])):
        for trial in velRecs:
            for vel in trial:
                for rec in vel:
                    newFile.write(str(rec[i]) + '\t')
        newFile.write('\n')
    newFile.close()
    # calcium peaks to file
    fname = 'iCaPkTuning.dat'
    newFile = open(basest + fname, 'w')
    metrics = [aPeaks_iCa, bPeaks_iCa, pkPN_iCa]
    for k in range(numSteps):
        for j in range(len(metric)):
            newFile.write(str(np.round(metrics[j][k], decimals=6)) + '\t')
        newFile.write('\n')
    newFile.close()
    # calcium raw to file
    fname = 'iCaTuningRecs.dat'
    newFile = open(basest + fname, 'w')
    for i in range(len(velRecs_iCa[0][0][0])):
        for trial in velRecs_iCa:
            for vel in trial:
                for rec in vel:
                    newFile.write(str(rec[i]) + '\t')
        newFile.write('\n')
    newFile.close()

    # plot results
    fig, axes = plt.subplots(2, 2, sharex='col')
    axes[0, 0].set_title('Peak')
    #axes[0,0].plot(vAxis, aPeaks, 'b-')
    #axes[0,0].plot(vAxis, bPeaks, 'r-')
    #axes[1,0].plot(vAxis, pkPN, 'g-')
    axes[0, 0].plot(vAxis, aPeaks_iCa, 'b-')
    axes[0, 0].plot(vAxis, bPeaks_iCa, 'r-')
    axes[1, 0].plot(vAxis, pkPN_iCa, 'g-')
    axes[1, 0].set_xlabel('velocity (um/s)')
    axes[1, 0].set_ylabel('PN DSi')
    axes[0, 1].set_title('Thresholded Area')
    axes[0, 1].plot(vAxis, aThrAreas, 'b-')
    axes[0, 1].plot(vAxis, bThrAreas, 'r-')
    axes[1, 1].plot(vAxis, arPN, 'g-')
    axes[1, 1].set_xlabel('velocity (um/s)')
    #fig.legend((fugal,petal),('centrifugal','centripetal'),'upper right')
    plt.show()


def setNoise():
    global nzSeed

    sacSomaA.seed_HHst = nzSeed
    nzSeed += 1
    sacSomaB.seed_HHst = nzSeed
    nzSeed += 1

    for dend in sacDendA:
        dend.seed_HHst = nzSeed
        nzSeed += 1
    for dend in sacDendB:
        dend.seed_HHst = nzSeed
        nzSeed += 1


def dirTuning():
    global xLocs, yLocs, lightXstart, nzSeed
    threshold = 18  #18
    speed = 1.2  #.7
    lightXstart = -50
    numTrials = 10

    dirXlocs = range(len(xLocs))
    dirYlocs = range(len(yLocs))

    # record dendritic tips of SAC A and B
    tipRecs = [h.Vector(), h.Vector()]
    tipRecs[0].record(sacDendA[2](1)._ref_v)
    tipRecs[1].record(sacDendB[2](1)._ref_v)
    tipCa = [h.Vector(), h.Vector()]
    tipCa[0].record(sacDendA[2](1)._ref_ica)
    tipCa[1].record(sacDendB[2](1)._ref_ica)
    # outputs
    aPeaks = range(8)
    bPeaks = range(8)
    aThrAreas = range(8)
    bThrAreas = range(8)
    #dirRecs = range(8)
    dirRecs = [range(8) for i in range(numTrials)]
    dirRecs_iCa = [range(8) for i in range(numTrials)]
    aPeaks_iCa = np.zeros(8)
    bPeaks_iCa = np.zeros(8)

    for k in range(numTrials):
        for i in range(len(dirLabel)):
            for j in range(len(xLocs)):
                dirXlocs[j], dirYlocs[j] = rotate(
                    (xOrigin, yOrigin), (xLocs[j], yLocs[j]), math.radians(dirLabel[i])
                )
            # model run
            setNoise()
            bar(speed, dirXlocs)
            # data stuff
            # store recordings
            dirRecs[k][i] = cp.deepcopy(tipRecs)
            dirRecs_iCa[k][i] = cp.deepcopy(tipCa)
            # make voltage relative to resting potential
            aRec = np.subtract(tipRecs[0], dend_eleak_hh)
            bRec = np.subtract(tipRecs[1], dend_eleak_hh)
            # calculate peak voltage
            aPeaks[i] = max(aRec)
            bPeaks[i] = max(bRec)
            # calcium
            aPeaks_iCa[i] += abs(min(tipCa[0]))
            bPeaks_iCa[i] += abs(min(tipCa[1]))
            # threshold voltage
            for j in range(len(tipRecs[0])):
                if aRec[j] < threshold:
                    aRec[j] = 0
                if bRec[j] < threshold:
                    bRec[j] = 0
            # calculate area above threshold
            aThrAreas[i] = sum(aRec)
            bThrAreas[i] = sum(bRec)
    # divide peak sums by num trials (average)
    aPeaks_iCa = np.divide(aPeaks_iCa, numTrials)
    bPeaks_iCa = np.divide(bPeaks_iCa, numTrials)

    # calcium peaks to file
    fname = 'iCaPkDirs.dat'
    newFile = open(basest + fname, 'w')
    metrics = [aPeaks_iCa, bPeaks_iCa]
    for k in range(8):
        for j in range(len(metrics)):
            newFile.write(str(np.round(metrics[j][k], decimals=6)) + '\t')
        newFile.write('\n')
    newFile.close()
    # voltage raw to file
    fname = 'vmDirRecs.dat'
    newFile = open(basest + fname, 'w')
    for i in range(len(dirRecs[0][0][0])):
        for trial in dirRecs:
            for direction in trial:
                for rec in direction:
                    newFile.write(str(rec[i]) + '\t')
        newFile.write('\n')
    newFile.close()
    # calcium raw to file
    fname = 'iCaDirRecs.dat'
    newFile = open(basest + fname, 'w')
    for i in range(len(dirRecs_iCa[0][0][0])):
        for trial in dirRecs_iCa:
            for direction in trial:
                for rec in direction:
                    newFile.write(str(rec[i]) + '\t')
        newFile.write('\n')
    newFile.close()

    # plot results
    # SAC A
    plt.figure(0)
    polar1 = plt.subplot(111, projection='polar')
    #circA = np.array(cp.deepcopy(aThrAreas))
    circA = np.array(cp.deepcopy(aPeaks_iCa))
    circA = circA[inds]
    circA = np.append(circA, circA[0])
    polar1.plot(circle, circA)
    polar1.set_rlabel_position(-22.5)  #labels away from line
    polar1.set_title("SACA threshold area", va='bottom')
    # SAC B
    plt.figure(1)
    polar1 = plt.subplot(111, projection='polar')
    #circB = np.array(cp.deepcopy(bThrAreas))
    circB = np.array(cp.deepcopy(bPeaks_iCa))
    circB = circB[inds]
    circB = np.append(circB, circB[0])
    polar1.plot(circle, circB)
    polar1.set_rlabel_position(-22.5)  #labels away from line
    polar1.set_title("SACB threshold area", va='bottom')
    plt.show()


h.xopen("sacs.ses")


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
