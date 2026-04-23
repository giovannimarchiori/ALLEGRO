# calculate and save histograms of capacitance per source vs theta
# output is saved in ROOT file with given filename
#
# execute script with
# python create_capacitance_file_theta_CERN_PCBv2_fix_capadensities.py
#
# Updated from the previous version by Juska in November 2025
#
# This is the second updated version where capacitance length densities are taken from measurements,
# shields-per-pad count is fixed and (soon) dielectric between signal pad and absorber
# is taken into account in the detector capacitance calculation


from ROOT import TH1F, TF1, TF2, TCanvas, TLegend, TFile, gStyle, gROOT
from math import ceil, sin, cos, log, tan, pi, sqrt, asin, degrees
import os

gROOT.SetBatch(True)

gStyle.SetPadTickY(1)

debug = False
verbose = True

# Add appendix to filenames for not mixing different versions
apdx = "_update2025"

# output file
if not os.path.isdir("root"):
    os.mkdir("root")
filename = "root/capacitances_perSource_ecalBarrelFCCee_theta%s.root" % apdx
# output folder for plot
if not os.path.isdir("plots"):
    os.mkdir("plots")

# layer 2 require special care as it is separated in several cells and that the shield run beneath the etch: cell 2 signal pad top capa: 0.68 + 0.20 = 0.88, cell 2 signal pad bot: 0.56 + 0.21 = 0.77, cell 3: 0.34 + 2.4 = 2.74, cell 4: 1 + 0.25 = 1.25, cell 5: 1.85 + 0.28 = 2.13

# Detector
ecal_active_rmin_mm = 2172.8  # ECAL calorimeter volume rmin (cm) * 10 -> mm
Nplanes = 1536  # Number of readout planes
electrode_inclination_deg = 50.18  # rotation angle (degrees)
angle = electrode_inclination_deg / 180. * pi  # inclination angle in radians
absorber_thickness_mm = 2.0  # total thickness of absorber (cm) * 10 -> mm
ecal_active_thickness_mm = 405.5  # ECAL thickness of calorimeter (cm) * 10 -> mm [IN PRINCIPLE NOT NEEDED]
pcb_thickness_mm = 1.2  # thickness of readout planes (cm) * 10 -> mm

# Segmentation (ECalBarrel_thetamodulemerged.xml)
total_electrode_length_mm = 573.937  # Total electrode length from calorimeter xml description (cm) * 10 -> mm (equal to sum of readoutLayerParallelLengths) [IN PRINCIPLE NOT NEEDED]
# careful, this is not really the radial spacing, it is, after dilution, the spacing in the parallel direction --> radial depth spacing will not be constant
readoutLayerRadialLengths = [1.69, 3.53, 3.69, 3.76, 3.84, 3.99, 4.15, 4.30, 4.45, 4.61, 4.45] # Updated to give real PCBv2 cell lengths
# JP PCBv2 cell lenghts by Juska: ls = [22, 46, 48, 48, 48, 48, 49, 50, 52, 54, 56, 58, 60, 58] (in mm)
# Re-calculated with radlen = l*sin(inclination_degree)
# It gives [1.69, 3.53, 3.69, 3.76, 3.84, 3.99, 4.15, 4.30, 4.45, 4.61, 4.45]

# GM NOT SURE ABOUT THESE CELL LENGTHS... IN XML WE HAVE
readoutLayerParallelLengths = [23.3596, 47.5685, 48.9843, 50.4000, 52.0989, 53.6562, 55.4966, 57.3371, 59.4607, 61.5843, 63.9910]  # length of each layer along the electrode direction

numLayers = len(readoutLayerParallelLengths)  # Number of longitudinal layers
grid_size_theta = 0.00245436925  # grid_size_theta
offset_theta = 0.5902785  # offset_theta                              
minTheta = offset_theta - grid_size_theta / 2.  # min theta of calorimeter: offset_theta - grid_size_theta/2
maxTheta = pi - minTheta  # max theta
numTheta = int(ceil((maxTheta - minTheta) / grid_size_theta))  # number of cells in theta
# layer containing the strips
stripLayer = 1  # JP So we start counting from zero here. Let's keep that in mind.
                # I keep the strips in the "1st" layer as it is still the baseline choice,
                # and the impact to capacitance map is small

theta_merging = 4  # in mergedCells_Theta, all other positions should be filled with 4
module_merging = 2  # in mergedModules, all elements should be equal to 2
nMergedThetaCells = [theta_merging]*numLayers
nMergedThetaCells[stripLayer] = 1
nMergedModules = [module_merging]*numLayers

# PCB: traces
# only one trace for strip layer because 4 cells instead of one. Version where we extract all channels from the back
# => [0, 1, 5, 6, ..]
tracesPerLayer = [i for i in range(numLayers)]
for i in range(stripLayer+1, numLayers):
    tracesPerLayer[i] += 3
    
# JP restore strip layer trace count to zero - we will route the trace(s) between strips!
# tracesPerLayer[stripLayer] = 0 #FIXME this screwed up one trick so had to disable. Strip capa becomes correct due to "capa density" set to zero.

# PCB dimensions [mm]  # JP Updated to CERN PCBv2. Old were mainly from Paris PCBv3.
hhv = 0.1
hs = 0.15
t = 0.035
tsh = 0.0175  # JP shields are thinner in PCBv2, needed to add this
w = 0.09
ws = 0.180
hm = 0.3
pcbThickness = 5 * t + 2 * tsh + 2 * hhv + 2 * hs + 2 * hm  # mm #JP updated also this equation
                                                            # Resulting 1.31 mm matches measurement well
                                                            # GM: 1.2 in simulation?
# make sure that the total thickness matches that in the simulation
if (abs(pcbThickness - pcb_thickness_mm)>1e-6):
    print(f"WARNING: calculated pcb thickness ({pcbThickness} mm) differs from value in simulation ({pcb_thickness_mm} mm)!!!\n")
    # exit(0)

# constants:
# distance from signal trace to shield (HS) - from impedance vs. trace width vs. distance to ground layer 2D plot (Z = 50 Ohm)
# trace width (W) - min value
# trace thickness (T) - min value
# distance from shield to the edge of PCB
# http://www.analog.com/media/en/training-seminars/design-handbooks/Basic-Linear-Design/Chapter12.pdf, page 40
# signal trace
epsilonR = 4.8  # JP in CERN PCBs we have actually 4.8, not 4.4
# conversion factor: 1 inch = 25.4 mm
inch2mm = 25.4

# capa per length from maxwel1 (pF/mm) OBSOLETE
# strip layer has smaller capacitance due to traces running beneath the anti-etch
capa_per_mm = [0.1149*2] * numLayers  # JP Updated to value measured from PCBv2 T3 cell 14
                                      # JP I also double these values as these are not doubled later on as should
                                      # (because we have two signal-pad-plus-shields layers in one cell)
# JP I don't know if the "traces running beneath anti-etch" effect makes sense
# I put it to zero. There's zero shields running under signal pads of strips, except for
# one in the case of logical trace ordering, but we won't do that.

capa_per_mm[stripLayer] = 0  # 0.0575*2  # JP See justification above

stripLineCapaDensity = 0.1868  # Measured from PCBv2 T3 cell 1

# multiplicative factors
# for the trace, factor 2 because we have two HV plate / absorber capa per cell
nmultTrace = 2  #FIXME JP I already doubled the capa density but still now get a correct value. To be understood.
# for the shield, where we use maxwell, the extra factor 2 (two signal pad / shield capa) is already accounted for
nmultShield = 1
# dielectric constants
epsilonRLAr = 1.5  # LAr at 88 K #TODO Double-check from Martin's ATLAS source
epsilon0 = 8.854 / 1000.  # pF/mm


# BEGINNING OF THE CALCULATION
rmin = ecal_active_rmin_mm
activeTotal = ecal_active_thickness_mm
passiveThickness = absorber_thickness_mm
deltaTheta = grid_size_theta
inclinedTotal = total_electrode_length_mm
if verbose:
    print("Parameters used in the calculation:")
    print("-----------------------------------")
    print("min theta =", minTheta)
    print("max theta =", maxTheta)
    print("delta theta =", grid_size_theta)
    print("num theta =", numTheta)
    print("number of planes =", Nplanes)
    print("r min = %f mm" % rmin)
    print("active total = %f mm" % activeTotal)
    print("readout layer radial lengths (in cm) =", readoutLayerRadialLengths)
    print("electronde inclination (deg) =", electrode_inclination_deg)
    print("electrode total length = %f mm" % inclinedTotal)
    print("passive thickness = %f mm" % passiveThickness)
    print("number of layers =", numLayers)
    print("merged cells in theta =", nMergedThetaCells)
    print("merged modules =", nMergedModules)
    print("traces per layer =", tracesPerLayer)
    print("pcb thickness: %f mm" % pcbThickness)
    print("capa per mm (pF/mm) = " , capa_per_mm)
    print("")

# Fill the layer length, trace length, etc
readoutLayerParallelLengths = []
real_radial_separation = [rmin]
real_radial_depth = []
inclinations_wrt_radial_dir_at_middleRadialDepth = []
trace_length = []
dilution_factor = inclinedTotal / activeTotal  # Dilution factor takes care of capa decrease due to gap widening (right?)
trace_length_inner = 0
trace_length_outer = 0
outer = False
current_electrode_length = 0

for idx in range(numLayers):  # first pass to get all length parallel to the readout, real radial separation, inclination at the middle of the layer

    readoutLayerRadialLengths[idx] *= 10 # change from cm to mm    
    parallel_length = readoutLayerRadialLengths[idx] * dilution_factor

    # Tricky point: in the xml geo, you define 'radial'segmentation, but these depths will be the one parallel to the plates after scaling by the dilution factor --> even when setting constant radial depth, the geometry builder will make constant parallel length step, not constant radial steps

    readoutLayerParallelLengths.append(parallel_length)
    if outer:  # prepare the starting trace length when starting to extract by the back of the PCB
        trace_length_outer += parallel_length
    if tracesPerLayer[idx] == 0 and tracesPerLayer[idx - 1] == 0:
        outer = True
    # sqrt(r**2+(L1+i*L2)**2+2*r*(L1+i*L2)*cos(alpha)) where L1 = 2.68, L2=12.09, r=192, alpha=50)
    current_electrode_length += parallel_length
    real_radial_separation.append(sqrt(rmin * rmin + current_electrode_length * current_electrode_length + 2 * rmin * current_electrode_length * cos(angle)))
    real_radial_depth.append(real_radial_separation[idx + 1] - real_radial_separation[idx])
    # treating the fact that radial angle decreases when radial depth increase
    # angle comprise by lines from  1) Interaction point to inner right edge of a cell, 2) Interaction point to outer left edge of the considered cell (useful to get the plate angle with radial direction that changes with increasing R)
    # based on scalene triangle sine law A/sin(a) = B/sin(b) = C/sin(c) (outer left edge aligned on the Y axis)
    inclinations_wrt_radial_dir_at_middleRadialDepth.append(asin(rmin * sin(angle) / (real_radial_separation[idx] + ((real_radial_separation[idx + 1] - real_radial_separation[idx]) / 2))))


# second pass to get trace lengths
outer = False
for idx in range(numLayers):
    if tracesPerLayer[idx] == 0 and tracesPerLayer[idx - 1] == 0:  # we change direction
        outer = True
    if outer:
        trace_length.append(trace_length_outer)
        if idx == numLayers - 1:
            trace_length_outer == 0
            continue
        trace_length_outer -= readoutLayerParallelLengths[idx + 1]
    else:
        trace_length.append(trace_length_inner)
        trace_length_inner += readoutLayerParallelLengths[idx]
        
# JP The signal trace lengths are now the wrong way around in the array.
# (this did not have impact when transferline capa was neglected)
# Let's invert it and it should be fine for the capacitance calculation.
trace_length.reverse()

print('Readout radial lengths originally asked: ', readoutLayerRadialLengths)
print('Readout parallel lengths: ', readoutLayerParallelLengths)
print("Real radial separation: ", real_radial_separation)
print("Real radial depth: ", real_radial_depth)
print("inclinations_wrt_radial_dir_at_middleRadialDepth: ", [degrees(inclinations) for inclinations in inclinations_wrt_radial_dir_at_middleRadialDepth])
print("Signal trace length per layer: ", trace_length)


gStyle.SetOptStat(0)

cImpedance = TCanvas("cImpedance", "", 600, 800)
cImpedance.Divide(1, 2)
cImpedance.cd(1)
fImpedance = TF2("fImpedance", "60/sqrt([0])*log(1.9*(2*x+[1])/(0.8*y+[1]))", 0.04, 0.2, 0.04, 0.2)
fImpedance.SetTitle("Impedance vs trace width and distance to ground")
fImpedance.SetParameters(epsilonR, t)
fImpedance.Draw("colz")
fImpedance.GetXaxis().SetTitle("Distance to ground [mm]")
fImpedance.GetYaxis().SetTitle("Trace width [mm]")
cImpedance.cd(2)
fImpedance1D = TF1("fImpedance1D", "60/sqrt([0])*log(1.9*(2*x+[1])/(0.8*[2]+[1]))", 0.04, 0.2)
fImpedance1D.SetTitle("Impedance vs distance to ground")
fImpedance1D.SetParameters(epsilonR, t, w)
fImpedance1D.Draw()
fImpedance1D.GetXaxis().SetTitle("Distance to ground [mm]")
fImpedance1D.GetYaxis().SetTitle("Impedance [#Omega]")

# prepare the TH1
hCapTrace = []
hCapShield = []
hCapDetector = []
line_color_number = 1
line_style_number = 1
for i in range(0, numLayers):
    if line_color_number == 8:
        line_color_number = 22
    if line_style_number == 8:
        line_style_number = 1
    # traces
    hCapTrace.append(TH1F())
    hCapTrace[i].SetBins(numTheta, minTheta, maxTheta)
    hCapTrace[i].SetLineColor(line_color_number)
    hCapTrace[i].SetLineStyle(line_style_number)
    hCapTrace[i].SetLineWidth(2)
    hCapTrace[i].SetTitle("Stripline capacitance; #theta; Capacitance [pF]")
    hCapTrace[i].SetName("hCapacitance_traces"+str(i))
    # shields
    hCapShield.append(TH1F())
    hCapShield[i].SetBins(numTheta, minTheta, maxTheta)
    hCapShield[i].SetLineColor(line_color_number)
    hCapShield[i].SetLineStyle(line_style_number)
    hCapShield[i].SetLineWidth(2)
    hCapShield[i].SetTitle("Signal pads - ground shields capacitance; #theta; Capacitance [pF]")
    hCapShield[i].SetName("hCapacitance_shields"+str(i))
    # area
    hCapDetector.append(TH1F())
    hCapDetector[i].SetBins(numTheta, minTheta, maxTheta)
    hCapDetector[i].SetLineColor(line_color_number)
    hCapDetector[i].SetLineStyle(line_style_number)
    hCapDetector[i].SetLineWidth(2)
    hCapDetector[i].SetTitle("Signal pad - absorber capacitance; #theta; Capacitance [pF]")
    hCapDetector[i].SetName("hCapacitance_detector"+str(i))
    if line_color_number > 8:
        line_color_number += 10
    else:
        line_color_number += 1
    line_style_number += 1

cTrace = TCanvas("cTrace", "", 600, 400)
cShield = TCanvas("cShield", "", 600, 400)
cDetector = TCanvas("cDetector", "", 600, 400)

legend = TLegend(0.1, 0.693, 0.8, 0.9)
legend.SetHeader("Longitudinal layers")
legend.SetNColumns(4)
capa_shield_max = 0
capa_det_max = 0
cellcapas = []
for i in range(0, len(readoutLayerParallelLengths)):
    print("--------------")
    for index in range(0, numTheta):
        theta = minTheta + index * deltaTheta
        thetaNext = minTheta + (index + 1) * deltaTheta
        eta = -log(tan(theta / 2.0))
        deltaEta = abs(-log(tan((minTheta + (index + 1) * deltaTheta) / 2.0)) - eta)
        if (debug):
            print("theta = ", theta)
            print("eta = ", eta)
            print("delta eta = ", deltaEta)

        # take into account the inclination in theta
        traceLength = trace_length[i] / sin(theta)
        # print("Layer %d trace length %f"%(i+1, traceLength))
        # Trace capacitance (stripline) - not used since already accounted for elsewhere
        logStripline = log(3.1 * hs / (0.8 * w + t))
        # analytical formula
        #capacitanceTrace = nmultTrace * 1 / inch2mm * 1.41 * epsilonR / logStripline * traceLength

        # JP calculate with value from measurement instead; analytical formula has assumptions that are not fulfilled
        capacitanceTrace = nmultTrace*stripLineCapaDensity*traceLength

        hCapTrace[i].SetBinContent(index + 1, capacitanceTrace)

        # Shield capacitance (microstrip)
        cellLength = readoutLayerParallelLengths[i] / sin(theta)
        logMicrostrip = log(5.98 * hm / (0.8 * ws + t))
        # analytical formula (nmultShield = 2)
        # capacitanceShield = nmultShield * nMergedModules[i] * cellLength * tracesPerLayer[i] * 1 / inch2mm * 0.67 * (epsilonR + 1.41) / logMicrostrip
        # from maxwell (nmultShield = 1)
        # dont multiply by nMergedThetaCells:  the shield/pad capa is reasonably independent of the cell size and the fact that there is some merging
        # done for theta cells is already taken into account by the tracesPerLayer
        capacitanceShield = nmultShield * nMergedModules[i] * cellLength * tracesPerLayer[i] * capa_per_mm[i]
        if capacitanceShield > capa_shield_max:
            capa_shield_max = capacitanceShield
        hCapShield[i].SetBinContent(index + 1, capacitanceShield)

        # Detector area (C = epsilon*A/d)
        # area = ( radius[i] * ( 1 / (tan(2. * atan(exp(- (index + 1) * deltaEta)))) -  1 / (tan(2. * atan(exp(- index * deltaEta))) ) )
        #         + radius[i + 1] * ( 1 / (tan(2. * atan(exp(- (index + 1) * deltaEta)))) -  1 / (tan(2. * atan(exp(- index * deltaEta))) ) )
        #         ) / 2. * (radius[i+1] - radius[i])
        # distance = (radius[i+1] + radius[i]) / 2. * pi / Nplanes * cos (angle) - pcbThickness / 2. - passiveThickness / 2.

        # Detector area (C = epsilon*A/d)
        area = abs(real_radial_separation[i] * (1 / tan(thetaNext) - 1 / tan(theta)) + real_radial_separation[i + 1] * (1 / tan(thetaNext) - 1 / tan(theta))
                 ) / 2. * (real_radial_separation[i+1] - real_radial_separation[i])

        # get the cell size perpendicular to the plate direction from the cell size on the circle at given radius and the inclination w.r.t. radial dir, then remove the PCB and lead thickness (no need for any factor here because we are perpendicular to the PCB and lead plates) --> gives the LAr gap size perpendicular
        distance = (2 * pi * (real_radial_separation[i+1] + real_radial_separation[i]) / 2. / Nplanes * cos (inclinations_wrt_radial_dir_at_middleRadialDepth[i]) - pcbThickness - passiveThickness) / 2. # divided by two because two lar gap per cell
        distance += hhv  # the capa is between signal plate and absorber --> need to add distance between HV plate and signal pad
        distance += t  # the capa is between signal plate and absorber --> need to add distance between HV plate and signal pad
        if (abs(theta - pi / 2.) < 1e-4):
            print("LAr gap size (perpendicular) + hhv + t: %f mm" % distance)
        # capacitanceDetector = nMergedModules[i] * nMergedThetaCells[i] * 2 * epsilon0 * epsilonRLAr * area / distance  # factor 2 is because there are 2 LAr gaps for each cell
        # JP Updated to include the effect of 100um dielectric layer
        capacitanceDetector = ( nMergedModules[i] * nMergedThetaCells[i] * 2 * epsilon0 * epsilonRLAr * epsilonR * area ) / ( (distance-hhv)*epsilonR + hhv*epsilonRLAr )
        # (this is the equation for the two-dielectric sandwitch capacitors, equivalent to two capacitors in series)

        hCapDetector[i].SetBinContent(index + 1, capacitanceDetector)
        if capacitanceDetector > capa_det_max:
            capa_det_max = capacitanceDetector
        if (abs(theta - pi / 2.) < 1e-4):
            print("layer %d" % (i + 1), "theta=%f" % theta, ": capacitanceTrace: %.0f pF," % capacitanceTrace, "capacitanceShield: %.0f pF," % capacitanceShield, "capacitanceDetector: %.0f pF," %capacitanceDetector, "total/2: %.0f pF" % ((capacitanceTrace + capacitanceShield + capacitanceDetector)/2.))
            # , "distance %.1f mm" %distance
            cellcapas.append(round((capacitanceTrace + capacitanceShield + capacitanceDetector)/2.))

    # Draw
    cTrace.cd()
    if i == 0:
        hCapTrace[i].Draw()
    else:
        hCapTrace[i].Draw("same")
    legend.AddEntry(hCapTrace[i], "layer %d" % (i + 1), "l")
    cShield.cd()
    if i == 0:
        hCapShield[i].Draw()
    else:
        hCapShield[i].Draw("same")
    cDetector.cd()
    if i == 0:
        hCapDetector[i].Draw()
    else:
        hCapDetector[i].Draw("same")

cellcapas.reverse()

print("Cell capas per electrode:")
print(cellcapas)

maximum = capa_shield_max

plots = TFile(filename, "RECREATE")

for i in range(0, len(readoutLayerParallelLengths)):
    hCapTrace[i].SetMinimum(0)
    hCapTrace[i].SetMaximum(maximum * 1.8)
    hCapTrace[i].Write()
    hCapShield[i].SetMinimum(0)
    hCapShield[i].SetMaximum(capa_shield_max * 1.5)
    hCapShield[i].Write()
    hCapDetector[i].SetMinimum(0)
    hCapDetector[i].SetMaximum(capa_det_max * 1.5)
    hCapDetector[i].Write()

cTrace.cd()
legend.Draw()
cTrace.Update()
cTrace.Write()
cTrace.Print("plots/capa_trace%s.png" % apdx)
cTrace.Print("plots/capa_trace%s.pdf" % apdx)
cShield.cd()
legend.Draw()
cShield.Update()
cShield.Write()
cShield.Print("plots/capa_shield%s.png" % apdx)
cShield.Print("plots/capa_shield%s.pdf" % apdx)
cDetector.cd()
legend.Draw()
cDetector.Update()
cDetector.Write()
cDetector.Print("plots/capa_detector%s.png" % apdx)
cDetector.Print("plots/capa_detector%s.pdf" % apdx)

fImpedance.Write()
fImpedance1D.Write()

#closeInput = raw_input("Press ENTER to exit")
