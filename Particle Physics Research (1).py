#!/usr/bin/env python
# coding: utf-8

# In[6]:


### If this is the first time you are running this notebook, you may need to uncomment the following 
### and run them once and only once. 
#!pip install mplhep --user 
#!pip install --user awkward1
#!pip install --user hist 
get_ipython().system('pip install tqdm')
### Python library imports
import uproot4 as uproot
#print(uproot.__version__)
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import awkward as ak
import math
from hist import Hist
import hist as histlib
import hist.intervals
import hist.plot
import itertools
from scipy.stats import gmean
from smutools import *
import vector

# Load style sheet (we'll use the ATLAS one!)
plt.style.use(hep.style.ATLAS)


# In[7]:


# Global information for watermarking plots, etc.
global data_folder
#file_dir = "/users/ssekula/work/EIC/ATHENA/delphes_EIC"  (inital dir)
file_dir = "/users/ssekula/work/Research-Summer-2022/MuIC/Simulation/Delphes"

data_folder = "."
filepattern1 = "pmu_275x960_ccdis_vbf_hcc-SM5F_Hcc_NLO-SnowmassCalo.root"
filepattern2 = "pmu_275x960_ccdis_dicharm-SM5F_Hcc_NLO-SnowmassCalo.root"
filepattern3 = 'pmu_275x960_ncdis_vbf_hcc-SM5F_Hcc_NLO-SnowmassCalo.root'
filepattern4 = "pmu_275x960_ncdis_dicharm-SM5F_Hcc_NLO-SnowmassCalo.root"
treename = "Delphes"


# Flags

Debug = True


# In[8]:


filepattern5 = "pmu_275x960_ncdis_dicharm-SM5F_Hcc_NLO.root"


# In[ ]:


def superflat(df):
    return np.concatenate(df).ravel()

# General classes for managing things like histograms while looping in UPROOT4 over
# input ROOT files. Also classes for quickly making nicely formatted and standardized
# plots.

class HistManager:
    def __init__(self, verbose=False):
        self.histograms = {}
        self.verbose = verbose
    

    def clear(self):
        self.histograms = {}

    def add(self, df, name, x, cuts=[], nbins=100, range=[0,1]):
        
        
        #branch = df[varString].ravel()
    
        mask = np.ones(len(x), dtype=bool)

        for cut in cuts:
            if len(mask) != len(cut):
                print("Error: existing bitmask and new cut bitmask have different sizes!")

            mask = np.bitwise_and(np.array(mask), np.array(cut))
        
        if name not in self.histograms:
            if self.verbose:
                print(f"Creating histogram named {name} ... ")
            self.histograms[name] = np.histogram(x[mask], nbins, range)
        else:
            (hist, bins) = np.histogram(x[mask], nbins, range)
            newhist = np.sum([hist, self.histograms[name][0]], axis=0)
            self.histograms[name] = (newhist, self.histograms[name][1])
    
    def get_all(self):
        return self.histograms

    def get(self, name):
        return self.histograms[name]

    def get_hist(self, name):
        return self.histograms[name][0]
    
    def get_bins(self, name):
        return self.histograms[name][1]

def ComparePlot(histfactory=None, histograms=[], labels=[], colors=[], 
                ylimits=[0,1], logplot=False, 
                axislabels=["x", "y"], 
                special_text="", 
                weights=[],
                verbose=False, 
                density=True,
                **kwargs):
    global version
    if histfactory == None:
        return
    fig, ax = plt.subplots()

    # Handle the weights
    if len(weights) != len(histograms):
        weights = np.ones(len(histograms))

    for index, name in enumerate(histograms):
        hep.histplot(histfactory.get_hist(name)*weights[index], 
                     histfactory.get_bins(name), 
                     color=colors[index], label=labels[index], density=density,
                     **kwargs)

    if logplot:
        plt.yscale('log')
    plt.ylim(ylimits)
    plt.xlabel(axislabels[0])
    plt.ylabel(axislabels[1])
    plt.legend()
    plt.title(r"$\bf{MuIC}$" +' simulation [Fast Simulation] \nMG5_aMC@NLO + Pythia8 + Delphes',fontsize=17)
    
    if special_text != "":
        plt.text(0.02,0.93, special_text, transform=ax.transAxes)
    
    plt.tight_layout()
    plotname = f"{'_'.join(histograms)}.png"
    if verbose:
        print(f"Creating {plotname}")
    plt.savefig(plotname, dpi=200)
    if not verbose:
        plt.close()
        
def CoveragePlot(angles, magnitudes, 
                 anglerange=np.radians(np.linspace(0, 180, 90)), 
                 magnituderange=np.linspace(0,1000,10), 
                 xlabel = "Magnitude",
                 title = "Coverage Plot"):

 
    values, thetaedges, redges = np.histogram2d(angles, magnitudes, bins=[anglerange, magnituderange])
    r, theta = np.meshgrid( redges[:-1], thetaedges[:-1])


    # Make the plot
    fig,ax = plt.subplots(subplot_kw=dict(projection='polar'),dpi=300)
    #ax.contourf(theta, r, values,levels=18)
    ax.contourf(theta, r, values)
    list=[0,np.pi/6,np.pi/3,np.pi/2,4*np.pi/6,5*np.pi/6,np.pi]
    ax.set_xticks(list)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    plt.xlabel(xlabel,labelpad=-40,fontsize=12)
    plt.title(f"{title}", fontsize=18)
#     plt.text(3.45/2,np.max(magnitudes)+12.5,'Degrees',fontsize=18,multialignment='right')
    ax.tick_params(axis='x', labelsize=10 )
    ax.tick_params(axis='y', labelsize=10 )
    plt.tight_layout()

    filename = title.replace(" ","_")
    filename = filename.replace(",","_")
    filename = filename.replace(">","to-")
    filename = filename.replace("<","from-")
    filename = filename.replace("\\","_")
    filename = filename.replace("/","_")
    filename = filename.replace("$","_")

    plt.savefig(f"{filename}.png")
    plt.savefig(f"{filename}.pdf")

    
### UPROOT4 Analysis Tools

class SampleHandler():

    def __init__(self, name="", files="", tree="", lumi_target=-1, xsection=-1):

        self.name = name
        self.files = files
        self.tree = tree
        self.lumi_target = lumi_target
        self.xsection = xsection

        # Open the file(s) in UPROOT
        self.data = uproot.open(f"{self.files}:{self.tree}")

        ngen = self.data.num_entries

        self.weight = 1.0
        print(f"Sample {name} has {ngen} generated events, a target luminosity of {self.lumi_target} inverse fb, and a theory cross-section of {self.xsection} fb ... ")

        if self.lumi_target == -1 or self.xsection == -1 or ngen <= 0:
            self.weight = 1.0
        else:
            self.weight = self.xsection*self.lumi_target/ngen
        print(f" ... setting sample weight to {self.weight}.")


#hi   
def EventLoop(files={}, branches=[], histmanager=None, analyses=[], cutflow=None, maxevents=-1, sample_handler=None):

    nevents = 0

    the_handler = sample_handler

    if the_handler == None:
        # Make a sample handler but default weights to 1.0
        samplename = list(files.keys())[0]
        the_handler = SampleHandler("a_sample", files=samplename, tree=files[samplename])

    nevents = 0
            
    files = {f"{the_handler.files}": the_handler.tree}
    for events in uproot.iterate([files], 
                                branches, 
                                library="np",
                                step_size=20000):

        
        df = pd.DataFrame(events, columns=events.keys())
        nevents += len(df)

        for analysis in analyses:
            analysis(dataframe=df, 
                     histmanager=histmanager,
                     cutflow=cutflow)
        
        print(f"... processed {nevents} ... ")

        if maxevents > 0 and nevents >= maxevents:
            break
            
    if cutflow != None:
        cutflow.passCut("All Events", nevents)
        
    print(f"Processed {nevents} collisions (events).")
    

class EventWeight():
    def __init__(self, samplename="", luminosity=1, ngen=0):
        self.samplename = samplename
        self.luminosity = luminosity
        self.ngen = ngen

        xsection = {}
        xsection["pmu_275x960_ccdis_vbf_hcc-SM5F_Hcc_NLO"] = 65 * 0.0289
        xsection["pmu_275x960_ncdis_vbf_hcc-SM5F_Hcc_NLO"] = 12 * 0.0289
        xsection["pmu_275x960_ccdis_dicharm-SM5F_Hcc_NLO"] = 5.4
        xsection["pmu_275x960_ncdis_dicharm-SM5F_Hcc_NLO"] = 43.7e3

        self.weight = 0.0

        if self.samplename in xsection.keys():
            self.weight = xsection[self.samplename]*self.luminosity/self.ngen
        else:
            print(f"Sample {self.samplename} not known to this tool. Returning weight of 1.0")
            self.weight = 1.0
        
    def w(self):
        return self.weight
        

        


class CutFlow():
    def __init__(self, verbose=False, name="", sample_handler=None):
        self.cutflow = {}
        self.cuts = []
        self.verbose = verbose
        self.name = name
        self.sample_handler = sample_handler
    
    def addCut(self, name):
        if name in self.cutflow:
            print(f"Cannot add cut {name} --- already exists!")
        else:
            self.cuts.append(name)
            self.cutflow[name] = 0
    
        return
    
    def passCut(self, name, number=1):
        weight = 1.0
        if self.sample_handler != None:
            weight = self.sample_handler.weight
        if name not in self.cutflow:
            self.cuts.append(name)
            self.cutflow[name] = number * weight
        else:
            self.cutflow[name] += number * weight
    
    def print(self, denominator=""):
        print(f"Cut Flow {self.name}")
        for i,name in enumerate(self.cuts):
            percentage = ""
            relative = ""
            if denominator in self.cuts:
                pct = self.cutflow[name]/self.cutflow[denominator]*100
                percentage = f"({pct:.2f}%)"
            if i > 0:
                pct = self.cutflow[name]/self.cutflow[self.cuts[i-1]]*100
                relative = f"({pct:.2f}%)"
            else:
                relative = "(N/A)"
                
            print(f"{name:>25}: {self.cutflow[name]:.3f} {percentage} {relative}")


#
# Code for running inside Analysis Functions (e.g. using DataFrame.apply())
#

def Helper_DiJetInfo(event, use_flavor=-1):
    # Find all dijet pairs passing certain requirements
    # If use_flavor set to 4 or 5, etc. it will require jets to be of that flavor
    #    -> currently uses truth information only
    # 

    Jet_Info = {}

    Jet_Info['Jet.PT'] = event['Jet.PT']
    Jet_Info['Jet.Eta'] = event['Jet.Eta']
    Jet_Info['Jet.Phi'] = event['Jet.Phi']
    Jet_Info['Jet.Mass'] = event['Jet.Mass']

    Jet_Info['Jet.Flavor'] = event['Jet.Flavor']

    if use_flavor >= 0:
        is_flavored = (Jet_Info['Jet.Flavor'] == use_flavor)

        for variable in Jet_Info:
            Jet_Info[variable] = Jet_Info[variable][is_flavored]


    
    # Construct dijet objects
    DiJet_p4 = vector.obj(pt=0, eta=0, phi=0, mass=0)
    DiJets = []

    for JetPair in itertools.combinations(range( len(Jet_Info['Jet.PT'])),2):
        jet1 = JetPair[0]
        jet2 = JetPair[1]

        jet1_P4 = vector.obj(pt=Jet_Info['Jet.PT'][jet1], 
                             eta=Jet_Info['Jet.Eta'][jet1], 
                            phi=Jet_Info['Jet.Phi'][jet1],
                            mass=Jet_Info['Jet.Mass'][jet1])
        jet2_P4 = vector.obj(pt=Jet_Info['Jet.PT'][jet2], 
                             eta=Jet_Info['Jet.Eta'][jet2], 
                            phi=Jet_Info['Jet.Phi'][jet2],
                            mass=Jet_Info['Jet.Mass'][jet2])

        DiJet_P4 = jet1_P4 + jet2_P4

        DiJets.append(DiJet_P4)

    event['DiJet.Mass'] = []
    event['DiJet.PT'] = []
    event['DiJet.Eta'] = []
    event['DiJet.Phi'] = []

    for dijet in DiJets:
        event['DiJet.Mass'].append( dijet.m )
        event['DiJet.PT'].append( dijet.pt )
        event['DiJet.Eta'].append( dijet.eta )
        event['DiJet.Phi'].append( dijet.phi )

    # Sort by DiJet.PT


    return event


# In[9]:


# Create a SampleHandler
hcc_ccdis = SampleHandler(name="CCDIS H->cc", 
                          files="/users/ssekula/work/Research-Summer-2022/MuIC/Simulation/Delphes/pmu_275x960_ccdis_vbf_hcc-SM5F_Hcc_NLO-SnowmassCalo.root",
                          tree="Delphes",
                          lumi_target = 1, # inverse fb
                          xsection = 65 * 0.0298 # fb
                          )

di_charm = SampleHandler(name="CCDIS di_charm", 
                          files="/users/ssekula/work/Research-Summer-2022/MuIC/Simulation/Delphes/pmu_275x960_ccdis_dicharm-SM5F_Hcc_NLO-SnowmassCalo.root",
                          tree="Delphes",
                          lumi_target = 1, # inverse fb
                          xsection = 5.4 # fb
                          )
hcc_nccdis = SampleHandler(name="NCDIS H->cc", 
                          files="/users/ssekula/work/Research-Summer-2022/MuIC/Simulation/Delphes/pmu_275x960_ncdis_vbf_hcc-SM5F_Hcc_NLO-SnowmassCalo.root",
                          tree="Delphes",
                          lumi_target = 1, # inverse fb
                          xsection = 12 * 0.0298 # fb
                          )
ncdi_charm = SampleHandler(name="NCDIS di_charm", 
                          files="/users/ssekula/work/Research-Summer-2022/MuIC/Simulation/Delphes/pmu_275x960_ncdis_dicharm-SM5F_Hcc_NLO-SnowmassCalo.root",
                          tree="Delphes",
                          lumi_target = 1, # inverse fb
                          xsection = 43700 # fb
                          )
ncdi_charm2 = SampleHandler(name="NCDIS di_charm2", 
                          files="/users/ssekula/work/Research-Summer-2022/MuIC/Simulation/Delphes/pmu_275x960_ncdis_dicharm-SM5F_Hcc_NLO.root",
                          tree="Delphes",
                          lumi_target = 1, # inverse fb
                          xsection = 43700 # fb
                          )


# In[14]:


global missing_met_data

histograms = HistManager()
histograms.clear()
missingenergy_branches = ['MissingET.MET','MissingET.Phi','Jet.PT','Jet.Eta','Jet.Phi','Jet.Mass','Jet.Flavor']
delphes_files1 = {f"{file_dir}/{data_folder}/{filepattern1}": treename}





missing_met_data = np.array([])

missing_met_data2 = np.array([])


def MET_Analysis(dataframe, histmanager, cutflow):
    
    global missing_met_data
    


   
    dataframe = dataframe.apply(Helper_DiJetInfo,axis = 1,use_flavor= 4)
    missing_met = superflat(dataframe["MissingET.MET"])
    missing_met_data = np.concatenate((missing_met_data, missing_met))
    
    histograms.add(dataframe,'met1',missing_met,cuts=[],nbins=100,range=[0,125])

EventLoop(sample_handler=hcc_ccdis, 
          branches=missingenergy_branches, 
          histmanager=histograms, 
          analyses=[MET_Analysis])



global missing_met_data2

delphes_files2 = {f"{file_dir}/{data_folder}/{filepattern2}": treename}


missing_met_data2 = np.array([])


def MET_Analysis2(dataframe, histmanager, cutflow):
    
    global missing_met_data2
    
    dataframe = dataframe.apply(Helper_DiJetInfo,axis = 1,use_flavor= 4)
    missing_met2 = superflat(dataframe["MissingET.MET"])
    missing_met_data2 = np.concatenate((missing_met_data2, missing_met2))
    histograms.add(dataframe,'met2',missing_met2,cuts=[],nbins=100,range=[0,125])


EventLoop(sample_handler=di_charm, 
          branches=missingenergy_branches, 
          histmanager=histograms, 
          analyses=[MET_Analysis2])

global missing_met_data3

delphes_files3 = {f"{file_dir}/{data_folder}/{filepattern3}": treename}


missing_met_data3 = np.array([])


def MET_Analysis3(dataframe, histmanager, cutflow):
    
    global missing_met_data3
    
    dataframe = dataframe.apply(Helper_DiJetInfo,axis = 1,use_flavor= 4)
    missing_met3 = superflat(dataframe["MissingET.MET"])
    missing_met_data3 = np.concatenate((missing_met_data3, missing_met3))
    histograms.add(dataframe,'met3',missing_met3,cuts=[],nbins=100,range=[0,125])


EventLoop(sample_handler=hcc_nccdis, 
          branches=missingenergy_branches, 
          histmanager=histograms, 
          analyses=[MET_Analysis3])


missing_met_data4 = np.array([])

global missing_met_data4

def MET_Analysis4(dataframe, histmanager, cutflow):
    
    global missing_met_data4
    
    dataframe = dataframe.apply(Helper_DiJetInfo,axis = 1,use_flavor= 4)
    missing_met4 = superflat(dataframe["MissingET.MET"])
    missing_met_data4 = np.concatenate((missing_met_data4, missing_met4))
    histograms.add(dataframe,'met4',missing_met4,cuts=[],nbins=100,range=[0,125])


EventLoop(sample_handler=ncdi_charm, 
          branches=missingenergy_branches, 
          histmanager=histograms, 
          analyses=[MET_Analysis4])


# In[ ]:


missing_met_data5 = np.array([])

def MET_Analysis5(dataframe, histmanager, cutflow):
    
    global missing_met_data5
    
    dataframe = dataframe.apply(Helper_DiJetInfo,axis = 1,use_flavor= 4)
    missing_met5 = superflat(dataframe["MissingET.MET"])
    missing_met_data5 = np.concatenate((missing_met_data5, missing_met5))
    histograms.add(dataframe,'met5',missing_met5,cuts=[],nbins=100,range=[0,125])


EventLoop(sample_handler=ncdi_charm2, 
          branches=missingenergy_branches, 
          histmanager=histograms, 
          analyses=[MET_Analysis5])


                          


# In[ ]:





# In[16]:



ComparePlot(histfactory=histograms, 
            histograms=['met1','met2','met3','met5'], 

            labels=['CC Signal','CC di-charm Background', 'NC Signal','NC di-charm Background'],
            colors=['b','r','m','g'], 
            weights = [hcc_ccdis.weight,di_charm.weight,hcc_nccdis.weight,ncdi_charm2.weight],
            ylimits=[0,0.1],
            logplot=False, 
            density = True,
            axislabels=["MET[GeV]", "Density of Events"],
           verbose=True)


# In[ ]:





# In[ ]:





# In[ ]:




