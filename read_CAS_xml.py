import xml.etree.ElementTree as et 
import pandas as pd
import os
import glob
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


path = '/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/'
extension = 'xml'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(len(result))
predicted_pka = []
u_list = []
name = 'c'

'''
{'predicted-boiling-point', 'experimental-melting-point', 'experimental-optical-rotatory-power', 'predicted-pKa', 
'experimental-median-lethal-dose', 'predicted-density', 'experimental-density', 'experimental-boiling-point'}
'''

cas = []
boiling_point = []
densities =[]
molar_volumes = []
enthalpies_of_vaporization = []
logp = []
vapor_pressures = []
logd = []
pka = []
bio_concentration_factor = []
koc = []
mass_solubilities = []
mass_intrinsic_solubilities = []
molar_solubilities = []
molar_intrinsic_solubilities = []
flash_points = []
hydrogen_bond_acceptors_counts = []
hydrogen_donor_acceptor_sums = []
freely_rotatable_bonds_counts = []
hydrogen_donor_counts = []
polar_surface_areas = []
experimental_boiling_point = []
experimental_melting_point = []
experimental_density = []
experimental_optical_rotatory_power = []
experimental_median_lethal_dose = []

for i in result:
    xtree = et.parse(i)
    xroot = xtree.getroot()

    
    try:
        cas.append(xroot.find("substance-uri").text)
    except(AttributeError):
        cas.append(np.nan)
        
    try:
        boiling_point.append(xroot.find("properties/predicted-properties/predicted-boiling-points/predicted-boiling-point/").text)
    except(AttributeError):
        boiling_point.append(np.nan)      
    
    try:
        densities.append(xroot.find("properties/predicted-properties/predicted-densities/predicted-density/").text)
    except(AttributeError):
        densities.append(np.nan)
        
        
    try:
        molar_volumes.append(xroot.find("properties/predicted-properties/predicted-molar-volumes/predicted-molar-volume/").text)
    except(AttributeError):
        molar_volumes.append(np.nan)
        
    try:
        enthalpies_of_vaporization.append(xroot.find("properties/predicted-properties/predicted-enthalpies-of-vaporization/predicted-enthalpy-of-vaporization/").text)
    except(AttributeError):
        enthalpies_of_vaporization.append(np.nan)
        
        
    try:
        logp.append(xroot.find("properties/predicted-properties/predicted-logPs/predicted-logP/").text)
    except(AttributeError):
        logp.append(np.nan)
              
    
    try:
        vapor_pressures.append(xroot.find("properties/predicted-properties/predicted-vapor-pressures/predicted-vapor-pressure/").text)
    except(AttributeError):
        vapor_pressures.append(np.nan)    
    try:
        logd.append(xroot.find("properties/predicted-properties/predicted-logDs/predicted-logD/").text)
    except(AttributeError):
        logd.append(np.nan)
        
    
    try:
        pka.append(xroot.find("properties/predicted-properties/predicted-pKas/predicted-pKa/").text)
    except(AttributeError):
        pka.append(np.nan)
        
    try:
        bio_concentration_factor.append(xroot.find("properties/predicted-properties/predicted-bioconcentration-factors/predicted-bioconcentration-factor/").text)
    except(AttributeError):
        bio_concentration_factor.append(np.nan)
        
    try:
        koc.append(xroot.find("properties/predicted-properties/predicted-Kocs/predicted-Koc/").text)
    except(AttributeError):
        koc.append(np.nan)        

    
    
    try:
        mass_solubilities.append(xroot.find("properties/predicted-properties/predicted-mass-solubilities/predicted-mass-solubility/").text)
    except(AttributeError):
        mass_solubilities.append(np.nan)    
        
        
    try:
        mass_intrinsic_solubilities.append(xroot.find("properties/predicted-properties/predicted-mass-intrinsic-solubilities/predicted-mass-intrinsic-solubility/").text)
    except(AttributeError):
        mass_intrinsic_solubilities.append(np.nan)    
        
    try:
        molar_solubilities.append(xroot.find("properties/predicted-properties/predicted-molar-solubilities/predicted-molar-solubility/").text)
    except(AttributeError):
        molar_solubilities.append(np.nan)    
    
    try:
        molar_intrinsic_solubilities.append(xroot.find("properties/predicted-properties/predicted-molar-intrinsic-solubilities/predicted-molar-intrinsic-solubility/").text)
    except(AttributeError):
        molar_intrinsic_solubilities.append(np.nan)   
    
    try:
        flash_points.append(xroot.find("properties/predicted-properties/predicted-flash-points/predicted-flash-point/").text)
    except(AttributeError):
        flash_points.append(np.nan)        
    try:
        hydrogen_bond_acceptors_counts.append(xroot.find("properties/predicted-properties/predicted-hydrogen-acceptors-counts/predicted-hydrogen-acceptors/").text)
    except(AttributeError):
        hydrogen_bond_acceptors_counts.append(np.nan)        
        
        
    try:
        hydrogen_donor_acceptor_sums.append(xroot.find("properties/predicted-properties/predicted-hydrogen-donor-acceptor-sums/predicted-hydrogen-donor-acceptor-sum/").text)
    except(AttributeError):
        hydrogen_donor_acceptor_sums.append(np.nan)       
        
    try:
        freely_rotatable_bonds_counts.append(xroot.find("properties/predicted-properties/predicted-freely-rotatable-bonds-counts/predicted-freely-rotatable-bonds/").text)
    except(AttributeError):
        freely_rotatable_bonds_counts.append(np.nan)    

        
    try:
        hydrogen_donor_counts.append(xroot.find("properties/predicted-properties/predicted-hydrogen-donors-counts/predicted-hydrogen-donors/").text)
    except(AttributeError):
        hydrogen_donor_counts.append(np.nan)      
        
        
    try:
        polar_surface_areas.append(xroot.find("properties/predicted-properties/predicted-polar-surface-areas/predicted-polar-surface-area/").text)
    except(AttributeError):
        polar_surface_areas.append(np.nan)              
        

        
    try:
        experimental_boiling_point.append(xroot.find("properties/preferred-values/experimental-boiling-point/").text)
    except(AttributeError):
        experimental_boiling_point.append(np.nan) 
        
        
    try:
        experimental_melting_point.append(xroot.find("properties/preferred-values/experimental-melting-point/").text)
    except(AttributeError):
        experimental_melting_point.append(np.nan)   
        

    try:
        experimental_density.append(xroot.find("properties/preferred-values/experimental-density/").text)
    except(AttributeError):
        experimental_density.append(np.nan)   
        
    try:
        experimental_optical_rotatory_power.append(xroot.find("properties/preferred-values/experimental-optical-rotatory-power/").text)
    except(AttributeError):
        experimental_optical_rotatory_power.append(np.nan)    
        
    try:
        experimental_median_lethal_dose.append(xroot.find("properties/preferred-values/experimental-median-lethal-dose/").text)
    except(AttributeError):
        experimental_median_lethal_dose.append(np.nan)  


    
df = pd.DataFrame({"CAS#":cas, "boiling_point": boiling_point, "densities": densities, "molar_volumes":molar_volumes, 
                   "enthalpies_of_vaporization":enthalpies_of_vaporization, "logP": logp, "vapor_pressures":vapor_pressures,
                   "logD": logd, "pKa": pka, 'bio_concentration_factor': bio_concentration_factor, "koc":koc, "mass_solubilities":mass_solubilities,
                   "mass_intrinsic_solubilities":mass_intrinsic_solubilities, "molar_solubilities":molar_solubilities, 
                   "molar_intrinsic_solubilities":molar_intrinsic_solubilities, "flash_points":flash_points, "hydrogen_bond_acceptors_counts":hydrogen_bond_acceptors_counts, 
                   "hydrogen_donor_acceptor_sums": hydrogen_donor_acceptor_sums, "freely_rotatable_bonds_counts":freely_rotatable_bonds_counts, 
                   "hydrogen_donor_counts":hydrogen_donor_counts, "polar_surface_areas":polar_surface_areas, 'experimental_boiling_point':experimental_boiling_point,
                   'experimental_melting_point':experimental_melting_point, 'experimental_density':experimental_density, 
                   'experimental_optical_rotatory_power':experimental_optical_rotatory_power, 'experimental_median_lethal_dose':experimental_median_lethal_dose                   
                   })
print(df.head())

df.to_csv("full_cas_predicted_experiment.csv")   
