import pandas as pd
from functools import reduce
#predicted-logDs; predicted-Kocs; predicted-bioconcentration-factors;
#predicted-mass-solubilities; predicted-molar-solubilities; 


def combine_prop(prop):
    df_pkas1 = pd.read_csv("/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/{}/{}_10k.csv".format(prop, prop))
    del df_pkas1['Unnamed: 0']
    df_pkas2 = pd.read_csv("/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/{}/{}_20k.csv".format(prop,prop))
    del df_pkas2['Unnamed: 0']
    df_pkas3 = pd.read_csv("/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/{}/{}_30k.csv".format(prop,prop))
    del df_pkas3['Unnamed: 0']
    df_pkas4 = pd.read_csv("/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/{}/{}_40k.csv".format(prop,prop))
    del df_pkas4['Unnamed: 0']
    df_pkas5 = pd.read_csv("/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/{}/{}_43k.csv".format(prop, prop))
    del df_pkas5['Unnamed: 0']
    frames_pka =[df_pkas1, df_pkas2, df_pkas3, df_pkas4, df_pkas5]
    print(df_pkas1.head())
    
    df_merged = pd.concat(frames_pka)
    print(len(df_merged))
    return df_merged
frames = []    
frames.append(combine_prop('predicted-pKas'))
frames.append(combine_prop('predicted-logDs'))
frames.append(combine_prop('predicted-Kocs'))
frames.append(combine_prop('predicted-bioconcentration-factors'))
frames.append(combine_prop('predicted-mass-solubilities'))
frames.append(combine_prop('predicted-molar-solubilities'))


def combine_sub(prop):
    df_merged = pd.read_csv("/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Substance/frameworks/{}.csv".format(prop, prop))
    del df_merged['Unnamed: 0']
    print(len(df_merged))
    return df_merged

frames.append(combine_sub('full_framework-graph-identifier'))
frames.append(combine_sub('full_framework-identifier'))
frames.append(combine_sub('full_framework-graph-node-identifier'))

df =  reduce(lambda  left,right: pd.merge(left,right,on=['cas'],
                                            how='outer'), frames)
df['cas'] = df['cas'].astype(str)
df['cas.rn'] = df['cas'].str[:-3] + '-' + df['cas'].str[-3:-1] + '-' + df['cas'].str[-1] 
print(df.head())
print(len(df))
print(df.info())

cols = ['boiling_point', 'densities', 'molar_volumes', 
        'enthalpies_of_vaporization', 'vapor_pressures', 'mass_intrinsic_solubilities', 
        'molar_intrinsic_solubilities', 'flash_points', 'hydrogen_bond_acceptors_counts', 
        'hydrogen_donor_acceptor_sums', 'freely_rotatable_bonds_counts', 'hydrogen_donor_counts', 
        'polar_surface_areas', 'experimental_boiling_point', 'experimental_melting_point', 'experimental_density', 
        'experimental_optical_rotatory_power', 'experimental_median_lethal_dose', 'cas.rn', 
        'cas.index.name', 'molecular.formula', 'molecular.weight', 'ROMol', 'SMILES']



full_w_smiles = pd.read_csv("/Users/sadrachpierre/Desktop/read_sdf/experimental_predicted_CAS_withSMILES.csv")

full_w_smiles = full_w_smiles[cols]
full_w_smiles['predicted-boiling-point'] = full_w_smiles['boiling_point']
del full_w_smiles['boiling_point']

full_w_smiles['predicted-densities'] = full_w_smiles['densities']
del full_w_smiles['densities']

full_w_smiles['predicted-molar-volumes'] = full_w_smiles['molar_volumes']
del full_w_smiles['molar_volumes']

full_w_smiles['predicted-enthalpies-of-vaporization'] = full_w_smiles['enthalpies_of_vaporization']
del full_w_smiles['enthalpies_of_vaporization']

full_w_smiles['predicted-vapor-pressures'] = full_w_smiles['vapor_pressures']
del full_w_smiles['vapor_pressures']


full_w_smiles['predicted-mass-intrinsic-solubilities'] = full_w_smiles['mass_intrinsic_solubilities']
del full_w_smiles['mass_intrinsic_solubilities']

full_w_smiles['predicted-molar-intrinsic-solubilities'] = full_w_smiles['molar_intrinsic_solubilities']
del full_w_smiles['molar_intrinsic_solubilities']

full_w_smiles['predicted-flash-points'] = full_w_smiles['flash_points']
del full_w_smiles['flash_points']


full_w_smiles['predicted-hydrogen-bond-acceptors-counts'] = full_w_smiles['hydrogen_bond_acceptors_counts']
del full_w_smiles['hydrogen_bond_acceptors_counts']


full_w_smiles['predicted-hydrogen-donor-acceptor-sums'] = full_w_smiles['hydrogen_donor_acceptor_sums']
del full_w_smiles['hydrogen_donor_acceptor_sums']


full_w_smiles['predicted-freely-rotatable-bonds-counts'] = full_w_smiles['freely_rotatable_bonds_counts']
del full_w_smiles['freely_rotatable_bonds_counts']


full_w_smiles['predicted-hydrogen-donor-counts'] = full_w_smiles['hydrogen_donor_counts']
del full_w_smiles['hydrogen_donor_counts']

full_w_smiles['predicted-polar-surface-areas'] = full_w_smiles['polar_surface_areas']
del full_w_smiles['polar_surface_areas']


full_w_smiles['experimental-boiling-point'] = full_w_smiles['experimental_boiling_point']
del full_w_smiles['experimental_boiling_point']

full_w_smiles['experimental-melting-point'] = full_w_smiles['experimental_melting_point']
del full_w_smiles['experimental_melting_point']

full_w_smiles['experimental-density'] = full_w_smiles['experimental_density']
del full_w_smiles['experimental_density']

full_w_smiles['experimental-optical-rotatory-power'] = full_w_smiles['experimental_optical_rotatory_power']
del full_w_smiles['experimental_optical_rotatory_power']

full_w_smiles['experimental-median-lethal-dose'] = full_w_smiles['experimental_median_lethal_dose']
del full_w_smiles['experimental_median_lethal_dose']

full_w_smiles['cas-index-name'] = full_w_smiles['cas.index.name']
del full_w_smiles['cas.index.name']

full_w_smiles['molecular-formula'] = full_w_smiles['molecular.formula']
del full_w_smiles['molecular.formula']

full_w_smiles['molecular-weight'] = full_w_smiles['molecular.weight']
del full_w_smiles['molecular.weight']



#cols = ['boiling_point', 'densities', 'molar_volumes', 
#        'enthalpies_of_vaporization', 'vapor_pressures', 'mass_intrinsic_solubilities', 
#        'molar_intrinsic_solubilities', 'flash_points', 'hydrogen_bond_acceptors_counts', 
#        'hydrogen_donor_acceptor_sums', 'freely_rotatable_bonds_counts', 'hydrogen_donor_counts', 
#        'polar_surface_areas', 'experimental_boiling_point', 'experimental_melting_point', 'experimental_density', 
#        'experimental_optical_rotatory_power', 'experimental_median_lethal_dose', 'cas.rn', 
#        'cas.index.name', 'molecular.formula', 'molecular.weight', 'ROMol', 'SMILES']



final_frames = [df, full_w_smiles]
final =  reduce(lambda  left,right: pd.merge(left,right,on=['cas.rn'],
                                            how='outer'), final_frames)

del final['cas']
print(final.info())

final.to_csv("full_cas_data.csv")
