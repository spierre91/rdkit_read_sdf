from rdkit.Chem import PandasTools, SDMolSupplier, MolToSmiles, MolFromSmiles
import os
from rdkit import RDConfig
import pandas as pd
import numpy as np 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sdf_path = '/Users/sadrachpierre/Desktop/read_sdf/new_antiviral_with_properties.sdf'
sdfFile = os.path.join(RDConfig.RDDataDir,'/Users/sadrachpierre/Desktop/read_sdf/new_antiviral_with_properties.sdf')

'''
frames = PandasTools.LoadSDF(sdfFile,smilesName='SMILES',molColName='Molecule',
           includeFingerprints=False)
'''
smiles_list = []
cas_rn = []
cas_index_name = []
molecular_formula =[]
molecular_weight = []
boiling_point_predicted =[]

density_predicted =[]
pka_predicted =[]
melting_point_experimental = []
boiling_point_experimental = []
density_experimental = []
ROMol = []
image_list =[]
           
data = SDMolSupplier(sdf_path, sanitize=False)
for molecule in data:
    smiles = MolToSmiles(molecule)
    smiles_list.append(smiles)
    image = MolFromSmiles(smiles)
    image_list.append(image)
    cas_rn.append(molecule.GetProp('cas.rn'))
    cas_index_name.append(molecule.GetProp('cas.index.name'))
    molecular_formula.append(molecule.GetProp('molecular.formula'))
    try:
        molecular_weight.append(molecule.GetProp('molecular.weight'))
    except(KeyError):
        molecular_weight.append(np.nan)
    try:
        boiling_point_predicted.append(molecule.GetProp('boiling.point.predicted'))
    except(KeyError):
        boiling_point_predicted.append(np.nan)

    try:
        density_predicted.append(molecule.GetProp('density.predicted'))
    except(KeyError):
        density_predicted.append(np.nan)
    try:
        pka_predicted.append(molecule.GetProp('pka.predicted'))
    except(KeyError):
        pka_predicted.append(np.nan)
    try:
        melting_point_experimental.append(molecule.GetProp('melting.point.experimental'))
    except(KeyError):
        melting_point_experimental.append(np.nan)
    try:
        boiling_point_experimental.append(molecule.GetProp('boiling.point.experimental'))
    except(KeyError):
        boiling_point_experimental.append(np.nan)
    try:
        density_experimental.append(molecule.GetProp('density.experimental'))
    except(KeyError):
        density_experimental.append(np.nan)
    



df = pd.DataFrame({'cas.rn':cas_rn, 'cas.index.name':cas_index_name,'molecular.formula':molecular_formula,
                   'molecular.weight':molecular_weight, 'boiling.point.predicted':boiling_point_predicted,
                   'density.predicted': density_predicted, 'pka.predicted':pka_predicted, 
                   'melting.point.experimental':melting_point_experimental, 
                   'boiling.point.experimental':boiling_point_experimental, 'density.experimental': density_experimental,
                    'ROMol':image_list, 'SMILES':smiles_list})


#df.to_csv("full_cas_data.csv")
print(df.head())  
print(len(df))
