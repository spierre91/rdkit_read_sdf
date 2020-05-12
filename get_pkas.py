import xml.etree.ElementTree as et 
import pandas as pd
import os
import glob
import numpy as np
from functools import reduce

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


path = '/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/'
extension = 'xml'
os.chdir(path)
#result = glob.glob('*.{}'.format(extension))#[:100]
#result = glob.glob('*.{}'.format(extension))[:10001]
#result = glob.glob('*.{}'.format(extension))[10001:20001]
#result = glob.glob('*.{}'.format(extension))[20001:30001]
#result = glob.glob('*.{}'.format(extension))[30001:40001]
result = glob.glob('*.{}'.format(extension))[40001:]



cas_list_full = []
for i in result:
    i = i[:-15]
    cas_list_full.append(i)
cas_list_full = set(cas_list_full)  


val_list = []
temp_list = []
acid_list = []
pka_type_list = []
print("hi", len(cas_list_full))
#predicted-logDs; predicted-Kocs; predicted-bioconcentration-factors;
#predicted-mass-solubilities; predicted-molar-solubilities; 
prop = 'predicted-pKas'
for i in result:
    
    xtree = et.parse(i)
    xroot = xtree.getroot()   
    i = i[:-15]            
    root = xroot.find('properties/predicted-properties/{}'.format(prop))
    try:
        for child in root:
            for grandchild in child:
                if 'standard-single-value' in str(grandchild):
                        value = grandchild.text 
                        #print((i, value))
                        val_list.append((i, value))
                if 'pKa-type' in str(grandchild):
                    #for gchild in grandchild:
                            pka_type = grandchild.text
                            pka_type_list.append((i, '{} '.format(prop) + pka_type))
                

    except(TypeError):
        val_list.append((i, 'nan'))
        pka_type_list.append((i, '{} '.format(prop) + 'unkown'))




conditions_list = []

total = [(i[0], i[1], j[1]) for i, j in zip(val_list, pka_type_list)]
print(val_list)
print(pka_type_list)
#print(total)

cas_list = []
for i in total:
    cas_list.append(i[0])
cas_list = set(cas_list)    
for i in total:
    conditions_list.append(i[2])
conditions_list = set(conditions_list)    



print(len(total))

nlist = []
count = 0
for i in conditions_list:
    for j in total:
        for k in cas_list: 
            if i == j[-1] and k == j[0]:
                nlist.append({i:j[1],'cas': j[0]})


df_list = []
print("len(nlist):", len(nlist))

frames = []       
data = pd.DataFrame.from_dict(nlist)



frames.append(data[['cas','{} Most Basic'.format(prop)]].dropna())
frames.append(data[['cas','{} Most Acidic'.format(prop)]].dropna())
frames.append(data[['cas','{} unkown'.format(prop)]].dropna())
start = frames[0]



df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['cas'],
                                            how='outer'), frames)


del df_merged['predicted-pKas unkown']
df_merged.drop_duplicates('cas', inplace=True)
print(df_merged.head())   
print(df_merged.info())
print(len(df_merged)) 
df_merged.to_csv("predicted-pKas/predicted-pKas_43k.csv".format(prop))
