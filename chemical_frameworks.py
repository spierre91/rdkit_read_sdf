import xml.etree.ElementTree as et 
import pandas as pd
import os
import glob
import numpy as np
from functools import reduce

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


path = '/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Substance/'
extension = 'xml'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))#[:100]
#result = glob.glob('*.{}'.format(extension))[:10001]
#result = glob.glob('*.{}'.format(extension))[10001:20001]
#result = glob.glob('*.{}'.format(extension))[20001:30001]
#result = glob.glob('*.{}'.format(extension))[30001:40001]
#result = glob.glob('*.{}'.format(extension))[40001:]



cas_list_full = []
for i in result:
    i = i[:-15]
    cas_list_full.append(i)
cas_list_full = set(cas_list_full)  


val_list = []
framework = []
cas = []
print("hi", len(cas_list_full))
#framework-identifier; framework-graph-node-identifier; framework-graph-identifier
prop = 'framework-graph-identifier'
for i in result:
    
    xtree = et.parse(i)
    xroot = xtree.getroot()   
    i = i[len('substance')+1:]      
    i = i[:-4]        
    root = xroot.find('structure/single-component-structure/single-component-connection-table/framework/{}'.format(prop))    
    try:
        val_list.append((i, root.text))
    except(AttributeError):
        val_list.append((i, np.nan))


for i in val_list:
    cas.append(i[0])
    
for i in val_list:
    framework.append(i[1])
    
df = pd.DataFrame({'cas':cas, prop:framework})
df.to_csv("frameworks/full_{}.csv".format(prop))
print(len(df))
