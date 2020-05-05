import xml.etree.ElementTree as et 
import pandas as pd
import os
import glob



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


path = '/Users/sadrachpierre/Desktop/analyze_cas/xml_files/COVID-Substance-Property-Content-Set/Properties/'
extension = 'xml'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))[:100]

val_list = []
temp_list = []
acid_list = []

#predicted-logDs; predicted-Koc; predicted-bioconcentration-factors;
#predicted-mass-solubilities; predicted-molar-solubilities; 
prop = 'predicted-logDs'
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
                if 'predicted-property-temperature-condition' in str(grandchild):
                    for gchild in grandchild:
                        if 'display-value' in str(gchild):
                            temp = gchild.text
                            temp_list.append((i, '{} '.format(prop) + temp + ' '))
                if 'predicted-property-pH-condition' in str(grandchild):
                    for gchild in grandchild:
                        if 'standard-single-value' in str(gchild):
                            acid = gchild.text
                            acid_list.append((i, 'pH ' + acid))                        

    except(TypeError):
        print(None)

conditions_list = []

total = [(i[0], i[1], j[1] + k[1]) for i, j, k in zip(val_list, temp_list, acid_list)]

cas_list = []
for i in total:
    cas_list.append(i[0])
cas_list = set(cas_list)    
for i in total:
    conditions_list.append(i[2])
conditions_list = set(conditions_list)    


print(total)
nlist = []
for i in conditions_list:
    for j in total:
        for k in cas_list: 
            if i == j[-1] and k == j[0]:
                nlist.append({i:j[1],'cas': j[0]})
            
df_list = []


            
data = pd.DataFrame(nlist)

    
print(data)
