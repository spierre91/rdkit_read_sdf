import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
df = pd.read_csv("modeling_data.csv")

df.dropna(inplace =  True)


test_list = list(set(df['Molecular Formula']))#[:50] 

import re

split_val = [re.findall(r"[^\W\d_]+|\d+", i ) for i in test_list]
flat_list = [item for sublist in split_val for item in sublist]
n_l = []
for i in flat_list:
    try:
        n_l.append(int(i))
    except(ValueError):
        n_l.append(i)
        
final_list = [x for x in n_l if not isinstance(x, int)]
#print(list(set(final_list)))



element_features = ['K', 'Br', 'P', 'O', 'Si', 'F', 'S', 'Cl', 'I', 'C', 'H', 'N', 'Na']

#print(df.head())
df['Nitrogen'] = np.where(df['Molecular Formula'].str.contains('N'), 1, 0)
df['Oxygen'] = np.where(df['Molecular Formula'].str.contains('O'), 1, 0)
df['Sodium'] = np.where(df['Molecular Formula'].str.contains('Na'), 1, 0)
df['Hydrogen'] = np.where(df['Molecular Formula'].str.contains('H'), 1, 0)
df['Carbon'] = np.where(df['Molecular Formula'].str.contains('C'), 1, 0)
df['Iodine'] = np.where(df['Molecular Formula'].str.contains('I'), 1, 0)
df['Chlorine'] = np.where(df['Molecular Formula'].str.contains('Cl'), 1, 0)
df['Sulfur'] = np.where(df['Molecular Formula'].str.contains('S'), 1, 0)
df['Fluorine'] = np.where(df['Molecular Formula'].str.contains('F'), 1, 0)
df['Silicon'] = np.where(df['Molecular Formula'].str.contains('Si'), 1, 0)
df['Phosphorus'] = np.where(df['Molecular Formula'].str.contains('P'), 1, 0)
df['Bromine'] = np.where(df['Molecular Formula'].str.contains('Br'), 1, 0)
df['Potassium'] = np.where(df['Molecular Formula'].str.contains('K'), 1, 0)
df['c(=O)'] = np.where(df['Smiles'].str.contains('c(=O)'), 1, 0)
df['c2ccccc2'] = np.where(df['Smiles'].str.contains('c2ccccc2'), 1, 0)
df['C@H'] = np.where(df['Smiles'].str.contains('C@H'), 1, 0)
df['nH'] = np.where(df['Smiles'].str.contains('nH'), 1, 0)
df['[O-]'] = np.where(df['Smiles'].str.contains('[O-]'), 1, 0)
df['[N+]'] = np.where(df['Smiles'].str.contains('[N+]'), 1, 0)
df['CCCc1nc2c'] = np.where(df['Smiles'].str.contains('CCCc1nc2c'), 1, 0)
df['SCc2ccc'] = np.where(df['Smiles'].str.contains('SCc2ccc'), 1, 0)
df['SC'] = np.where(df['Smiles'].str.contains('SC'), 1, 0)
df['(=O)'] = np.where(df['Smiles'].str.contains('(=O)'), 1, 0)
df['c2c'] = np.where(df['Smiles'].str.contains('c2c'), 1, 0)

df['ccc'] = np.where(df['Smiles'].str.contains('ccc'), 1, 0)
df['cccc'] = np.where(df['Smiles'].str.contains('cccc'), 1, 0) 
df['c1c'] = np.where(df['Smiles'].str.contains('c1c'), 1, 0)
df['c1cc'] = np.where(df['Smiles'].str.contains('c1cc'), 1, 0)
df['c1ccc'] = np.where(df['Smiles'].str.contains('c1ccc'), 1, 0)    
df['r'] = np.where(df['Smiles'].str.contains('r'), 1, 0)
df['1'] = np.where(df['Smiles'].str.contains('1'), 1, 0) 
df['P'] = np.where(df['Smiles'].str.contains('P'), 1, 0)
df['n'] = np.where(df['Smiles'].str.contains('n'), 1, 0)
df['\('] = np.where(df['Smiles'].str.contains('\('), 1, 0)
df['F'] = np.where(df['Smiles'].str.contains('F'), 1, 0)
df['I'] = np.where(df['Smiles'].str.contains('I'), 1, 0)
df['c'] = np.where(df['Smiles'].str.contains('c'), 1, 0)
df['C'] = np.where(df['Smiles'].str.contains('C'), 1, 0)
df['o'] = np.where(df['Smiles'].str.contains('o'), 1, 0)
df['-'] = np.where(df['Smiles'].str.contains('-'), 1, 0)
df['4'] = np.where(df['Smiles'].str.contains('4'), 1, 0)
df['l'] = np.where(df['Smiles'].str.contains('l'), 1, 0)
df['\]'] = np.where(df['Smiles'].str.contains('\]'), 1, 0)
df['.'] = np.where(df['Smiles'].str.contains('.'), 1, 0)
df['H'] = np.where(df['Smiles'].str.contains('H'), 1, 0)
df['N'] = np.where(df['Smiles'].str.contains('N'), 1, 0)
df['@'] = np.where(df['Smiles'].str.contains('@'), 1, 0)
df['\+'] = np.where(df['Smiles'].str.contains('\+'), 1, 0)
df['\)'] = np.where(df['Smiles'].str.contains('\)'), 1, 0)
df['\['] = np.where(df['Smiles'].str.contains('\['), 1, 0)
df['O'] = np.where(df['Smiles'].str.contains('O'), 1, 0)
df['B'] = np.where(df['Smiles'].str.contains('B'), 1, 0)
df['5'] = np.where(df['Smiles'].str.contains('5'), 1, 0)
df['s'] = np.where(df['Smiles'].str.contains('s'), 1, 0)
df['a'] = np.where(df['Smiles'].str.contains('a'), 1, 0)
df['#'] = np.where(df['Smiles'].str.contains('#'), 1, 0)
df['='] = np.where(df['Smiles'].str.contains('='), 1, 0)
df['S'] = np.where(df['Smiles'].str.contains('S'), 1, 0)
df['2'] = np.where(df['Smiles'].str.contains('2'), 1, 0)
df['3'] = np.where(df['Smiles'].str.contains('3'), 1, 0)

print(df.head())
df_1 = df[df['Grade'] == 1.0]
df_2 = df[df['Grade'] == 2.0]
df_3 = df[df['Grade'] == 3.0]
df_4 = df[df['Grade'] == 4.0]
df_5 = df[df['Grade'] == 5.0]
df_6 = df[df['Grade'] == 6.0]
df_7 = df[df['Grade'] == 7.0]
df_8 = df[df['Grade'] == 8.0]
df_9 = df[df['Grade'] == 9.0]
df_10 = df[df['Grade'] == 10.0]


#def long_substr(data):
#  substrs = lambda x: {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}
#  s = substrs(data[0])
#  for val in data[1:]:
#    s.intersection_update(substrs(val))
#  return max(s, key=len)

#print(long_substr(list(df_1['Smiles'])))
#print(long_substr(list(df_2['Smiles'])))
#print(long_substr(list(df_3['Smiles'])))
#print(long_substr(list(df_4['Smiles'])))
#print(long_substr(list(df_5['Smiles'])))
#print(long_substr(list(df_6['Smiles'])))
#print(long_substr(list(df_7['Smiles'])))
#print(long_substr(list(df_8['Smiles'])))
#print(long_substr(list(df_9['Smiles'])))
#print(long_substr(list(df_10['Smiles'])))



def print_confusion_matrix():
    val = []
    features = ['Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 
                'Molecular Weight', 'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', 
             '#Rotatable Bonds', 'QED Weighted', 'CX LogP',
             'CX LogD', 'Aromatic Rings', 'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', 
             '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)','Nitrogen', 
             'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 
             'Silicon', 'Phosphorus', 'Bromine', 'Potassium', '[O-]', '[N+]', 'nH', 'C@H', 'CCCc1nc2c','SCc2ccc', 
             'SC', '(=O)', 'c2c', 'ccc', 'cccc','c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\]', '.', 
           'H', 'N', '@', '\+', '\)', '\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3']  
    X = np.array(df[features])      
    y = np.array(df['Grade'])
    for k in range(1, 50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = k, test_size = 0.2)
                        
        model =  RandomForestClassifier(n_estimators = 600, max_depth =600, random_state = 42)
        model.fit(X_train, y_train)                
        #print(dict(zip(features, model.feature_importances_)))        
        y_pred = model.predict(X_test)
        #print("Accurracy: ", f1_score(y_test, y_pred, average = None))            
        conmat = confusion_matrix(y_test, y_pred)    
        val.append(np.mat(conmat))
   
    return np.array(val)

    
v = print_confusion_matrix()

def get_plot(figsize = (10,7), fontsize=14):
    classnames = list(set(df['Grade']))
    result =  np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
    for i,j in zip(v,v):
        for l in range(len(i)):
            for m in range(len(i)):
                result[l][m] += i[l][m] 
    df_cm = pd.DataFrame(
        result, index=classnames, columns=classnames, 
    )
     
    df_cm = df_cm / df_cm.sum(axis=1)    
    print(result/len(v))
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()    
get_plot()
