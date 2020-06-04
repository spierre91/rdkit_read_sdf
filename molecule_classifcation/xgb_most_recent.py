import pandas as pd 
import numpy as np 
import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

df = pd.read_csv("chembl_cas_combined.csv")
df.drop_duplicates(inplace = True, subset='chembl_id', keep="last")
df['n_count'] = df['smiles'].str.count('n')
print(df.head())

print(len(set(df['chembl_id'])))

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
df['c(=O)'] = np.where(df['smiles'].str.contains('c(=O)'), 1, 0)
df['c2ccccc2'] = np.where(df['smiles'].str.contains('c2ccccc2'), 1, 0)
df['C@H'] = np.where(df['smiles'].str.contains('C@H'), 1, 0)
df['nH'] = np.where(df['smiles'].str.contains('nH'), 1, 0)
df['[O-]'] = np.where(df['smiles'].str.contains('[O-]'), 1, 0)
df['[N+]'] = np.where(df['smiles'].str.contains('[N+]'), 1, 0)
df['CCCc1nc2c'] = np.where(df['smiles'].str.contains('CCCc1nc2c'), 1, 0)
df['SCc2ccc'] = np.where(df['smiles'].str.contains('SCc2ccc'), 1, 0)
df['SC'] = np.where(df['smiles'].str.contains('SC'), 1, 0)
df['(=O)'] = np.where(df['smiles'].str.contains('(=O)'), 1, 0)
df['c2c'] = np.where(df['smiles'].str.contains('c2c'), 1, 0)

df['ccc'] = np.where(df['smiles'].str.contains('ccc'), 1, 0)
df['cccc'] = np.where(df['smiles'].str.contains('cccc'), 1, 0) 
df['c1c'] = np.where(df['smiles'].str.contains('c1c'), 1, 0)
df['c1cc'] = np.where(df['smiles'].str.contains('c1cc'), 1, 0)
df['c1ccc'] = np.where(df['smiles'].str.contains('c1ccc'), 1, 0)    
df['r'] = np.where(df['smiles'].str.contains('r'), 1, 0)
df['1'] = np.where(df['smiles'].str.contains('1'), 1, 0) 
df['P'] = np.where(df['smiles'].str.contains('P'), 1, 0)
df['n'] = np.where(df['smiles'].str.contains('n'), 1, 0)
df['\('] = np.where(df['smiles'].str.contains('\('), 1, 0)
df['F'] = np.where(df['smiles'].str.contains('F'), 1, 0)
df['I'] = np.where(df['smiles'].str.contains('I'), 1, 0)
df['c'] = np.where(df['smiles'].str.contains('c'), 1, 0)
df['C'] = np.where(df['smiles'].str.contains('C'), 1, 0)
df['o'] = np.where(df['smiles'].str.contains('o'), 1, 0)
df['-'] = np.where(df['smiles'].str.contains('-'), 1, 0)
df['4'] = np.where(df['smiles'].str.contains('4'), 1, 0)
df['l'] = np.where(df['smiles'].str.contains('l'), 1, 0)
df['\]'] = np.where(df['smiles'].str.contains('\]'), 1, 0)
df['.'] = np.where(df['smiles'].str.contains('.'), 1, 0)
df['H'] = np.where(df['smiles'].str.contains('H'), 1, 0)
df['N'] = np.where(df['smiles'].str.contains('N'), 1, 0)
df['@'] = np.where(df['smiles'].str.contains('@'), 1, 0)
df['\+'] = np.where(df['smiles'].str.contains('\+'), 1, 0)
df['\)'] = np.where(df['smiles'].str.contains('\)'), 1, 0)
df['\['] = np.where(df['smiles'].str.contains('\['), 1, 0)
df['O'] = np.where(df['smiles'].str.contains('O'), 1, 0)
df['B'] = np.where(df['smiles'].str.contains('B'), 1, 0)
df['5'] = np.where(df['smiles'].str.contains('5'), 1, 0)
df['s'] = np.where(df['smiles'].str.contains('s'), 1, 0)
df['a'] = np.where(df['smiles'].str.contains('a'), 1, 0)
df['#'] = np.where(df['smiles'].str.contains('#'), 1, 0)
df['='] = np.where(df['smiles'].str.contains('='), 1, 0)
df['S'] = np.where(df['smiles'].str.contains('S'), 1, 0)
df['2'] = np.where(df['smiles'].str.contains('2'), 1, 0)
df['3'] = np.where(df['smiles'].str.contains('3'), 1, 0)
df['Assay_cat'] = df['Assay'].astype('category')
df['Assay_cat'] = df['Assay_cat'].cat.codes

df['framework-graph-identifier_cat'] = df['framework-graph-identifier'].astype('category')
df['framework-graph-identifier_cat'] = df['framework-graph-identifier_cat'].cat.codes

df['framework-graph-node-identifier_cat'] = df['framework-graph-node-identifier'].astype('category')
df['framework-graph-node-identifier_cat'] = df['framework-graph-node-identifier_cat'].cat.codes

df['framework-identifier_cat'] = df['framework-identifier'].astype('category')
df['framework-identifier_cat'] = df['framework-identifier_cat'].cat.codes
print(list(df.columns))

#'framework-graph-identifier', 'framework-graph-node-identifier', 'framework-identifier', 
#df = df[df['predicted-Kocs 25 °C pH 1']<=2000]
#df = df[df['predicted-boiling-point']>600]
#df = df[(df['Chlorine']> 0)|  (df['Bromine']> 0)| (df['Sulfur']> 0)|(df['Fluorine']> 0)|(df['Nitrogen']> 0)]
cols = ['predicted-bioconcentration-factors 25 °C pH 1', 
            'predicted-bioconcentration-factors 25 °C pH 2', 'predicted-bioconcentration-factors 25 °C pH 3', 
            'predicted-bioconcentration-factors 25 °C pH 4', 'predicted-bioconcentration-factors 25 °C pH 5', 
            'predicted-bioconcentration-factors 25 °C pH 6', 'predicted-bioconcentration-factors 25 °C pH 7', 
            'predicted-bioconcentration-factors 25 °C pH 8', 'predicted-bioconcentration-factors 25 °C pH 9', 
            'predicted-bioconcentration-factors 25 °C pH 10', 'predicted-Kocs 25 °C pH 1', 'predicted-Kocs 25 °C pH 2', 
            'predicted-Kocs 25 °C pH 3', 'predicted-Kocs 25 °C pH 4', 'predicted-Kocs 25 °C pH 5', 'predicted-Kocs 25 °C pH 6',
            'predicted-Kocs 25 °C pH 7', 'predicted-Kocs 25 °C pH 8', 'predicted-Kocs 25 °C pH 9', 'predicted-Kocs 25 °C pH 10', 
            'predicted-logDs 25 °C pH 1', 'predicted-logDs 25 °C pH 2', 'predicted-logDs 25 °C pH 3', 'predicted-logDs 25 °C pH 4', 
            'predicted-logDs 25 °C pH 5', 'predicted-logDs 25 °C pH 6', 'predicted-logDs 25 °C pH 7', 'predicted-logDs 25 °C pH 8',
            'predicted-logDs 25 °C pH 9', 'predicted-logDs 25 °C pH 10', 'predicted-mass-solubilities 25 °C pH 1', 
            'predicted-mass-solubilities 25 °C pH 2', 'predicted-mass-solubilities 25 °C pH 3', 'predicted-mass-solubilities 25 °C pH 4', 
            'predicted-mass-solubilities 25 °C pH 5', 'predicted-mass-solubilities 25 °C pH 6', 'predicted-mass-solubilities 25 °C pH 7', 
            'predicted-mass-solubilities 25 °C pH 8', 'predicted-mass-solubilities 25 °C pH 9', 'predicted-mass-solubilities 25 °C pH 10', 
            'predicted-molar-solubilities 25 °C pH 1', 'predicted-molar-solubilities 25 °C pH 2', 'predicted-molar-solubilities 25 °C pH 3', 
            'predicted-molar-solubilities 25 °C pH 4', 'predicted-molar-solubilities 25 °C pH 5', 'predicted-molar-solubilities 25 °C pH 6', 
            'predicted-molar-solubilities 25 °C pH 7', 'predicted-molar-solubilities 25 °C pH 8', 'predicted-molar-solubilities 25 °C pH 9', 
            'predicted-molar-solubilities 25 °C pH 10', 'predicted-pKas Most Basic', 'predicted-pKas Most Acidic', 
            'predicted-densities', 'predicted-molar-volumes',  'predicted-mass-intrinsic-solubilities', 
            'predicted-molar-intrinsic-solubilities', 'hydrogen-bond-acceptors-counts', 'hydrogen-donor-acceptor-sums', 
            'freely-rotatable-bonds-counts', 'hydrogen-donor-counts', 'polar-surface-areas', 'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Molecular Weight', 
            'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'CX LogP', 'CX LogD', 'Aromatic Rings',
            'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)',
            'Nitrogen', 'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus', 
            'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
            'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
            '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3', 'predicted-boiling-point', 'predicted-enthalpies-of-vaporization', 
            'Grade', 'Assay_cat', 'n_count', 'framework-graph-identifier_cat', 'framework-graph-node-identifier_cat', 'framework-identifier_cat'] 
            
#            'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Molecular Weight', 
#            'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'CX LogP', 'CX LogD', 'Aromatic Rings',
#            'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)', 'Grade', 
#            'Nitrogen', 'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus', 
#            'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
#            'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
#            '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3']

df = df[cols]
df['predicted-boiling-point'].fillna(df['predicted-boiling-point'].mean(), inplace = True)
df['predicted-enthalpies-of-vaporization'].fillna(df['predicted-enthalpies-of-vaporization'].mean(), inplace = True)
df['Ligand Efficiency BEI'].fillna(df['Ligand Efficiency BEI'].mean(), inplace = True)
df['Ligand Efficiency LE'].fillna(df['Ligand Efficiency LE'].mean(), inplace = True)
df['Ligand Efficiency LLE'].fillna(df['Ligand Efficiency LLE'].mean(), inplace = True)
df['Ligand Efficiency SEI'].fillna(df['Ligand Efficiency SEI'].mean(), inplace = True)

#df['Grade'].fillna(df['Grade'].mode(), inplace = True)
df.dropna(how = 'any', inplace = True, subset = ['Grade'])
#df['Grade'].fillna(1, inplace = True)
df.fillna(0, inplace = True)


print(df.info())
print(df.head())


def print_confusion_matrix():
    val = []
#    features =  ['predicted-bioconcentration-factors 25 °C pH 1', 
#                'predicted-bioconcentration-factors 25 °C pH 2', 'predicted-bioconcentration-factors 25 °C pH 3', 
#                'predicted-bioconcentration-factors 25 °C pH 4', 'predicted-bioconcentration-factors 25 °C pH 5', 
#                'predicted-bioconcentration-factors 25 °C pH 6', 'predicted-bioconcentration-factors 25 °C pH 7', 
#                'predicted-bioconcentration-factors 25 °C pH 8', 'predicted-bioconcentration-factors 25 °C pH 9', 
#                'predicted-bioconcentration-factors 25 °C pH 10', 'predicted-Kocs 25 °C pH 1', 'predicted-Kocs 25 °C pH 2', 
#                'predicted-Kocs 25 °C pH 3', 'predicted-Kocs 25 °C pH 4', 'predicted-Kocs 25 °C pH 5', 'predicted-Kocs 25 °C pH 6',
#                'predicted-Kocs 25 °C pH 7', 'predicted-Kocs 25 °C pH 8', 'predicted-Kocs 25 °C pH 9', 'predicted-Kocs 25 °C pH 10', 
#                'predicted-logDs 25 °C pH 1', 'predicted-logDs 25 °C pH 2', 'predicted-logDs 25 °C pH 3', 'predicted-logDs 25 °C pH 4', 
#                'predicted-logDs 25 °C pH 5', 'predicted-logDs 25 °C pH 6', 'predicted-logDs 25 °C pH 7', 'predicted-logDs 25 °C pH 8',
#                'predicted-logDs 25 °C pH 9', 'predicted-logDs 25 °C pH 10', 'predicted-mass-solubilities 25 °C pH 1', 
#                'predicted-mass-solubilities 25 °C pH 2', 'predicted-mass-solubilities 25 °C pH 3', 'predicted-mass-solubilities 25 °C pH 4', 
#                'predicted-mass-solubilities 25 °C pH 5', 'predicted-mass-solubilities 25 °C pH 6', 'predicted-mass-solubilities 25 °C pH 7', 
#                'predicted-mass-solubilities 25 °C pH 8', 'predicted-mass-solubilities 25 °C pH 9', 'predicted-mass-solubilities 25 °C pH 10', 
#                'predicted-molar-solubilities 25 °C pH 1', 'predicted-molar-solubilities 25 °C pH 2', 'predicted-molar-solubilities 25 °C pH 3', 
#                'predicted-molar-solubilities 25 °C pH 4', 'predicted-molar-solubilities 25 °C pH 5', 'predicted-molar-solubilities 25 °C pH 6', 
#                'predicted-molar-solubilities 25 °C pH 7', 'predicted-molar-solubilities 25 °C pH 8', 'predicted-molar-solubilities 25 °C pH 9', 
#                'predicted-molar-solubilities 25 °C pH 10','predicted-pKas Most Basic', 'predicted-pKas Most Acidic', 
#                    'predicted-boiling-point', 'predicted-enthalpies-of-vaporization', 'predicted-densities', 
#                    'predicted-molar-volumes',  'predicted-mass-intrinsic-solubilities', 
#                    'predicted-molar-intrinsic-solubilities', 'predicted-molar-intrinsic-solubilities', 'hydrogen-bond-acceptors-counts', 'hydrogen-donor-acceptor-sums', 
#                'freely-rotatable-bonds-counts', 'hydrogen-donor-counts', 'polar-surface-areas',
#                    'Nitrogen', 'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus', 
#                'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
#                'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
#                '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3', 'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Molecular Weight', 
#                'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'CX LogP', 'CX LogD', 'Aromatic Rings',
#                'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)',] 
                

                
    features = ['predicted-Kocs 25 °C pH 1', 'predicted-Kocs 25 °C pH 7', 'predicted-Kocs 25 °C pH 10', 
                'predicted-logDs 25 °C pH 1', 'predicted-logDs 25 °C pH 7', 'predicted-logDs 25 °C pH 10', 
                'predicted-bioconcentration-factors 25 °C pH 1', 'predicted-bioconcentration-factors 25 °C pH 7', 'predicted-bioconcentration-factors 25 °C pH 10',
                'predicted-mass-solubilities 25 °C pH 1', 'predicted-mass-solubilities 25 °C pH 7', 'predicted-mass-solubilities 25 °C pH 10',
                'predicted-molar-solubilities 25 °C pH 1', 'predicted-molar-solubilities 25 °C pH 7', 'predicted-molar-solubilities 25 °C pH 10',                 
                'predicted-pKas Most Basic', 'predicted-pKas Most Acidic', 
                'predicted-boiling-point', 'predicted-enthalpies-of-vaporization', 'predicted-densities', 
                'predicted-molar-volumes',  'predicted-mass-intrinsic-solubilities', 
                'predicted-molar-intrinsic-solubilities', 'hydrogen-bond-acceptors-counts', 'hydrogen-donor-acceptor-sums', 
                'freely-rotatable-bonds-counts', 'hydrogen-donor-counts', 'polar-surface-areas',
                'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 'Molecular Weight', 
                'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'CX LogP', 'CX LogD', 'Aromatic Rings',
                'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)',
                'Nitrogen', 'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus', 
                'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
                'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
                '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3']
    #,'framework-graph-identifier_cat', 'framework-graph-node-identifier_cat', 'framework-identifier_cat' ] 
    
    

        
    
    
#len(['Nitrogen', 'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus',
#    'Bromine', 'Potassium', '1', '\\(', '-', '4', 'l', '\\]', '.', '@', 
#    '\\+', '\\)', '\\[' , '5', '#', '=', '2', '3', ])
#            
            
#            'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
#            'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
#            '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3'] 
    X = np.array(df[features])      
    y = np.array(df['Grade'])
    
    for k in range(1, 50):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = k, test_size = 0.2)
                        
        model = xgb.XGBClassifier(random_state = 1, objective = 'multi:softmax', n_estimators = 700, max_depth =700)#xgb.XGBClassifier(random_state = 42)#
        model.fit(X_train, y_train)                
        #print(dict(zip(features, model.feature_importances_)))        
        y_pred = model.predict(X_test)
        #print("Accurracy: ", f1_score(y_test, y_pred, average = None))            
        conmat = confusion_matrix(y_test, y_pred)    
        print(len(np.mat(conmat)))
        if len(np.mat(conmat)) == 10:
            val.append(np.mat(conmat))
   
    return np.array(val)

    
v = print_confusion_matrix()

for i in v:
    print(i)


def get_plot(figsize = (10,7), fontsize=14):
    classnames = list(set(df['Grade']))
    print(classnames)
    result =  np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
    for i,j in zip(v,v):
        print(i)
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
    plt.title('Classification with CAS & Chembl data')
    plt.show()    
get_plot()

print(len(df))
