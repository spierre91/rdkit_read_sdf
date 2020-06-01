import pandas as pd 
import numpy as np 
import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv("chembl_cas_combined.csv")
print(len(set(df['chembl_id'])))
test = df[df['chembl_id'].isin(['191', '1014', '41194', 
         '304639', '305544','429847', '1951143' ])]

train = df[~df['chembl_id'].isin(['191', '1014', '41194', 
         '304639', '305544','429847', '1951143' ])]





df.drop_duplicates(inplace = True, subset='chembl_id', keep="last")



#df_chembl = pd.read_csv("/Users/sadrachpierre/Desktop/chembl-data/modeling_data.csv")
#df_chembl_match = pd.read_csv("Chembl-match-via-Inchi-kc.csv")

#df = pd.merge(final, df_chembl, on = 'smiles')

#print(df_chembl.head())
#print(df_chembl_match.head())
#print(len(set(df_chembl['Smiles'])))
#print(len(set(df_chembl_match['chembl_id'])))

#df['chembl_id'] = df['chembl_id'].str.lstrip('CHEMBL')


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
            'Grade', 'chembl_id', 'Assay_cat'] 

df = df[cols]
df['predicted-boiling-point'].fillna(df['predicted-boiling-point'].mean(), inplace = True)
df['predicted-enthalpies-of-vaporization'].fillna(df['predicted-enthalpies-of-vaporization'].mean(), inplace = True)


#df['predicted-enthalpies-of-vaporization'] = np.log(df['predicted-enthalpies-of-vaporization'])
#
#df['predicted-boiling-point'] = np.log(df['predicted-boiling-point'])
#df['predicted-molar-volumes'] = np.log(df['predicted-molar-volumes'])
#df['Molecular Weight (Monoisotopic)'] = np.log(df['Molecular Weight (Monoisotopic)'])
#df['Ligand Efficiency LE'] = np.log(df['Ligand Efficiency LE'])
#df['Ligand Efficiency LLE'] = np.log(df['Ligand Efficiency LLE'])
#df['Ligand Efficiency BEI'] = np.log(df['Ligand Efficiency BEI'])
#df['Ligand Efficiency SEI'] = np.log(df['Ligand Efficiency SEI'])
print(len(df))

#df['Grade'].fillna(2, inplace = True)
#df.dropna(how = 'any', inplace = True, subset = ['Grade'])



#df.fillna(0, inplace = True)
test = df[df['chembl_id'].isin(['191', '1014', '41194', 
         '304639', '305544','429847', '1951143' ])]

train = df[~df['chembl_id'].isin(['191', '1014', '41194', 
         '304639', '305544','429847', '1951143' ])]


print(train)



train['Ligand Efficiency LE'].fillna(train['Ligand Efficiency LE'].mean(), inplace = True)
train['Ligand Efficiency LLE'].fillna(train['Ligand Efficiency LLE'].mean(), inplace = True)
train['Ligand Efficiency BEI'].fillna(train['Ligand Efficiency BEI'].mean(), inplace = True)
train['Ligand Efficiency SEI'].fillna(train['Ligand Efficiency SEI'].mean(), inplace = True)

train['predicted-Kocs 25 °C pH 1'].fillna(train['predicted-Kocs 25 °C pH 1'].mean(), inplace = True)
train['predicted-Kocs 25 °C pH 7'].fillna(train['predicted-Kocs 25 °C pH 7'].mean(), inplace = True)
train['predicted-Kocs 25 °C pH 10'].fillna(train['predicted-Kocs 25 °C pH 10'].mean(), inplace = True)

train['predicted-bioconcentration-factors 25 °C pH 1'].fillna(train['predicted-bioconcentration-factors 25 °C pH 1'].mean(), inplace = True)
train['predicted-bioconcentration-factors 25 °C pH 7'].fillna(train['predicted-bioconcentration-factors 25 °C pH 7'].mean(), inplace = True)
train['predicted-bioconcentration-factors 25 °C pH 10'].fillna(train['predicted-bioconcentration-factors 25 °C pH 10'].mean(), inplace = True)

train['predicted-logDs 25 °C pH 1'].fillna(train['predicted-logDs 25 °C pH 1'].mean(), inplace = True)
train['predicted-logDs 25 °C pH 7'].fillna(train['predicted-logDs 25 °C pH 7'].mean(), inplace = True)
train['predicted-logDs 25 °C pH 10'].fillna(train['predicted-logDs 25 °C pH 10'].mean(), inplace = True)

train['predicted-mass-solubilities 25 °C pH 1'].fillna(train['predicted-mass-solubilities 25 °C pH 1'].mean(), inplace = True)
train['predicted-mass-solubilities 25 °C pH 7'].fillna(train['predicted-mass-solubilities 25 °C pH 7'].mean(), inplace = True)
train['predicted-mass-solubilities 25 °C pH 10'].fillna(train['predicted-mass-solubilities 25 °C pH 10'].mean(), inplace = True)

train['predicted-molar-solubilities 25 °C pH 1'].fillna(train['predicted-molar-solubilities 25 °C pH 1'].mean(), inplace = True)
train['predicted-molar-solubilities 25 °C pH 7'].fillna(train['predicted-molar-solubilities 25 °C pH 7'].mean(), inplace = True)
train['predicted-molar-solubilities 25 °C pH 10'].fillna(train['predicted-molar-solubilities 25 °C pH 10'].mean(), inplace = True)

train['predicted-pKas Most Basic'].fillna(train['predicted-pKas Most Basic'].mean(), inplace = True)
train['predicted-pKas Most Acidic'].fillna(train['predicted-pKas Most Acidic'].mean(), inplace = True)
train['predicted-densities'].fillna(train['predicted-densities'].mean(), inplace = True)
print(train.isna().sum())
train.fillna(0, inplace = True)




f1 = ['predicted-Kocs 25 °C pH 1', 'predicted-Kocs 25 °C pH 7', 'predicted-Kocs 25 °C pH 10', 
            'predicted-logDs 25 °C pH 1', 'predicted-logDs 25 °C pH 7', 'predicted-logDs 25 °C pH 10', 
            'predicted-bioconcentration-factors 25 °C pH 1', 'predicted-bioconcentration-factors 25 °C pH 7', 'predicted-bioconcentration-factors 25 °C pH 10',
            'predicted-mass-solubilities 25 °C pH 1', 'predicted-mass-solubilities 25 °C pH 7', 'predicted-mass-solubilities 25 °C pH 10',
            'predicted-molar-solubilities 25 °C pH 1', 'predicted-molar-solubilities 25 °C pH 7', 'predicted-molar-solubilities 25 °C pH 10', 
            'predicted-pKas Most Basic', 'predicted-pKas Most Acidic', 
            'predicted-boiling-point', 'predicted-enthalpies-of-vaporization', 'predicted-densities', 
            'predicted-molar-volumes',  'predicted-mass-intrinsic-solubilities', 
            'predicted-molar-intrinsic-solubilities', 'hydrogen-bond-acceptors-counts', 'hydrogen-donor-acceptor-sums', 
            'freely-rotatable-bonds-counts', 'hydrogen-donor-counts', 'polar-surface-areas',
            'Molecular Weight', 
            'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'CX LogP', 'CX LogD', 'Aromatic Rings',
            'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)',
            'Nitrogen', 'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus', 
            'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
            'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
            '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3', 'Assay_cat'] 
#df['Ligand Efficiency BEI'].fillna(df['Ligand Efficiency BEI'].mean(), inplace = True)
#
#df['Ligand Efficiency LLE'].fillna(df['Ligand Efficiency LLE'].mean(), inplace = True)
#df['Ligand Efficiency SEI'].fillna(df['Ligand Efficiency SEI'].mean(), inplace = True)



pred_le = RandomForestRegressor(random_state= 42)
X_le = np.array(train[f1])
y_le = np.array(train['Ligand Efficiency LE'])
pred_le.fit(X_le, y_le)

test['Ligand Efficiency LE'] = pred_le.predict(test[f1])


pred_lle = RandomForestRegressor(random_state= 42)
X_lle = np.array(train[f1])
y_lle = np.array(train['Ligand Efficiency LLE'])
pred_lle.fit(X_lle, y_lle)
test['Ligand Efficiency LLE'] = pred_lle.predict(test[f1])



pred_bei = RandomForestRegressor(random_state= 42)
X_bei = np.array(train[f1])
y_bei = np.array(train['Ligand Efficiency BEI'])
pred_bei.fit(X_bei, y_bei)
test['Ligand Efficiency BEI'] = pred_bei.predict(test[f1])


pred_sei = RandomForestRegressor(random_state= 42)
X_sei = np.array(train[f1])
y_sei = np.array(train['Ligand Efficiency SEI'])
pred_sei.fit(X_sei, y_sei)
test['Ligand Efficiency SEI'] = pred_sei.predict(test[f1])



print(len(test))

#train = train[train['Grade'].isin([2,6,7])]
print(test)


#test.fillna(0, inplace=True)
print(len(test))
print(len(train))
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
            'Molecular Weight', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency BEI', 'Ligand Efficiency SEI',
            'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'CX LogP', 'CX LogD', 'Aromatic Rings',
            'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)',          
            'Nitrogen', 'Oxygen', 'Sodium', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 'Silicon', 'Phosphorus', 
            'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2', 'C@H', 'nH', '[O-]', '[N+]', 'CCCc1nc2c', 'SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 
            'cccc', 'c1c', 'c1cc', 'c1ccc', 'r', '1', 'P', 'n', '\\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\\]', '.', 'H', 'N', '@', 
            '\\+', '\\)', '\\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3', ] 






X_train = np.array(train[features])      
y_train = np.array(train['Grade'])

X_test = np.array(test[features])      
y_test = np.array(test['Grade'])
#1500: pred [ 8  6  8  5  5 10]  true [ 9 10  8  6  4  9] 
#1000: pred [ 8  7  8  5  5 10]; true [ 9 10  8  6  4  9]  
#900: pred [ 8  7  8  5  5 10]; true [ 9 10  8  6  4  9]  
#800: pred [ 8  7  8  5  5 10]; true [ 9 10  8  6  4  9]       
#700: pred [ 8  7  8  5  5 10]; true [ 9 10  8  6  4  9]  
#600:pred [ 8  7  8  5  6 10]; true [ 9 10  8  6  4  9]  
       
#model = xgb.XGBClassifier(random_state = 42)
model =xgb.XGBClassifier(random_state = 1, objective = 'multi:softmax', n_estimators = 700, max_depth =700)#RandomForestClassifier(n_estimators = 15, max_depth =15, random_state = 42)#xgb.XGBClassifier(random_state = 1, objective = 'multi:softmax', n_estimators = 15)#xgb.XGBClassifier(random_state = 42)#RandomForestClassifier(n_estimators = 600, max_depth =600, random_state = 42)#xgb.XGBClassifier(learning_rate = 0.1, n_estimators =300, random_state = 1, objective = 'multi:softmax')   #RandomForestClassifier(n_estimators = 1200, max_depth =600, random_state = 42)
model.fit(X_train, y_train)                
print(sorted(dict(zip(features, model.feature_importances_))))    
val = [] 
for i in sorted(dict(zip(features, model.feature_importances_))):
    val.append(dict(zip(features, model.feature_importances_))[i])
    
print(dict(zip(sorted(dict(zip(features, model.feature_importances_))), val)))
y_pred = model.predict(X_test)

print(y_pred)
print(y_test)
from collections import Counter 
print(Counter(y_train))



import matplotlib.pyplot as plt 


men_means = y_pred
women_means = y_test

N = 6
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, men_means, width, label='Predicted')
plt.bar(ind + width, women_means, width,
    label='True Value')

plt.ylabel('Activity Score')
plt.xlabel('CheMBL Number')
plt.title('Predicted vs True Values')

plt.xticks(ind + width / 2, list(test['chembl_id']))
plt.legend(loc='best')
plt.show()
