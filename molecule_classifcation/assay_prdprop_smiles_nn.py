import seaborn as sns
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
df = pd.read_csv("modeling_data.csv")
s = ''


df.dropna(inplace =  True)
s = s.join(list(df['Smiles']))
s = list(set([i for i in s]))
print(s)
#['r', '1', 'P', 'n', '(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', ']', '.', 'H', 'N', '@', '+', ')', '[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3']

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




print(df.head())

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
print(list(set(final_list)))

element_features = ['K', 'Br', 'P', 'O', 'Si', 'F', 'S', 'Cl', 'I', 'C', 'H', 'N', 'Na']

print(df.head())
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
print(df_7.head())
#print(list(df_7['Smiles']))
#print("------------------------")
#print(list(df_10['Smiles']))

def long_substr(data):
  substrs = lambda x: {x[i:i+j] for i in range(len(x)) for j in range(len(x) - i + 1)}
  s = substrs(data[0])
  for val in data[1:]:
    s.intersection_update(substrs(val))
  return max(s, key=len)

print(long_substr(list(df_1['Smiles'])))
print(long_substr(list(df_2['Smiles'])))
print(long_substr(list(df_3['Smiles'])))
print(long_substr(list(df_4['Smiles'])))
print(long_substr(list(df_5['Smiles'])))
print(long_substr(list(df_6['Smiles'])))
print(long_substr(list(df_7['Smiles'])))
print(long_substr(list(df_8['Smiles'])))
print(long_substr(list(df_9['Smiles'])))
print(long_substr(list(df_10['Smiles'])))



features = ['Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 
            'Molecular Weight', 'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', 
         '#Rotatable Bonds', 'QED Weighted', 'CX LogP',
         'CX LogD', 'Aromatic Rings', 'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', 
         '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)','Nitrogen', 
         'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 
         'Silicon', 'Phosphorus', 'Bromine', 'Potassium', 'c2ccccc2', 'c(=O)',
         '[O-]', '[N+]', 'nH', 'C@H', 'CCCc1nc2c','SCc2ccc', 'SC', '(=O)', 'c2c', 'ccc', 'cccc','c1c', 'c1cc', 'c1ccc',
         'r', '1', 'P', 'n', '\(', 'F', 'I', 'c', 'C', 'o', '-', '4', 'l', '\]', '.', 
           'H', 'N', '@', '\+', '\)', '\[', 'O', 'B', '5', 's', 'a', '#', '=', 'S', '2', '3']


#features = ['Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 
#            'Molecular Weight', 'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', 
#         '#Rotatable Bonds', 'QED Weighted', 'CX LogP',
#         'CX LogD', 'Aromatic Rings', 'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', 
#         '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)','Nitrogen', 
#         'Oxygen', 'Sodium', 'Hydrogen', 'Carbon', 'Iodine', 'Chlorine', 'Sulfur', 'Fluorine', 
#         'Silicon', 'Phosphorus', 'Bromine', 'Potassium', 'c(=O)', 'c2ccccc2',]
       
X = np.array(df[features])      
y = np.array(df['Grade'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.5)




#model =  RandomForestClassifier(n_estimators = 600, max_depth =600, random_state = 42)
#model.fit(X_train, y_train)

model = Sequential()



model.add(Dense(64, activation = 'relu', input_shape = (80, )))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(11, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_crossentropy'])

model.fit(X_train, y_train, batch_size=1, epochs=500, validation_split = 0.5)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
#print(dict(zip(features, model.feature_importances_)))

#y_pred = model.predict(X_test)
from sklearn.metrics import f1_score
print("Accurracy: ", f1_score(y_test, y_pred, average = None))
import matplotlib.pyplot as plt 

classnames = list(set(df['Grade']))
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    df_cm = df_cm / df_cm.sum(axis=1)

    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


print_confusion_matrix(conmat, classnames)
