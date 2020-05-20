import pandas as pd 

df_activity = pd.read_csv("CHEMBL26-chembl_26_activity.csv")


df_labels = pd.read_csv("chembl-ic50-kc.csv")


df_properties = pd.read_csv("CHEMBL26-chembl_26_molecule.csv")


df_properties['ChEMBL_ID'] = df_properties['ChEMBL ID']
df_activity['ChEMBL_ID'] = df_activity['Molecule ChEMBL ID']

df = pd.merge(df_properties, df_labels, on = 'ChEMBL_ID')

df['ms_cat'] = df['Molecular Species'].astype('category')
df['ms_cat'] = df['ms_cat'].cat.codes
df['Molecular Species'].dropna(inplace = True)
print(df.head())

df = pd.merge(df, df_activity, on = 'ChEMBL_ID')
print(df.head())
print(len(df))





import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df['AlogP'] = df['AlogP_y']
df['#RO5 Violations'] = df['#RO5 Violations_y']
df['Molecular Weight'] = df['Molecular Weight_y']  
df['Smiles'] = df['Smiles_y']
df = df[['Molecular Formula', 'Smiles', 'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 
          'Molecular Weight', 'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', 
         '#Rotatable Bonds', 'QED Weighted', 'CX LogP',
         'CX LogD', 'Aromatic Rings', 'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', 
         '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)', 'Grade']]
         
#df.to_csv("modeling_data.csv")
print(df.info())         

df.dropna(inplace =  True)

print(df.head())
print(df.corr())

import seaborn as sns
sns.heatmap(df.corr())
from collections import Counter

print(Counter(df['Grade']))
features = ['Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI', 
            'Molecular Weight', 'AlogP', 'PSA', 'HBA', 'HBD', '#RO5 Violations', 
         '#Rotatable Bonds', 'QED Weighted', 'CX LogP',
         'CX LogD', 'Aromatic Rings', 'Heavy Atoms', 'HBA Lipinski', 'HBD Lipinski', 
         '#RO5 Violations (Lipinski)', 'Molecular Weight (Monoisotopic)']
         
X = np.array(df[features])      
y = np.array(df['Grade'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.5)

print(Counter(y_train))


model =  RandomForestClassifier(n_estimators = 300, max_depth =300, random_state = 42)
model.fit(X_train, y_train)




y_pred = model.predict(X_test)
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


