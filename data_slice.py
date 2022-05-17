import pandas as pd
from rdkit import Chem

def get_canonical_smile(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile))
df=pd.read_csv('HT2A.csv',sep=';')
smiles=list(df[df['pChEMBL Value']>7]['Smiles'])
file=open('data/receptor_5HT2A.txt','a')
for smile in smiles:
	smile=get_canonical_smile(smile)
	file.write(smile+'\n')
	
