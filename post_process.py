import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *
from rdkit import Chem


def get_data():
    f = open('generated_smiles.txt')
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    f.close()
    return lines


def smiles_in_db(smile):
    smile = '!'+get_canonical_smile(smile)+' '
    if smile in data:
        return True
    return False

def valid_smile(smile):
    return Chem.MolFromSmiles(smile) is not None

def get_canonical_smile(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile))

data,vocabs=load_data()
new_compounds=get_data()

file=open('results/non_redundant_smiles.txt','a')
non_redundant=[]
count=0
for smile in new_compounds:
#    print(smile)
    if smiles_in_db(smile):
        non_redundant.append(smile)
        count+=1
        file.write(smile+'\n')
#        print(smile)
print(count)