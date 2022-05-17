from rdkit import Chem
from numpy import random
import numpy as np
'''
loading chembl into lines of smiles
'''
def load_chembl():
#    f = open('data/receptor_5HT2A.txt')
    f = open('data/chembl_smiles.txt')
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    f.close()
    return lines

def load_chembl_focused():
    f = open('data/receptor_5HT2A.txt')
#    f = open('data/chembl_smiles.txt')
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    f.close()
    return lines


chembl = load_chembl()
data_set = []
dictionary = {}
'''
create a character dictionary
'''
focused_chembl = load_chembl_focused()
focused_data_set = []
focused_dictionary = {}

def added_to_dictionary(smile):
    for char in smile:
        if char not in dictionary:
            dictionary[char] = True
def added_to_focused_dictionary(smile):
    for char in smile:
        if char not in focused_dictionary:
            focused_dictionary[char] = True


print('processing smiles !!!')

'''
! is the start token and ? is the end token
'''###This part works for all dataset 
for i in range(len(chembl)):
    if len(chembl[i])<=100:
        smile = '!'+chembl[i]+' '
        data_set.append(smile)
        added_to_dictionary(smile)

# count=0
for i in range(len(focused_chembl)):
    added_to_focused_dictionary(focused_chembl[i])
    char_len=len(focused_chembl[i])
    char_count=0
    for char in focused_chembl[i]:
        if char in dictionary:
            char_count+=1
    if char_count==char_len:
        smile = '!'+focused_chembl[i]+' '
        focused_data_set.append(smile)
#   count+=1
print(len(focused_data_set))

print('saving smiles')
vocabs = [ele for ele in dictionary]
#vocabs = [ele for ele in focused_dictionary]
print(vocabs)
print(len(vocabs))

'''
save the smiles string and vocabs
'''
np.savez_compressed('data/smiles_data.npz', data_set=np.array(data_set, dtype=object),vocabs=np.array(vocabs), dtype=object)
#np.savez_compressed('data/focused_smiles.npz', data_set=np.array(focused_data_set, dtype=object),vocabs=np.array(vocabs), dtype=object)

for ele in focused_dictionary:
    if ele not in dictionary:
        print(ele+'Not exists')