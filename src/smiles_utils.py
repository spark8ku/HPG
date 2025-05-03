import json
from collections import OrderedDict
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.linalg import block_diag
import numpy as np

def _convertToAdj(smiles_list):
    adj = [Chem.rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(i),useBO=True)+np.eye(len(Chem.MolFromSmiles(i).GetAtoms())) if i !='gas' else np.zeros([10,10]) for i in smiles_list]
    adj = block_diag(*adj)
    return adj
    
def _convertToFeatures(smiles_list):
    features = [np.concatenate([np.concatenate([atom_symbol_HNums(atom),atom_degree(atom),atom_Aroma(atom),atom_Hybrid(atom),atom_ring(atom),atom_FC(atom)],axis=0).reshape(1,-1) for atom in Chem.MolFromSmiles(i).GetAtoms()],axis=0) if i !='gas' else np.zeros([10,43]) for i in smiles_list]
    EZ_features = [EZ_stereo(smiles) if smiles !='gas' else np.zeros([10,2]) for smiles in smiles_list]
    n_f = block_diag(*[np.ones([len(Chem.MolFromSmiles(i).GetAtoms())]) if i !='gas' else np.ones([10]) for i in smiles_list])
    features = np.concatenate(features,axis=0)
    EZ_features = np.concatenate(EZ_features,axis=0)
    features = np.concatenate([features,EZ_features],axis=1)
    return features, n_f

def atom_symbol_HNums(atom):
    
    return np.array(one_of_k_encoding(atom.GetSymbol(),
                                      ['C', 'N', 'O','S', 'H', 'F', 'Cl', 'Br', 'I','Se','Te','Si','P','B','Ca','Mg','Al','Sb','Ge','As'])+
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))


def atom_degree(atom):
    return np.array(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5 ,6])).astype(int) 


def atom_Aroma(atom):
    return np.array([atom.GetIsAromatic()]).astype(int)


def atom_Hybrid(atom):
    return np.array(one_of_k_encoding(str(atom.GetHybridization()),['S','SP','SP2','SP3','SP3D','SP3D2'])).astype(int)


def atom_ring(atom):
    return np.array([atom.IsInRing()]).astype(int)


def atom_FC(atom):
    return np.array(one_of_k_encoding(atom.GetFormalCharge(), [-4,-3,-2,-1, 0, 1, 2, 3, 4])).astype(int)


def EZ_stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Adj = Chem.rdmolops.GetAdjacencyMatrix(mol,useBO=True)
    double_idx = np.where(np.triu(Adj,0)==2)
    stereo_mat = np.zeros([Adj.shape[0],2])
    for i in range(len(double_idx[0])):
        i_atom = int(double_idx[0][i])
        f_atom = int(double_idx[1][i])
        stereo = mol.GetBondBetweenAtoms(i_atom,f_atom).GetStereo()
        if stereo == Chem.BondStereo.STEREOE:
            stereo_mat[i_atom,0] = stereo_mat[f_atom,0] = 1
        elif stereo == Chem.BondStereo.STEREOZ:
            stereo_mat[i_atom,1] = stereo_mat[f_atom,1] = 1
    return stereo_mat


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def mol_smiles(smi):
    return Chem.MolFromSmiles(smi)

def canon_smiles(smi):
    if smi == 'gas':
        return smi
    elif Chem.MolFromSmiles(smi) == None:
        return ''
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)
