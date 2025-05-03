import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, InMemoryDataset
from rdkit import Chem
from src.smiles_utils import *
from dgl import DGLGraph
import dgl
import networkx as nx

def bond_features(bond):
    bt = bond.GetBondType()
    # 결합 유형에 따라 숫자 할당
    bond_order = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 1.5  # 방향족 결합을 1.5로 표현
    }

    numeric_bond_type = bond_order[bt]
    
    # 수치화된 결합 유형 및 반지 여부를 텐서로 반환
    return numeric_bond_type
    
def isNaN(num):
    return num != num
    
def mol2dgl_single(smi):
    if isNaN(smi):
        smi = "C"
    # else:
    n_nodes = 0
    n_edges = 0
    bond_x = []
    mol = Chem.MolFromSmiles(smi)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    g = DGLGraph()
    # g=dgl.graph()
    nodeF = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        nodeF.append(np.concatenate([atom_symbol_HNums(atom),
                                            atom_degree(atom),
                                            atom_Aroma(atom),
                                            atom_Hybrid(atom),
                                            atom_ring(atom),
                                            atom_FC(atom)],
                                           axis=0))
    g.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        features = bond_features(bond)

        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
    g.add_edges(bond_src, bond_dst)
    g.ndata['h'] = torch.Tensor(nodeF)
    g.edata["e"] = torch.Tensor(bond_x)
    return g
    
def pol2hig_mk2(polG):
    pg = dgl.from_networkx(polG.to_networkx(), edge_attrs=["degree"])
    src_list, dst_list = [], []
    nodeF = []
    edgeF = []
    
    src_h, dst_h = pg.edges()
    src_list.append(src_h)
    dst_list.append(dst_h)
    
    total_num_node = pg.num_nodes()
    nodeF.append(torch.ones((total_num_node, 49)))
    # edgeF.append(np.log10(pg.edata["degree"])+10)
    edgeF.append(pg.edata["degree"])
                 
    for i, node in enumerate(polG.nodes):
        name = node.fragment.name
        smiles = node.fragment.smiles.replace("[R]","[Mg:1]").replace("[Q]","[Mg:2]").replace("[T]","[Mg:3]").replace("[U]","[Mg:4]")
        g = mol2dgl_single(smiles)
        src, dst = g.edges()
        node = g.nodes() + total_num_node

        src_list.append(src + total_num_node)
        dst_list.append(dst + total_num_node)
        
        nodeF.append(g.ndata["h"])
        edgeF.append(g.edata["e"])
        
        src_list.append(node)
        dst_list.append(torch.tensor([i] * g.num_nodes()))
        
        edgeF.append(torch.tensor([1] * g.num_nodes()))
        total_num_node += g.num_nodes()
    nodeF = torch.cat(nodeF)
    final_src = torch.cat(src_list)
    final_dst = torch.cat(dst_list)
    final_graph = dgl.graph((final_src, final_dst), num_nodes=total_num_node)
    final_graph.ndata['h'] = torch.Tensor(nodeF)
    final_graph.edata["e"] = torch.cat(edgeF)
    return final_graph
    
def mon2hig_mk2(smiles):
    mg = dgl.from_networkx(nx.MultiGraph())
    mg.add_nodes(1)
    src_list, dst_list, nodeF, edgeF = [], [], [], []
    total_num_node = 1
    nodeF.append(torch.ones((total_num_node, 49)))

    g = mol2dgl_single(smiles)
    
    src, dst = g.edges()
    node = g.nodes() + total_num_node
    
    src_list.append(src + total_num_node)
    dst_list.append(dst + total_num_node)
    
    nodeF.append(g.ndata["h"])
    edgeF.append(g.edata["e"])
    
    src_list.append(node)
    dst_list.append(torch.tensor([0] * g.num_nodes()))

    edgeF.append(torch.tensor([1] * g.num_nodes()))
    total_num_node += g.num_nodes()
    nodeF = torch.cat(nodeF)
    final_src = torch.cat(src_list)
    final_dst = torch.cat(dst_list)
    final_graph = dgl.graph((final_src, final_dst), num_nodes=total_num_node)
    final_graph.ndata['h'] = torch.Tensor(nodeF)
    final_graph.edata["e"] = torch.cat(edgeF)

    return final_graph        