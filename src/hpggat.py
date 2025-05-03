import dgl.function as fn
from torch import nn
import torch
import torch.nn.functional as F
import copy
import pandas as pd
from src.mol2graph import *

component_list = ["component1","component2","component3","component4","component5","component6"]
smiles_col_list = ["smiles1","smiles2","smiles3","smiles4","smiles5","smiles6"]
Mw_list = ["Mw1","Mw2","Mw3","Mw4","Mw5","Mw6"]
type_list = ["type1","type2","type3","type4","type5","type6"]

def zero_pad_lists(input_list, target_length=6):
    padded_list = input_list + [0] * (target_length - len(input_list))
    return padded_list
    
def slash_to_list(ratio_list, mode="float"):
    ratio_list = str(ratio_list).split("/")

    if mode == "float":
        return [float(i) for i in ratio_list]
    else:
        return [i for i in ratio_list]
        
def normalize_ratio(s):

    tot = sum(s)
    ratio_list = [float(i/tot) for i in s]

    return ratio_list
    
def calc_wt_ratio_from_mol_wt(single_record):
    mol_wt_list = [single_record.Mw1,single_record.Mw2,single_record.Mw3,
                   single_record.Mw4,single_record.Mw5,single_record.Mw6] 
    # if molar ratio is recorded, calculate weight ratio
    if not isNaN(single_record["mol_ratio"]):
        mol_list = str(single_record["mol_ratio"]).split("/")
        mol_list = [float(i) for i in mol_list]
        wt_list = normalize_ratio([float(i*j) for i, j in zip(mol_list, mol_wt_list)])
    elif not isNaN(single_record["wt_ratio"]):
        wt_list = str(single_record["wt_ratio"]).split("/")
        wt_list = normalize_ratio([float(i) for i in wt_list])
    else:
        wt_list = normalize_ratio([float(i) for i in mol_wt_list if not isNaN(i)])
    # single component
    if len(pd.DataFrame(mol_wt_list).dropna()) == 1:
        wt_list = [1]
    ratio_list = zero_pad_lists(wt_list)
    return ratio_list

def data_from_df(processed_df, polymerhiG_cache):
    data_list = list()
    val_data_list = list()

    for _idf in range(len(processed_df)):
        _df = processed_df.iloc[_idf]
        ratio = torch.tensor(calc_wt_ratio_from_mol_wt(_df))
        if isNaN(sum(ratio)):
            print(_idf)
            break
        
        s_list = []
        for ss in smiles_col_list:
            smiles = _df[ss]
            if isNaN(smiles) or smiles == "X":
                smiles = "C"
            if smiles.startswith("polymer_"):
                id = _df[ss].split("polymer_")[1]
                s_list.append(polymerhiG_cache[id])
            else:
                s_list.append(mon2hig_mk2(smiles))
    
        if _df.tags == 1:
            data_list.append(Data(c1=s_list[0], c2=s_list[1], c3=s_list[2],
                                  c4=s_list[3], c5=s_list[4], c6=s_list[5],
                                  wt_ratio=ratio, temp=_df.Temperature, y = np.log10(_df.Conductivity)))
        else:
            val_data_list.append(Data(c1=s_list[0], c2=s_list[1], c3=s_list[2],
                                  c4=s_list[3], c5=s_list[4], c6=s_list[5],
                                  wt_ratio=ratio, temp=_df.Temperature, y = np.log10(_df.Conductivity)))
    return data_list, val_data_list

class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, edge_feats, num_heads, activation=None):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats

        self.W_node = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.W_edge = nn.Linear(edge_feats, num_heads, bias=False)
        self.attention_src = nn.Parameter(torch.Tensor(num_heads, out_feats))
        self.attention_dst = nn.Parameter(torch.Tensor(num_heads, out_feats))

        self.activation = activation
        self._init_weights()

    def edge_attention(self, edges):
        src_feat = edges.src['h_trans'].view(-1, self.num_heads, self.out_feats)  
        dst_feat = edges.dst['h_trans'].view(-1, self.num_heads, self.out_feats)
        edge_feat = edges.data['e_trans'].view(-1, self.num_heads, 1)  

        # Compute attention scores
        attn = (
            torch.sum(src_feat * self.attention_src, dim=-1) +
            torch.sum(dst_feat * self.attention_dst, dim=-1) +
            edge_feat.squeeze(-1)
        )
        return {'e_attn': F.leaky_relu(attn)}

    def message_func(self, edges):
        weighted_msg = edges.src['h_trans'].view(-1, self.num_heads, self.out_feats) * edges.data['alpha'].unsqueeze(-1) 
        return {'m': weighted_msg}

    def reduce_func(self, nodes):
        h_new = torch.sum(nodes.mailbox['m'], dim=1)  # (N, num_heads, out_feats)
        return {'h_new': h_new.mean(dim=1)}#h_new.view(-1, self.num_heads * self.out_feats)}

    def forward(self, g, node_features):
        edge_features = g.edata["e"].reshape(-1,1)
        g.ndata['h_trans'] = self.W_node(node_features)  
        g.edata['e_trans'] = self.W_edge(edge_features) 

        g.apply_edges(self.edge_attention)

        g.edata['alpha'] = dgl.nn.functional.edge_softmax(g, g.edata['e_attn'])

        g.update_all(self.message_func, self.reduce_func)

        h_final = g.ndata['h_new']
        if self.activation is not None:
            h_final = self.activation(h_final)

        return h_final

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attention_src)
        nn.init.xavier_uniform_(self.attention_dst)

class GATNet(nn.Module):
    def __init__(self, pred_dim=1, hidden_mode=False):
        super().__init__()
        self.pred_dim = pred_dim
        self.depth = 6
        self.dims = [49,]+[128,]*self.depth
        self.activation = F.leaky_relu
        self.hidden_mode = hidden_mode
        self.GAT_list_1 = nn.ModuleList([GAT(self.dims[i], self.dims[i+1], edge_feats=1, num_heads=8, activation=self.activation) for i in range(len(self.dims)-1)])
        self.linear_g1 = nn.Linear(128,64)
        self.dropout = nn.Dropout(0)
        self.linear_ratio = nn.Linear(6,6)
        self.linear_temp = nn.Linear(1,16)
        self.linear1 = nn.Linear(64*6+6+16, 512)
        # self.batchNorm1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, pred_dim)
        self.dropout_pred = nn.Dropout(0.2)
        
        self.linear_ratio.apply(self._init_weights)
        self.linear_temp.apply(self._init_weights)
        self.linear1.apply(self._init_weights)
        self.linear2.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
            
    def forward(self, inputs):
        [g1, g2, g3, g4, g5, g6, ratio,temp] = inputs
        h1 = g1.ndata['h']
        h2 = g2.ndata['h']
        h3 = g3.ndata['h']
        h4 = g4.ndata['h']
        h5 = g5.ndata['h']
        h6 = g6.ndata['h']
        for conv1 in self.GAT_list_1:
            h1 = self.dropout(conv1(g1, h1))
            h2 = self.dropout(conv1(g2, h2))
            h3 = self.dropout(conv1(g3, h3))
            h4 = self.dropout(conv1(g4, h4))
            h5 = self.dropout(conv1(g5, h5))
            h6 = self.dropout(conv1(g6, h6))
        g1.ndata['h'] = h1
        g2.ndata['h'] = h2
        g3.ndata['h'] = h3
        g4.ndata['h'] = h4
        g5.ndata['h'] = h5
        g6.ndata['h'] = h6
        hg1 = self.dropout(self.linear_g1(dgl.sum_nodes(g1, 'h'))) #.unsqueeze(1)
        hg2 = self.dropout(self.linear_g1(dgl.sum_nodes(g2, 'h'))) #.unsqueeze(1)
        hg3 = self.dropout(self.linear_g1(dgl.sum_nodes(g3, 'h'))) #.unsqueeze(1)
        hg4 = self.dropout(self.linear_g1(dgl.sum_nodes(g4, 'h'))) #.unsqueeze(1)
        hg5 = self.dropout(self.linear_g1(dgl.sum_nodes(g5, 'h'))) #.unsqueeze(1)
        hg6 = self.dropout(self.linear_g1(dgl.sum_nodes(g6, 'h'))) #.unsqueeze(1)
        hg = torch.cat([hg1,hg2,hg3,hg4,hg5,hg6,self.activation(self.linear_ratio(ratio)),self.activation(self.linear_temp(temp))],axis=1)
        hidden = self.activation(self.linear1(hg))
        # output hidden layer
        if self.hidden_mode:
            return hidden
        out = self.dropout_pred(self.linear2(hidden))
        return out



def collate(sample):
    c1 = dgl.batch([s.c1 for s in sample])
    c2 = dgl.batch([s.c2 for s in sample])
    c3 = dgl.batch([s.c3 for s in sample])
    c4 = dgl.batch([s.c4 for s in sample])
    c5 = dgl.batch([s.c5 for s in sample])
    c6 = dgl.batch([s.c6 for s in sample])
    # labels = torch.tensor([label for _, label in samples])
    temp = torch.tensor([(s.temp-33.923080)/40.216135 for s in sample], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor([(s.y-(-3.981499))/1.836202 for s in sample], dtype=torch.float32).unsqueeze(1) # Normalize(Standardize) y labels.
    ratio = torch.cat([s.wt_ratio.reshape(1,-1) for s in sample],axis=0).float()
    return [c1,c2,c3,c4,c5,c6,ratio,temp], y