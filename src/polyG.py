from rdkit import Chem
from rdkit.Chem import Descriptors
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
import copy

global fragment_cache,polymerG_cache

fragment_cache = {}
polymerG_cache = {}

class Fragment:
    def __init__(self, name, smiles):
        self.name = name
        self.smiles = smiles

    def __repr__(self):
        return f"Fragment(name={self.name}, smiles={self.smiles})"

    @staticmethod
    def get_fragment(name, smiles):
        # Use the smiles string as the key for caching
        if smiles not in fragment_cache:
            fragment_cache[smiles] = Fragment(name, smiles)
        return fragment_cache[smiles]

class Node: # Block or Random
    def __init__(self, fragment, connections, name=None):
        self.fragment = copy.copy(fragment)
        self.connections = connections
        if name:
            self.fragment.name = name
            
    def __repr__(self):
        return f"Node(fragment={self.fragment}, connections={self.connections})"

class PolymerGraph:
    def __init__(self, name, nodes, structure=None, mol_ratio=True, PDI="?",MW=None):
        self.name = name
        self.nodes = nodes
        self.smiles = self.generate_smiles()
        self.graph_string = self.generate_graph_string()
        self.structure = structure
        self.PDI = PDI
        if not mol_ratio:
            structure = 'rand'
            pass # I have to code the wt_ratio to mol_ratio
        if MW:
            self.MW = MW
        else:
            self.MW = mass_polyG(self)
        polymerG_cache[self.name]=self
            
    def generate_smiles(self):
        smiles = []
        for node in self.nodes:
            fragment_smiles = node.fragment.smiles
            smiles.append(fragment_smiles)
        return '.'.join(smiles)

    def generate_graph_string(self):
        graph_string = []
        node_indices = {node.fragment.name: idx + 1 for idx, node in enumerate(self.nodes)}
        for node in self.nodes:
            idx = node_indices[node.fragment.name]
            # graph_string.append(f"<[@{idx}];{node.fragment.structure};[@{idx}]")
            for conn_name, (conn_value, conn_time) in node.connections.items():
                conn_idx = node_indices[conn_value.split('.')[0]]
                graph_string.append(f"<[@{idx}];{node.fragment.smiles};[@{idx}].{conn_name} -> [@{conn_idx}].{conn_value.split('.')[1]}:{str(conn_time)}>")
        return ''.join(graph_string)
        
    def to_networkx(self):
        G = nx.MultiDiGraph()
        node_indices = {node.fragment.name: idx + 1 for idx, node in enumerate(self.nodes)}
        for node in self.nodes:
            idx = node_indices[node.fragment.name]
            G.add_node(idx, label=node.fragment.name, smiles=node.fragment.smiles)
            for conn_name, (conn_value,degree) in node.connections.items():
                conn_parts = conn_value.split('.')
                conn_fragment_name = conn_parts[0]
                conn_dst = conn_parts[1]
                if degree == "?":
                    degree = 1
                if conn_fragment_name in node_indices:
                    conn_idx = node_indices[conn_fragment_name]
                    G.add_edge(idx, conn_idx, degree=float(degree), src=conn_name, dst=conn_dst)
                else:
                    G.add_edge(idx, idx, degree=float(degree), src=conn_name, dst=conn_dst)  # self-loop
        return G

    def __repr__(self):
        return f"PolymerGraph(name={self.name}, nodes={self.nodes})"


def create_polymer_graph_from_string(name, graph_string):
    nodes = {}
    entries = graph_string.strip('<>').split('><')
    
    for entry in entries:
        parts = entry.split(';')
        node_idx = parts[0][2:-1]  
        structure = parts[1] 

        fragment = Fragment(f"Fragment_{node_idx}", structure)
         
        connection_info = parts[2]
        conn_detail, degree = connection_info.split(':')
        src_conn_point, dest = conn_detail.split(' -> ')
        dest_idx, dest_conn_point = dest.split('.')

        dest_idx = dest_idx[2:-1]  # Remove brackets from [@2]

        if node_idx not in nodes:
            nodes[node_idx] = Node(fragment, {})

        nodes[node_idx].connections[dest_conn_point] = (f"Fragment_{dest_idx}.{dest_conn_point}", int(degree))

    return PolymerGraph(name, list(nodes.values()))


def draw_polyG(polyG):
    fig, ax = plt.subplots()
    net = polyG.to_networkx()
    pos = nx.spring_layout(net)
    nx.draw_networkx_nodes(net, pos)
    
    for i, (u, v, data) in enumerate(net.edges(data=True)):
        rad = 0.3 * (i - 1)
        arrow = FancyArrowPatch(posA=pos[u], posB=pos[v],
                                connectionstyle=f'arc3,rad={rad}',
                                arrowstyle='-|>')
        ax.add_patch(arrow)

    for u, v, data in net.edges(data=True):
        if u == v:
            x, y = pos[u]
            loop_radius = 0.3  # self-loop 크기 조절
            loop = Arc((x, y), loop_radius, loop_radius, angle=0, theta1=0, theta2=300)
            ax.add_patch(loop)

    nx.draw_networkx_labels(net, pos)
    ax.autoscale()
    
    plt.show()

def mass_polyG(polyG):
    
    G = polyG.to_networkx()
    
    total_mass = 0
    for node, data in G.nodes(data=True):
        smiles = data['smiles'].replace("R","H").replace("Q","H").replace("T","H").replace("U","H")
        print(smiles)
        node_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)) 
        print(node_mass)
        self_loop = 1
        for _, target, edge_data in G.edges(node, data=True):
            if target == node:
                if edge_data["degree"] == "?":
                    ed = 1
                else:
                    ed = edge_data["degree"]
                self_loop = ed
                print("self loop", target, ed)
        total_mass += node_mass*self_loop
    if polyG.PDI == "?":
        PDI = 1
    else:
        PDI = polyG.PDI
    print("PDI is",polyG.PDI)
    total_mass *= PDI
    print("MW is",total_mass)
    return total_mass