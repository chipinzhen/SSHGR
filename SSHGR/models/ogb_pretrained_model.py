import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros


from rdkit.Chem import AllChem
from rdkit import Chem
from torch_geometric.data import Data
import numpy as np


num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__(aggr=aggr)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index[0].size(1), ), dtype=dtype,
                                     device=edge_index[0].device)
        row, col = edge_index[0]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__(aggr=aggr)

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)



class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        print(gnn_type)
        print('----------------------------------------------------')
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        # print(x.shape)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # print(h.shape)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))

class Gin(torch.nn.Module):
    def __init__(self, args, gin_gnn, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super(Gin, self).__init__()

                # allowable node and edge features
        self.allowable_features = {
            'possible_atomic_num_list' : list(range(1, 119)),
            'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            'possible_chirality_list' : [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_hybridization_list' : [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
            ],
            'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
            'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'possible_bonds' : [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'possible_bond_dirs' : [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        }

        self.args = args
        self.gin_gnn = gin_gnn
        self.drop_ratio = drop_ratio
        self.JK = JK

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((len(gin_gnn.gnns) + 1) * gin_gnn.x_embedding2.embedding_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(gin_gnn.x_embedding2.embedding_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((len(gin_gnn.gnns) + 1) * gin_gnn.x_embedding2.embedding_dim, set2set_iter)
            else:
                self.pool = Set2Set(gin_gnn.x_embedding2.embedding_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, features, features_batch=None):
        smiles_list, datalist = self.transform(features)
        node_representation_list = []
        for data in datalist:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            if self.args.cuda:
                x = x.cuda()
                edge_index = edge_index.cuda()
                edge_attr = edge_attr.cuda()

            node_representation = self.gin_gnn(x, edge_index, edge_attr)
            
            if self.args.cuda:
                batch = torch.zeros(x.shape[0]).cuda().long()
            else:
                batch = torch.zeros(x.shape[0]).long()
                
            node_representation_list.append(self.pool(node_representation, batch)[0])

        node_representation_list = torch.stack(node_representation_list, dim=0)
        return node_representation_list


    def transform(self, smiles_list):


        datalist = []
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs_list[i]
            data = self.mol_to_graph_data_obj_simple(rdkit_mol)
            # data.id = torch.tensor(smiles2idx[smiles_list[i]])
            datalist.append(data)
            # print ('i', i)
            if data.x.shape[0] == 1:
                print(smiles_list[i])
        return smiles_list, datalist

    def mol_to_graph_data_obj_simple(self, mol):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. NB: Uses simplified atom and bond features, and represent
        as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        num_atom_features = 2   # atom type,  chirality tag
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [self.allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [self.allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2   # bond type, bond direction
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [self.allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [self.allowable_features[
                                                'possible_bond_dirs'].index(
                    bond.GetBondDir())]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list),
                                     dtype=torch.long)
        else:   # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data



class GCN(torch.nn.Module):
    def __init__(self, args, gcn_gnn, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super(GCN, self).__init__()

                # allowable node and edge features
        self.allowable_features = {
            'possible_atomic_num_list' : list(range(1, 119)),
            'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            'possible_chirality_list' : [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_hybridization_list' : [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
            ],
            'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
            'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'possible_bonds' : [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'possible_bond_dirs' : [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        }

        self.args = args
        self.gcn_gnn = gcn_gnn
        self.drop_ratio = drop_ratio
        self.JK = JK

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((len(gcn_gnn.gnns) + 1) * gcn_gnn.x_embedding2.embedding_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(gcn_gnn.x_embedding2.embedding_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((len(gcn_gnn.gnns) + 1) * gcn_gnn.x_embedding2.embedding_dim, set2set_iter)
            else:
                self.pool = Set2Set(gcn_gnn.x_embedding2.embedding_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, features, features_batch=None):
        smiles_list, datalist = self.transform(features)
        node_representation_list = []
        for data in datalist:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            if self.args.cuda:
                x = x.cuda()
                edge_index = edge_index.cuda()
                edge_attr = edge_attr.cuda()

            node_representation = self.gcn_gnn(x, edge_index, edge_attr)
            
            if self.args.cuda:
                batch = torch.zeros(x.shape[0]).cuda().long()
            else:
                batch = torch.zeros(x.shape[0]).long()
                
            node_representation_list.append(self.pool(node_representation, batch)[0])

        node_representation_list = torch.stack(node_representation_list, dim=0)
        return node_representation_list


    def transform(self, smiles_list):


        datalist = []
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs_list[i]
            data = self.mol_to_graph_data_obj_simple(rdkit_mol)
            # data.id = torch.tensor(smiles2idx[smiles_list[i]])
            datalist.append(data)
            # print ('i', i)
            if data.x.shape[0] == 1:
                print(smiles_list[i])
        return smiles_list, datalist

    def mol_to_graph_data_obj_simple(self, mol):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. NB: Uses simplified atom and bond features, and represent
        as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        num_atom_features = 2   # atom type,  chirality tag
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [self.allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [self.allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2   # bond type, bond direction
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [self.allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [self.allowable_features[
                                                'possible_bond_dirs'].index(
                    bond.GetBondDir())]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list),
                                     dtype=torch.long)
        else:   # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data



class GraphSAGE(torch.nn.Module):
    def __init__(self, args, gs_gnn, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super(GraphSAGE, self).__init__()

                # allowable node and edge features
        self.allowable_features = {
            'possible_atomic_num_list' : list(range(1, 119)),
            'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            'possible_chirality_list' : [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_hybridization_list' : [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
            ],
            'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
            'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'possible_bonds' : [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'possible_bond_dirs' : [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        }

        self.args = args
        self.gs_gnn = gs_gnn
        self.drop_ratio = drop_ratio
        self.JK = JK

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((len(gs_gnn.gnns) + 1) * gs_gnn.x_embedding2.embedding_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(gs_gnn.x_embedding2.embedding_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((len(gs_gnn.gnns) + 1) * gs_gnn.x_embedding2.embedding_dim, set2set_iter)
            else:
                self.pool = Set2Set(gs_gnn.x_embedding2.embedding_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, features, features_batch=None):

        smiles_list, datalist = self.transform(features)
        node_representation_list = []
        for data in datalist:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            if self.args.cuda:
                x = x.cuda()
                edge_index = edge_index.cuda()
                edge_attr = edge_attr.cuda()

            node_representation = self.gs_gnn(x, edge_index, edge_attr)
            
            if self.args.cuda:
                batch = torch.zeros(x.shape[0]).cuda().long()
            else:
                batch = torch.zeros(x.shape[0]).long()
                
            node_representation_list.append(self.pool(node_representation, batch)[0])

        node_representation_list = torch.stack(node_representation_list, dim=0)
        return node_representation_list


    def transform(self, smiles_list):


        datalist = []
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs_list[i]
            data = self.mol_to_graph_data_obj_simple(rdkit_mol)
            # data.id = torch.tensor(smiles2idx[smiles_list[i]])
            datalist.append(data)
            # print ('i', i)
            if data.x.shape[0] == 1:
                print(smiles_list[i])
        return smiles_list, datalist

    def mol_to_graph_data_obj_simple(self, mol):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. NB: Uses simplified atom and bond features, and represent
        as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        num_atom_features = 2   # atom type,  chirality tag
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [self.allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [self.allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2   # bond type, bond direction
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [self.allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [self.allowable_features[
                                                'possible_bond_dirs'].index(
                    bond.GetBondDir())]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list),
                                     dtype=torch.long)
        else:   # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data


class Gat(torch.nn.Module):
    def __init__(self, args, gat_gnn, JK = "last", drop_ratio = 0, graph_pooling = "mean"):
        super(Gat, self).__init__()

                # allowable node and edge features
        self.allowable_features = {
            'possible_atomic_num_list' : list(range(1, 119)),
            'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            'possible_chirality_list' : [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_hybridization_list' : [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
            ],
            'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
            'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'possible_bonds' : [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC
            ],
            'possible_bond_dirs' : [ # only for double bond stereo information
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        }

        self.args = args
        self.gat_gnn = gat_gnn
        self.drop_ratio = drop_ratio
        self.JK = JK

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((len(gat_gnn.gnns) + 1) * gat_gnn.x_embedding2.embedding_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(gat_gnn.x_embedding2.embedding_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((len(gat_gnn.gnns) + 1) * gat_gnn.x_embedding2.embedding_dim, set2set_iter)
            else:
                self.pool = Set2Set(gat_gnn.x_embedding2.embedding_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, features, features_batch=None):
        smiles_list, datalist = self.transform(features)
        node_representation_list = []
        for data in datalist:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            if self.args.cuda:
                x = x.cuda()
                edge_index = edge_index.cuda()
                edge_attr = edge_attr.cuda()

            node_representation = self.gat_gnn(x, edge_index, edge_attr)
            
            if self.args.cuda:
                batch = torch.zeros(x.shape[0]).cuda().long()
            else:
                batch = torch.zeros(x.shape[0]).long()
                
            node_representation_list.append(self.pool(node_representation, batch)[0])

        node_representation_list = torch.stack(node_representation_list, dim=0)
        return node_representation_list


    def transform(self, smiles_list):


        datalist = []
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs_list[i]
            data = self.mol_to_graph_data_obj_simple(rdkit_mol)
            # data.id = torch.tensor(smiles2idx[smiles_list[i]])
            datalist.append(data)
            # print ('i', i)
            if data.x.shape[0] == 1:
                print(smiles_list[i])
        return smiles_list, datalist

    def mol_to_graph_data_obj_simple(self, mol):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. NB: Uses simplified atom and bond features, and represent
        as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        num_atom_features = 2   # atom type,  chirality tag
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [self.allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [self.allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2   # bond type, bond direction
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [self.allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [self.allowable_features[
                                                'possible_bond_dirs'].index(
                    bond.GetBondDir())]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list),
                                     dtype=torch.long)
        else:   # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

if __name__ == "__main__":
    pass

