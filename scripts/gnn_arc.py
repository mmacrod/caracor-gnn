"""
gnn_arc.py

Graph Neural Network architecture used in CARACOR-GNN.

Defines `GNNmodel`, a GATv2-based network for graph-level classification of atrial
cardiomyopathy tasks (density or location).
"""

import torch
from torch_geometric.nn import  GATv2Conv
from torch_geometric.nn import global_mean_pool, PairNorm


class GNNmodel(torch.nn.Module):
    def __init__(self,args):
        super(GNNmodel, self).__init__()


        ######################
        ##### GNN Layers #####
        gnn_layers = list()
        batch_norm_layers = list()
 
        ### First GNN
        gnn_layers.append(
            GATv2Conv(200, args.gnn_units*2,concat=False,heads=args.gnn_heads,bias=False)) 
        batch_norm_layers.append(PairNorm())
    
        for _ in range(1, args.gnn_layers):
            gnn_layers.append(
                GATv2Conv(args.gnn_units*2, args.gnn_units*2,concat = False,heads=args.gnn_heads,bias=False))
            batch_norm_layers.append(PairNorm())
            
        ### Second GNN
        gnn_layers.append(
            GATv2Conv(args.gnn_units*2, args.gnn_units,concat =False,heads=args.gnn_heads,bias=False))
        batch_norm_layers.append(PairNorm())
        
        
        for _ in range(1, max(args.gnn_layers,2)):
            gnn_layers.append(
                GATv2Conv(args.gnn_units, args.gnn_units, concat = False,heads=args.gnn_heads,bias=False))
            batch_norm_layers.append(PairNorm())
            
        ## Clasificacion
        gnn_layers.append(
            GATv2Conv(args.gnn_units,args.num_classes,concat =False,heads=args.gnn_heads,bias=False))
            
            
        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.batch_norm_layers = torch.nn.ModuleList(batch_norm_layers)
        self.dropout = torch.nn.Dropout(p=args.dropout)
        

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_gnn = x
        
        ######################################
        ##### GNN layers
        for gnn, batch_norm in zip(self.gnn_layers[:-1], self.batch_norm_layers):
            x_gnn = self.dropout(x_gnn)
            x_gnn = gnn(x_gnn, edge_index)
            x_gnn = batch_norm(x_gnn)
            x_gnn = x_gnn.relu()

        gnn = self.gnn_layers[-1]
        x_gnn = self.dropout(x_gnn)
        x_gnn = gnn(x_gnn, edge_index) 
        x_gnn = x_gnn.relu()

        ############################
        ##### Pooling layer
        out = global_mean_pool(x_gnn, batch)
        
        return out
