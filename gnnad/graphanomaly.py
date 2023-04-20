# -*- coding: utf-8 -*-
"""
Graph Neural Network-Based Anomaly Detection.

References
----------
[1] Deng, Ailin, and Bryan Hooi. "Graph neural network-based anomaly detection in multivariate time series."
Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 5. 2021.
[2] Buchhorn, Katie, et al. "Graph Neural Network-Based Anomaly Detection for River Network Systems"
arXiv preprint arXiv:2304.09367 (2023).
"""

import math
import os
import random
from datetime import datetime
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import iqr, rankdata
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import Linear, Parameter
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torchsummary import summary
from sklearn.metrics import confusion_matrix, accuracy_score

__author__ = ["KatieBuc"]


class TimeDataset(Dataset):
    """
    A PyTorch dataset class for time series data, to provideadditional functionality for
    processing time series data.

    Attributes
    ----------
    raw_data : list
        A list of raw data
    config : dict
        A dictionary containing the configuration of dataset
    edge_index : np.ndarray
        Edge index of the dataset
    mode : str
        The mode of dataset, either 'train' or 'test'
    x : torch.Tensor
        Feature data
    y : torch.Tensor
        Target data
    labels : torch.Tensor
        Anomaly labels of the data
    """

    def __init__(self, raw_data, edge_index, mode="train", config=None):
        self.raw_data = raw_data
        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        # to tensor
        data = torch.tensor(raw_data[:-1]).double()
        labels = torch.tensor(raw_data[-1]).double()
        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr, labels_arr = [], [], []
        slide_win, slide_stride = self.config["slide_win"], self.config["slide_stride"]
        is_train = self.mode == "train"
        total_time_len = data.shape[1]

        for i in (
            range(slide_win, total_time_len, slide_stride)
            if is_train
            else range(slide_win, total_time_len)
        ):
            ft = data[:, i - slide_win : i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()

        return feature, y, label, edge_index


class GraphLayer(MessagePassing):
    """
    Class for graph convolutional layers using message passing.

    Attributes
    ----------
    in_channels : int
        Number of input channels for the layer
    out_channels : int
        Number of output channels for the layer
    heads : int
        Number of heads for multi-head attention
    concat_heads : bool
        Whether to concatenate across heads
    negative_slope : float
        Slope for LeakyReLU
    dropout : float
        Dropout rate
    lin : nn.Module
        Linear layer for transforming input
    att_i : nn.Parameter
        Attention parameter related to x_i
    att_j : nn.Parameter
        Attention parameter related to x_j
    att_em_i : nn.Parameter
        Attention parameter related to embedding of x_i
    att_em_j : nn.Parameter
        Attention parameter related to embedding of x_j
    bias : nn.Parameter
        Bias parameter added after message propagation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat_heads=True,
        negative_slope=0.2,
        dropout=0,
    ):
        super(GraphLayer, self).__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        # parameters related to weight matrix W
        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        # attention parameters related to x_i, x_j
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))

        # attention parameters related embeddings v_i, v_j
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        # if concatenating the across heads, consider the change of out_channels
        self._out_channels = heads * out_channels if concat_heads else out_channels
        self.bias = Parameter(torch.Tensor(self._out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise parameters of GraphLayer."""
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        self.bias.data.zero_()

    def forward(self, x, edge_index, embedding):
        """Forward method for propagating messages of GraphLayer.

        Parameters
        ----------
        x : tensor
            has shape [N x batch_size, in_channels], where N is the number of nodes
        edge_index : tensor
            has shape [2, E x batch_size], where E is the number of edges
            with E = topk x N
        embedding : tensor
            has shape [N x batch_size, out_channels]
        """
        # linearly transform node feature matrix
        assert torch.is_tensor(x)
        x = self.lin(x)

        # add self loops, nodes are in dim 0 of x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        # propagate messages
        out = self.propagate(
            edge_index,
            x=(x, x),
            embedding=embedding,
            edges=edge_index,
        )

        # transform [N x batch_size, 1, _out_channels] to [N x batch_size, _out_channels]
        out = out.view(-1, self._out_channels)

        # apply final bias vector
        out += self.bias

        return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges):
        """Calculate the attention weights using the embedding vector, eq (6)-(8) in [1].

        Parameters
        ----------
        x_i : tensor
            has shape [(topk x N x batch_size), out_channels]
        x_j : tensor
            has shape [(topk x N x batch_size), out_channels]
        edge_index_i : tensor
            has shape [(topk x N x batch_size)]
        size_i : int
            with value (N x batch_size)
        embedding : tensor
            has shape [(N x batch_size), out_channels]
        edges : tensor
            has shape [2, (topk x N x batch_size)]
        """
        # transform to [(topk x N x batch_size), 1, out_channels]
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            # [(topk x N x batch_size), 1, out_channels]
            embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)

            # [(topk x N x batch_size), 1, 2 x out_channels]
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat(
                (x_j, embedding_j), dim=-1
            )  # concatenates along the last dim, i.e. columns in this case

        # concatenate learnable parameters to become [1, 1, 2 x out_channels]
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        # [(topk x N x batch_size), 1]
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(
            -1
        )  # the matrix multiplication between a^T and g's in eqn (7)

        alpha = alpha.view(-1, self.heads, 1)  # [(topk x N x batch_size), 1, 1]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, None, size_i)  # eqn (8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # save to self
        self.alpha = alpha

        # multiply node feature by alpha
        return x_j * alpha

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


class OutLayer(nn.Module):
    """
    Output layer used to transform graph layers into a prediction.

    Attributes
    ----------
    mlp : nn.ModuleList
        A module list that contains a sequence of transformations in the output layer
    """

    def __init__(self, in_num, layer_num, inter_dim):
        """
        Parameters
        ----------
        in_num : int
            input dimension of network
        layer_num : int
            number of layers in network
        inter_dim : int
            internal dimensions of layers in network
        """
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_dim, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_dim
                modules.extend(
                    (
                        nn.Linear(layer_in_num, inter_dim),
                        nn.BatchNorm1d(inter_dim),
                        nn.ReLU(),
                    )
                )
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        """
        Forward pass of output layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        out : torch.Tensor
            Output tensor
        """
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    """
    Calculates the node representations, z_i, in eq (5) of [1].

    Attributes
    ----------
    gnn : GraphLayer
        Graph convolutional layer
    bn : nn.BatchNorm1d
        Batch normalization layer
    relu : nn.ReLU
        ReLU activation function
    """

    def __init__(self, in_channel, out_channel, heads=1):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels for the layer
        out_channels : int
            Number of output channels for the layer
        heads : int
            Number of heads for multi-head attention
        """
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, heads=heads, concat_heads=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, embedding=None):
        out = self.gnn(x, edge_index, embedding)
        out = self.bn(out)
        return self.relu(out)


class GDN(nn.Module):
    """
    A graph-based network model for time series data, as introduced in [1].

    Attributes
    ----------
    embedding : nn.Embedding
        Node embeddings for the graph
    bn_outlayer_in : nn.BatchNorm1d
        Batch normalization layer applied before the output layer
    gnn_layers : nn.ModuleList
        List of GNNLayer instances used in the network
    learned_graph : tensor
        Topk indices represneting the learned graph, with shape [N, top_k]
    out_layer : OutLayer
        Output layer for the network
    dp : nn.Dropout
        Dropout layer applied before the output layer
    cache_fc_edge_idx : tensor
        has shape [2, (E x batch_size)] where E is the number of edges
    """

    def __init__(
        self,
        fc_edge_idx,
        n_nodes,
        embed_dim=64,
        out_layer_inter_dim=256,
        slide_win=15,
        out_layer_num=1,
        topk=20,
    ):
        """
        Parameters
        ----------
        fc_edge_idx : torch.LongTensor
            Edge indices of fully connected graph for the input time series
        n_nodes : int
            Number of nodes in the graph
        embed_dim : int, optional (default=64)
            Dimension of node embeddings
        out_layer_inter_dim : int, optional (default=256)
            Internal dimensions of layers in the output network
        slide_win : int, optional (default=15)
            Size of sliding window used for input time series
        out_layer_num : int, optional (default=1)
            Number of layers in OutLayer
        topk : int, optional (default=20)
            Number of top-k neighbors to consider when creating learned graph
        """
        super(GDN, self).__init__()

        self.fc_edge_idx = fc_edge_idx
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.out_layer_inter_dim = out_layer_inter_dim
        self.slide_win = slide_win
        self.out_layer_num = out_layer_num
        self.topk = topk

    def _initialise_layers(self):
        self.embedding = nn.Embedding(self.n_nodes, self.embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim)

        self.gnn_layers = nn.ModuleList(
            [
                GNNLayer(
                    self.slide_win,
                    self.embed_dim,
                    heads=1,
                )
            ]
        )
        self.out_layer = OutLayer(
            self.embed_dim, self.out_layer_num, inter_dim=self.out_layer_inter_dim
        )

        self.dp = nn.Dropout(0.2)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data):
        x = data.clone().detach()
        device = data.device
        batch_size = x.shape[0]

        x = x.view(-1, self.slide_win).contiguous()  # [(batch_size x N), slide_win]

        self.cache_fc_edge_idx = get_batch_edge_index(
            self.fc_edge_idx, batch_size, self.n_nodes
        ).to(device)

        idxs = torch.arange(self.n_nodes).to(device)
        weights = self.embedding(idxs).detach().clone()  # [N, embed_dim]
        batch_embeddings = self.embedding(idxs).repeat(
            batch_size, 1
        )  # [(N x batch_size), embed_dim]

        # e_{ji} in eqn (2)
        cos_ji_mat = torch.matmul(weights, weights.T)  # [N , N]
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat

        # A_{ji} in eqn (3)
        topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]
        self.learned_graph = topk_indices_ji  # [N x topk]

        gated_i = (
            torch.arange(0, self.n_nodes)
            .repeat_interleave(self.topk)
            .unsqueeze(0)
            .to(device)
        )  # [N x topk]
        gated_j = topk_indices_ji.flatten().unsqueeze(0)  # [N x topk]
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)  # [2, (N x topk)]

        batch_gated_edge_index = get_batch_edge_index(
            gated_edge_index, batch_size, self.n_nodes
        ).to(
            device
        )  # [2, (N x topk x batch_size)]

        gcn_out = self.gnn_layers[0](
            x,
            batch_gated_edge_index,
            embedding=batch_embeddings,
        )
        gcn_out = gcn_out.view(
            batch_size, self.n_nodes, -1
        )  # [batch_size, N, embed_dim]

        # eqn (9), element-wise multiply node representation z_i with corresponding embedding v_i
        out = torch.mul(gcn_out, self.embedding(idxs))  # [batch_size, N, embed_dim]
        out = out.permute(0, 2, 1)  # [batch_size, embed_dim, N]
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)  # [batch_size, N, embed_dim]
        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, self.n_nodes)  # [batch_size, N]

        return out


def get_batch_edge_index(edge_index, batch_size, n_nodes):
    """
    Replicates neighbour relations for new batch index values.

    Parameters
    ----------
    edge_index : tensor
        has shape [2, E] where E is the number of edges
    batch_size : int
        the size of the batch
    n_nodes : int
        number of nodes, N

    Returns
    -------
    batch_edge_index : tensor
        has shape [2, (E x batch_size)] where E is the number of edges

    Example
    -------
    >>> edge_index = tensor([[0, 2, 1, 2, 2, 1],
                             [0, 0, 1, 1, 2, 2]])
    >>> get_batch_edge_index(edge_index, 2, 3)
    >>> tensor([[0, 2, 1, 2, 2, 1, 3, 5, 4, 5, 5, 4],
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    """
    edge_index = edge_index.clone().detach()
    edge_num = edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_size).contiguous()

    for i in range(batch_size):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * n_nodes

    return batch_edge_index.long()


class GNNAD:
    """
    Graph Neural Network-based Anomaly Detection for Multivariate Timeseries.
    """

    def __init__(
        self,
        batch: int = 128,
        epoch: int = 100,
        slide_win: int = 15,
        embed_dim: int = 64,
        slide_stride: int = 5,
        random_seed: int = 0,
        out_layer_num: int = 1,
        out_layer_inter_dim: int = 256,
        decay: float = 0,
        validate_ratio: float = 0.1,
        topk: int = 20,
        device: str = "cpu",
        save_model_name: str = "",
        early_stop_win: int = 15,
        lr: float = 0.001,
        shuffle_train: bool = True,
        threshold_type: str = None,
        suppress_print: bool = False,
        smoothen_error: bool = True,
        use_deterministic: bool = False,
    ):
        """
        Parameters
        ----------
        batch : int, optional (default=128)
            Batch size for training the model
        epoch : int, optional (default=100)
            Number of epochs to train the model
        slide_win : int, optional (default=15)
            Size of sliding window used as feature input
        embed_dim : int, optional (default=64)
            Dimension of the node embeddings in the GDN model
        slide_stride : int, optional (default=5)
            Stride of the sliding window
        random_seed : int, optional (default=0)
            Seed for random number generation for reproducibility
        out_layer_num : int, optional (default=1)
            Number of layers in the output network
        out_layer_inter_dim : int, optional (default=256)
            Internal dimensions of layers in the output network
        decay : float, optional (default=0)
            Weight decay factor for regularization during training
        validate_ratio : float, optional (default=0.1)
            Ratio of data to use for validation during training
        topk : int, optional (default=20)
            Number of permissable neighbours in the learned graph
        device : str, optional (default="cpu")
            Device to use for training the model ('cpu' or 'cuda')
        save_model_name : str, optional (default="")
            Name to use for saving the trained model
        early_stop_win : int, optional (default=15)
            Number of consecutive epochs without improvement in validation loss to
            trigger early stopping
        lr : float, optional (default=0.001)
            Learning rate for training the model
        shuffle_train : bool, optional (default=True)
            Whether to shuffle the training data during training
        threshold_type : str, optional (default=None)
            Type of threshold to use for anomaly detection ("max_validation")
        suppress_print : bool, optional (default=False)
            Whether to suppress print statements during training
        smoothen_error : bool, optional (default=True)
            Whether to smoothen the anomaly scores before thresholding
        use_deterministic : bool, optional (default=False)
            Whether to use deterministic algorithms for reproducibility and unit testing
        """

        self.batch = batch
        self.epoch = epoch
        self.slide_win = slide_win
        self.embed_dim = embed_dim
        self.slide_stride = slide_stride
        self.random_seed = random_seed
        self.out_layer_num = out_layer_num
        self.out_layer_inter_dim = out_layer_inter_dim
        self.decay = decay
        self.validate_ratio = validate_ratio
        self.topk = topk
        self.device = device
        self.save_model_name = save_model_name
        self.early_stop_win = early_stop_win
        self.lr = lr
        self.shuffle_train = shuffle_train
        self.threshold_type = threshold_type
        self.suppress_print = suppress_print
        self.smoothen_error = smoothen_error
        self.use_deterministic = use_deterministic

    def _set_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        if self.use_deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _get_loader_generator(self):
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        return g

    def _split_train_validation(self, data):
        dataset_len = len(data)
        validate_use_len = int(dataset_len * self.validate_ratio)
        validate_start_idx = random.randrange(dataset_len - validate_use_len)
        idx = torch.arange(dataset_len)

        train_sub_idx = torch.cat(
            [idx[:validate_start_idx], idx[validate_start_idx + validate_use_len :]]
        )
        train_subset = Subset(data, train_sub_idx)

        validate_sub_idx = idx[
            validate_start_idx : validate_start_idx + validate_use_len
        ]
        validate_subset = Subset(data, validate_sub_idx)

        return train_subset, validate_subset

    def _load_data(self, X_train, X_test, y_test):
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        feature_list = X_train.columns[
            X_train.columns.str[0] != "_"
        ].to_list()  # convention is to pass non-features as '_'
        assert len(feature_list) == len(set(feature_list))

        fc_struc = {
            ft: [x for x in feature_list if x != ft] for ft in feature_list
        }  # fully connected structure

        edge_idx_tuples = [
            (feature_list.index(child), feature_list.index(node_name))
            for node_name, node_list in fc_struc.items()
            for child in node_list
        ]

        fc_edge_idx = [
            [x[0] for x in edge_idx_tuples],
            [x[1] for x in edge_idx_tuples],
        ]
        fc_edge_idx = torch.tensor(fc_edge_idx, dtype=torch.long)

        train_input = parse_data(X_train, feature_list)
        test_input = parse_data(X_test, feature_list, labels=y_test)

        cfg = {
            "slide_win": self.slide_win,
            "slide_stride": self.slide_stride,
        }

        train_dataset = TimeDataset(train_input, fc_edge_idx, mode="train", config=cfg)
        test_dataset = TimeDataset(test_input, fc_edge_idx, mode="test", config=cfg)

        train_subset, validate_subset = self._split_train_validation(train_dataset)

        # get data loaders
        g = self._get_loader_generator() if self.use_deterministic else None

        train_dataloader = DataLoader(
            train_subset,
            batch_size=self.batch,
            shuffle=self.shuffle_train,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        validate_dataloader = DataLoader(
            validate_subset,
            batch_size=self.batch,
            shuffle=False,
            generator=g,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch,
            shuffle=False,
            generator=g,
        )

        # save to self
        self.n_nodes = len(feature_list)
        self.fc_edge_idx = fc_edge_idx
        self.feature_list = feature_list
        self.test_input = test_input
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader

    def _load_model(self):
        # instantiate model
        model = GDN(
            self.fc_edge_idx,
            n_nodes=self.n_nodes,
            slide_win=self.slide_win,
            out_layer_num=self.out_layer_num,
            out_layer_inter_dim=self.out_layer_inter_dim,
            topk=self.topk,
            embed_dim=self.embed_dim,
        ).to(self.device)

        model._initialise_layers()

        self.model = model

    def _get_model_path(self):
        datestr = datetime.now().strftime("%m%d-%H%M%S")
        model_name = datestr if len(self.save_model_name) == 0 else self.save_model_name
        model_path = f"./pretrained/{model_name}.pt"
        dirname = os.path.dirname(model_path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

        self.model_path = model_path

    def _test(self, model, dataloader):
        start = datetime.now()

        test_loss_list = []
        acu_loss = 0

        t_test_predicted_list = []
        t_test_ground_list = []
        t_test_labels_list = []

        model.eval()

        for i, (x, y, labels, edge_index) in enumerate(dataloader):
            x, y, labels, edge_index = [
                item.to(self.device).float() for item in [x, y, labels, edge_index]
            ]

            with torch.no_grad():
                predicted = model(x).float().to(self.device)

                loss = loss_func(predicted, y)

                labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

                if len(t_test_predicted_list) <= 0:
                    t_test_predicted_list = predicted
                    t_test_ground_list = y
                    t_test_labels_list = labels
                else:
                    t_test_predicted_list = torch.cat(
                        (t_test_predicted_list, predicted), dim=0
                    )
                    t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                    t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            test_loss_list.append(loss.item())
            acu_loss += loss.item()

            if i % 10000 == 1 and i > 1:
                print(str_time_elapsed(start, i, len(dataloader)))

        test_predicted_list = t_test_predicted_list.tolist()
        test_ground_list = t_test_ground_list.tolist()
        test_labels_list = t_test_labels_list.tolist()

        avg_loss = sum(test_loss_list) / len(test_loss_list)

        return avg_loss, np.array(
            [test_predicted_list, test_ground_list, test_labels_list]
        )

    def _train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.decay
        )

        train_log = []
        max_loss = 1e8
        stop_improve_count = 0

        for i_epoch in range(self.epoch):
            acu_loss = 0
            self.model.train()

            for i, (x, y, _, edge_index) in enumerate(self.train_dataloader):
                x, y, edge_index = [
                    item.float().to(self.device) for item in [x, y, edge_index]
                ]
                optimizer.zero_grad()

                out = self.model(x).float().to(self.device)

                loss = loss_func(out, y)

                loss.backward()
                optimizer.step()

                train_log.append(loss.item())
                acu_loss += loss.item()

            # each epoch
            if not self.suppress_print:
                print(
                    "epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})".format(
                        i_epoch, self.epoch, acu_loss / (i + 1), acu_loss
                    ),
                    flush=True,
                )

            # use val dataset to judge
            if self.validate_dataloader is not None:
                val_loss, _ = self._test(self.model, self.validate_dataloader)

                if val_loss < max_loss:
                    torch.save(self.model.state_dict(), self.model_path)

                    max_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1

                if stop_improve_count >= self.early_stop_win:
                    break

            elif acu_loss < max_loss:
                torch.save(self.model.state_dict(), self.model_path)
                max_loss = acu_loss

        self.train_log = train_log

    def _get_score(self):
        # read in best model
        self.model.load_state_dict(torch.load(self.model_path))
        best_model = self.model.to(self.device)

        # store results to self
        test_avg_loss, self.test_result = self._test(best_model, self.test_dataloader)
        _, self.validate_result = self._test(best_model, self.validate_dataloader)

        test_labels = self.test_result[2, :, 0]
        test_err_scores = get_full_err_scores(self.test_result, self.smoothen_error)
        validate_err_scores = get_full_err_scores(
            self.validate_result, self.smoothen_error
        )
        topk_err_indices, topk_err_scores = aggregate_error_scores(test_err_scores)

        # get threshold value
        if self.threshold_type == "max_validation":
            threshold = np.max(validate_err_scores)
            f1 = None
        else:
            final_topk_fmeas, thresholds = eval_scores(topk_err_scores, test_labels)
            th_i = final_topk_fmeas.index(max(final_topk_fmeas))
            threshold = thresholds[th_i]
            f1 = max(final_topk_fmeas)

        # get prediction labels for decided threshold
        pred_labels = np.zeros(len(topk_err_scores))
        pred_labels[topk_err_scores > threshold] = 1

        pred_labels = pred_labels.astype(int)
        test_labels = test_labels.astype(int)

        # calculate metrics
        precision = precision_score(test_labels, pred_labels)
        recall = recall_score(test_labels, pred_labels)
        f1 = f1_score(test_labels, pred_labels) if f1 is None else f1
        auc = roc_auc_score(test_labels, topk_err_scores)

        # save to self
        self.validate_err_scores = validate_err_scores
        self.test_err_scores = test_err_scores
        self.topk_err_indices = topk_err_indices
        self.topk_err_scores = topk_err_scores
        self.pred_labels = pred_labels
        self.test_labels = test_labels
        self.threshold = threshold
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.auc = auc
        self.test_avg_loss = test_avg_loss

        if not self.suppress_print:
            print("=========================** Result **============================\n")
            self.print_eval_metrics()

    def fit(self, X_train, X_test, y_test):
        self._set_seeds()
        self._load_data(X_train, X_test, y_test)
        self._load_model()
        self._get_model_path()
        self._train()
        self._get_score()

        return self

    def summary(self):
        return summary(self.model, (self.n_nodes, self.slide_win))

    def print_named_parameters(self):
        for name, param in self.model.named_parameters():
            print(name, param)

    def sensor_threshold_preds(self, tau):
        threshold_i = np.empty(self.n_nodes)
        for i in range(self.n_nodes):
            idxs = self.model.learned_graph[i].numpy()
            threshold_i[i] = np.percentile(
                self.validate_err_scores[idxs].flatten(), tau
            )

        pred_labels = np.empty(self.test_err_scores.shape[1])

        for t in range(self.test_err_scores.shape[1]):
            pred_labels[t] = any(self.test_err_scores[:, t] > threshold_i)

        self.threshold_i = threshold_i
        return pred_labels.astype(int)

    def get_sensor_preds(self):
        sensor_preds = np.empty(self.test_err_scores.shape)

        for t in range(self.test_err_scores.shape[1]):
            sensor_preds[:, t] = self.test_err_scores[:, t] > self.threshold_i

        return sensor_preds

    def print_eval_metrics(self, preds=None):
        preds = self.pred_labels if preds is None else preds
        recall, precision, accuracy, specificity, f1 = eval_metrics(
            self.test_labels, preds
        )
        print("recall: %.1f" % recall)
        print("precision: %.1f" % precision)
        print("accuracy: %.1f" % accuracy)
        print("specificity: %.1f" % specificity)
        print("f1: %.1f" % f1)


def eval_metrics(truth, preds):
    precision = precision_score(truth, preds) * 100
    recall = recall_score(truth, preds) * 100
    f1 = f1_score(truth, preds) * 100
    accuracy = accuracy_score(truth, preds) * 100
    tn, fp, fn, tp = confusion_matrix(truth, preds).ravel()
    specificity = tn / (tn + fp) * 100

    return recall, precision, accuracy, specificity, f1


def loss_func(y_pred, y_true):
    return F.mse_loss(y_pred, y_true, reduction="mean")


def parse_data(data, feature_list, labels=None):
    """
    In the case of training data, fill the last column with zeros. This is an
    implicit assumption in the uhnsupervised training case - that the data is
    non-anomalous. For the test data, keep the labels.
    """
    labels = [0] * data.shape[0] if labels is None else labels
    res = data[feature_list].T.values.tolist()
    res.append(labels)
    return res


def str_seconds_to_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def str_time_elapsed(start, i, total):
    now = datetime.now()
    elapsed = (now - start).seconds
    frac_complete = (i + 1) / total
    remaining = elapsed / frac_complete - elapsed
    return f"{str_seconds_to_minutes(elapsed)} (- {str_seconds_to_minutes(remaining)})"


def get_full_err_scores(test_result, smoothen_error=True):
    """Get stacked array of error scores for each feature by applying the
    `get_err_scores` function on every slice of the `test_result` tensor.
    """
    all_scores = [
        get_err_scores(test_result[:2, :, i], smoothen_error)
        for i in range(test_result.shape[-1])
    ]
    return np.vstack(all_scores)


def get_err_scores(test_result_list, smoothen_error):
    """
    Calculate the error scores, normalised by the median and interquartile range.

    Parameters
    ----------
    test_result_list (list):
        List containing two lists of predicted and ground truth values
    smoothen_error (bool):
        A boolean value indicating whether error smoothing should be applied or not

    Returns
    -------
    err_scores (np.ndarray):
        An array of error scores
    """
    test_predict, test_ground = test_result_list

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_ground)

    test_delta = np.abs(
        np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_ground).astype(np.float64),
        )
    )
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    if smoothen_error:
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3
        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i - before_num : i + 1])

        return smoothed_err_scores
    return err_scores


def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))
    return np.median(np_arr), iqr(np_arr)


def aggregate_error_scores(test_err_scores, topk=1):
    # finds topk feature idx of max scores for each time point
    topk_indices = np.argpartition(test_err_scores, -topk, axis=0)[-topk:]

    # for each time, sum the topk error scores
    topk_err_scores = np.sum(
        np.take_along_axis(test_err_scores, topk_indices, axis=0), axis=0
    )

    return topk_indices, topk_err_scores


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps=400):
    padding_list = [0] * (len(true_scores) - len(scores))

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_rank = rankdata(scores, method="ordinal")  # rank of score
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps

    for i in range(th_steps):
        cur_pred = scores_rank > th_vals[i] * len(scores)
        fmeas[i] = f1_score(true_scores, cur_pred)
        score_index = scores_rank.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    return fmeas, thresholds


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
