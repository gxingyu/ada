import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from pygda.models import BaseGDA
from pygda.nn import A2GNNBase
from pygda.utils import logger, MMD
from pygda.metrics import eval_macro_f1, eval_micro_f1
from utils.loss import CF
from utils.loss import SampleNet

class NCFM(BaseGDA):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        mode='node',
        num_layers=2,
        dropout=0.25,
        act=F.relu,
        s_pnums=0,
        t_pnums=10,
        adv=False,
        weight=0.05,
        weight_decay=0.001,
        lr=0.005,
        epoch=150,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        t_batchsize=4096,
        alpha=0.5,
        **kwargs):
        
        super(NCFM, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            weight_decay=weight_decay,
            lr=lr,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            **kwargs)
        
        self.s_pnums=s_pnums
        self.t_pnums=t_pnums
        self.adv=adv
        self.weight=weight
        self.mode=mode
        self.t_batchsize=t_batchsize
        self.alpha=alpha

    def init_model(self, **kwargs):

        return A2GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            adv=self.adv,
            dropout=self.dropout,
            act=self.act,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha):
        # source domain cross entropy loss
        source_logits = self.a2gnn(source_data, self.s_pnums)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        if self.mode == 'node':
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.a2gnn.feat_bottleneck(source_data.x, source_data.edge_index, source_batch, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(target_data.x, target_data.edge_index, target_batch, self.t_pnums)
        
        # Adv loss
        if self.adv:
            source_dlogits = self.a2gnn.domain_classifier(source_features, alpha)
            target_dlogits = self.a2gnn.domain_classifier(target_features, alpha)
            
            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0]
                ).to(self.device)
            
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.weight * domain_loss
        else:
            sampled_t = self.sample_net(self.device)
            # CF loss
            cf_loss = CF(self.alpha, 1-self.alpha, source_features, target_features,sampled_t)
            # MMD loss
            #mmd_loss = MMD(source_features, target_features)
            #loss = loss + mmd_loss * self.weight
            loss = loss + self.weight * cf_loss
        target_logits = self.a2gnn(target_data, self.t_pnums)

        return loss, source_logits, target_logits, cf_loss

    def fit(self, source_data, target_data):
        if self.mode == 'node':
            self.num_source_nodes, _ = source_data.x.shape
            self.num_target_nodes, _ = target_data.x.shape

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.source_batch_size)
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.target_batch_size)
            else:
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
        elif self.mode == 'graph':
            if self.batch_size == 0:
                num_source_graphs = len(source_data)
                num_target_graphs = len(target_data)
                self.source_loader = DataLoader(source_data, batch_size=num_source_graphs, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=num_target_graphs, shuffle=True)
            else:
                self.source_loader = DataLoader(source_data, batch_size=self.batch_size, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=self.batch_size, shuffle=True)
        else:
            assert self.mode in ('graph', 'node'), 'Invalid train mode'

        self.a2gnn = self.init_model(**self.kwargs)

        # 初始化采样网络
        feature_dim = self.a2gnn.feat_bottleneck.out_features if hasattr(self.a2gnn.feat_bottleneck,
                                                                         'out_features') else 128
        self.sample_net = SampleNet(feature_dim=feature_dim, t_batchsize=self.t_batchsize, t_var=1).to(self.device) # t_batchsize=4096/2048/1024/512

        # 分别为主网络和采样网络创建优化器
        optimizer_main = torch.optim.Adam(
            self.a2gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        optimizer_sample = torch.optim.Adam(
            self.sample_net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()
        max_ma = -1
        max_mi = -1
        mi_list = []
        loss_list = []
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.a2gnn.train()
                self.sample_net.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                # 第一步：最大化对齐loss（训练采样网络）
                if not self.adv:  # 只在使用CF loss时进行
                    # 冻结主网络参数
                    for param in self.a2gnn.parameters():
                        param.requires_grad = False

                    optimizer_sample.zero_grad()
                    loss, source_logits, target_logits, cf_loss = self.forward_model(sampled_source_data,
                                                                                     sampled_target_data, alpha)

                    # 最大化cf_loss等价于最小化-cf_loss
                    (-cf_loss).backward()
                    optimizer_sample.step()

                    # 解冻主网络参数
                    for param in self.a2gnn.parameters():
                        param.requires_grad = True

                # 第二步：最小化source loss + 对齐loss（训练主网络）
                optimizer_main.zero_grad()
                loss, source_logits, target_logits, cf_loss = self.forward_model(sampled_source_data,
                                                                                 sampled_target_data, alpha)
                epoch_loss += loss.item()

                loss.backward()
                optimizer_main.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_logits, source_labels = source_logits, sampled_source_data.y
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            loss_list.append(epoch_loss)
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
            
            logits, labels = self.predict(target_data)
            preds = logits.argmax(dim=1)
            mi_f1 = eval_micro_f1(labels, preds)
            mi_list.append(mi_f1)
            ma_f1 = eval_macro_f1(labels, preds)
            max_mi = max(mi_f1, max_mi)
            max_ma = max(ma_f1, max_ma)
        print(loss_list)
        return max_mi, max_ma
    
    def process_graph(self, data):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        Placeholder method as preprocessing is handled through:
        
        - Asymmetric propagation mechanisms
        - Domain-specific feature processing
        - Batch-wise data handling
        """
    def predictAndUpdate(self, data):
        self.a2gnn.eval()
        data_loader = NeighborLoader(
            data,
            self.num_neigh,
            batch_size=data.x.shape[0])
        for idx, sampled_data in enumerate(data_loader):
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                node_idx = sampled_data.x[:, -1]  # 假设最后一维是节点索引，取出这一维
                sampled_data.x = sampled_data.x[:, :-1]
                logits = self.a2gnn(sampled_data, self.s_pnums)

                if idx == 0:
                    logits, labels = logits, sampled_data.y
                else:
                    sampled_logits, sampled_labels = logits, sampled_data.y
                    logits = torch.cat((logits, sampled_logits))
                    labels = torch.cat((labels, sampled_labels))
                # 获取预测的标签
                predicted_labels = logits.argmax(dim=-1)
                for i, idx in enumerate(node_idx):
                    data.y[int(idx)] = predicted_labels[i].item()
        return data
    def predict(self, data, source=False):
        """
        Make predictions on input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        source : bool, optional
            Whether predicting on source domain. Default: ``False``.

        Returns
        -------
        tuple
            Contains:
            - logits : torch.Tensor
                Model predictions.
            - labels : torch.Tensor
                True labels.

        Notes
        -----
        - Uses different propagation steps for source/target
        - Handles batch processing efficiently
        - Concatenates results for full predictions
        - Maintains evaluation mode consistency
        """

        self.a2gnn.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.a2gnn(sampled_data, self.s_pnums)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))
        else:
            for idx, sampled_data in enumerate(self.target_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.a2gnn(sampled_data, self.t_pnums)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))
        return logits, labels
