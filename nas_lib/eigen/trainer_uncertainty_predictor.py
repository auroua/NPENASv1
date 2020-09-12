# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved

import torch
import random
from ..models.gin_uncertainty_predictor import NasBenchGINGaussianAgent
from gnn_lib.data import Data, Batch
from ..layers.loss_gausian import Criterion
from ..utils.metric_logger import MetricLogger
from ..utils.utils_solver import CosineLR, gen_batch_idx, make_agent_optimizer


class NasBenchGinGaussianTrainer:
    def __init__(self, agent_type, lr=0.01, device=None, epochs=10, train_images=10, batch_size=10, input_dim=6):
        self.nas_agent = NasBenchGINGaussianAgent(input_dim=input_dim)
        self.agent_type = agent_type

        self.criterion = Criterion()
        self.optimizer = make_agent_optimizer(self.nas_agent, base_lr=lr, weight_deacy=1e-4, bias_multiply=True)
        self.device = device
        self.nas_agent.to(self.device)
        self.lr = lr
        self.scheduler = CosineLR(self.optimizer, epochs=epochs, train_images=train_images, batch_size=batch_size)
        self.batch_size = batch_size
        self.epoch = epochs

    def fit(self, edge_index, node_feature, val_accuracy, logger=None):
        meters = MetricLogger(delimiter=" ")
        self.nas_agent.train()
        for epoch in range(self.epoch):
            idx_list = list(range(len(edge_index)))
            random.shuffle(idx_list)
            batch_idx_list = gen_batch_idx(idx_list, self.batch_size)
            counter = 0
            for batch_idx in batch_idx_list:
                counter += len(batch_idx)
                data_list = []
                target_list = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                    target_list.append(val_accuracy[idx])
                val_tensor = torch.tensor(target_list, dtype=torch.float32)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                val_tensor = val_tensor.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                # batch_nodes = F.normalize(batch_nodes, p=2, dim=-1)

                pred, mean, std = self.nas_agent(batch_nodes, batch_edge_idx, batch_idx)
                val_tensor = val_tensor.unsqueeze(dim=1)
                loss = self.criterion(mean, std, val_tensor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                meters.update(loss=loss.item())
        if logger:
            logger.info(meters.delimiter.join(['{loss}'.format(loss=str(meters))]))
        return meters.meters['loss'].avg

    def pred(self, edge_index, node_feature):
        pred_list = []
        mean_list = []
        std_list = []
        idx_list = list(range(len(edge_index)))
        self.nas_agent.eval()
        batch_idx_list = gen_batch_idx(idx_list, 64)
        with torch.no_grad():
            for batch_idx in batch_idx_list:
                data_list = []
                for idx in batch_idx:
                    g_d = Data(edge_index=edge_index[idx].long(), x=node_feature[idx].float())
                    data_list.append(g_d)
                batch = Batch.from_data_list(data_list)
                batch = batch.to(self.device)
                batch_nodes, batch_edge_idx, batch_idx = batch.x, batch.edge_index, batch.batch
                pred, mean, std = self.nas_agent(batch_nodes, batch_edge_idx, batch_idx)

                pred = pred.squeeze()
                mean = mean.squeeze()
                std = std.squeeze()
                if len(pred.size()) == 0:
                    pred.unsqueeze_(0)
                    mean.unsqueeze_(0)
                    std.unsqueeze_(0)
                pred_list.append(pred)
                mean_list.append(mean)
                std_list.append(std)
        return torch.cat(pred_list, dim=0), torch.cat(mean_list, dim=0), torch.cat(std_list, dim=0)

