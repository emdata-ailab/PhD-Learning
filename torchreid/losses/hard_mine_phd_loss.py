from __future__ import division, absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class PhdLoss(nn.Module):
    """PhD loss with hard positive/negative mining.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
        k_ap (int, optional): slack factor for postive sample mining. Default is 3.
        k_an (int, optional): slack factor for negative sample mining. Default is 6.
        normalize (bool, optional): normalize the feature vector before rank loss caluation. Default is True.
        vis_batch_hard (bool, optional): visual the hard samples in PhD metric learning. Default is False.
    Return:
        phd_loss (torch.tensor): PhD loss 
    """

    def __init__(self, margin=0.3, k_ap=3, k_an=6, normalize=True, vis_batch_hard=False):
        super(PhdLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.k_an = k_an   # (k-an越大，越难)
        self.k_ap = k_ap   # (k-ap越小,越难 )
        self.normalize = normalize
        self.vis_batch_hard = vis_batch_hard

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, sequence_length, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """
        assert inputs.dim() == 3, "3 dims should be supplied."
        b, s, d = inputs.size()
        if self.normalize:
            inputs = F.normalize(inputs, p=2, dim=-1)
        inputs = torch.reshape(inputs, (b*s, d))
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(b*s, b*s)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
        
        ## sweep the distance matrix by a window with s*s
        dist_ny = dist.numpy() if not dist.is_cuda else dist.cpu().numpy()
        strides = dist_ny.itemsize * np.array([b*s*s, s, b*s, 1])
        shape = (b, b, s, s)
        dist_ny = np.lib.stride_tricks.as_strided(dist_ny, shape=shape, strides=strides)
        dist_ny = dist_ny.reshape(-1, s, s) ## (b*b, s, s)

        ## mask maker
        matches = targets.unsqueeze(1).eq(targets.unsqueeze(0)).reshape(-1).byte()
        #print(matches)

        diffs = matches ^ 1
        
        ## minimum matching
        dist = torch.tensor(dist_ny, device=dist.device)
        #print(dist.size())
        dist_ij_min, dist_ij_min_idx = torch.min(dist, 1)    ## (b*b, s)   
        dist_ji_min, dist_ji_min_idx = torch.min(dist, 2)    ## (b*b, s)

        ## for positive
        dist_ij_min_ap = dist_ij_min[matches.nonzero()].squeeze()     ## (#ap, s)
        dist_ji_min_ap = dist_ji_min[matches.nonzero()].squeeze()     ## (#ap, s)
        #print(dist_ij_min_ap.size(), dist_ji_min_ap.size())
        ## unidirectional distance
        dist_ij_min_ap_kmax, dist_ij_min_ap_kmax_idx = torch.topk(dist_ij_min_ap, self.k_ap)   ## (#ap, k_ap)
        dist_ji_min_ap_kmax, dist_ji_min_ap_kmax_idx = torch.topk(dist_ji_min_ap, self.k_ap)   ## (#ap, k_ap)
        ## bidirectional distance
        combine_undistance_ap = torch.stack([dist_ij_min_ap_kmax[:,-1], dist_ji_min_ap_kmax[:,-1]], dim=1)
        #print(combine_undistance_ap.size())
        dist_ij_ji_ap, ij_ji_ap_index = torch.max(combine_undistance_ap, dim=1)        ## (#ap, )
        ## hard mining
        dist_ij_ji_ap = dist_ij_ji_ap.reshape(b, -1)
        dist_ap, dist_ap_index = torch.max(dist_ij_ji_ap, 1)
        p_index = (matches.nonzero()%b).reshape(b, -1).gather(1, dist_ap_index.unsqueeze(1))

        ## for negative
        dist_ij_min_an = dist_ij_min[diffs.nonzero()].squeeze()       ## (#an, s)
        dist_ji_min_an = dist_ji_min[diffs.nonzero()].squeeze()       ## (#an, s)
        ## unidirectional distance
        dist_ij_min_an_kmax, dist_ij_min_an_kmax_idx = torch.topk(dist_ij_min_an, self.k_an)   ## (#ap, k_ap)
        dist_ji_min_an_kmax, dist_ji_min_an_kmax_idx = torch.topk(dist_ji_min_an, self.k_an)   ## (#ap, k_ap)
        ## bidirectional distance
        combine_undistance_an = torch.stack([dist_ij_min_an_kmax[:,-1], dist_ji_min_an_kmax[:,-1]], dim=1)
        # dist_ij_ji_an = torch.maximum(dist_ij_min_an_kmax[:, -1], dist_ji_min_an_kmax[:, -1)
        dist_ij_ji_an, ij_ji_an_index = torch.min(combine_undistance_an, dim=1)  # maybe better
        ## hard mining
        dist_ij_ji_an = dist_ij_ji_an.reshape(b, -1)
        dist_an, dist_an_index = torch.min(dist_ij_ji_an, 1)
        n_index = (diffs.nonzero()%b).reshape(b, -1).gather(1, dist_an_index.unsqueeze(1))

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        params_for_vis_batch_hard = []

        if self.vis_batch_hard:
            # for positive
            col_frame_index_ap = dist_ij_min_idx[matches.nonzero()].squeeze()   # (#ap, s)
            row_frame_index_ap = dist_ji_min_idx[matches.nonzero()].squeeze()   # (#ap, s)
            col_frame_index_ap_chosen = torch.gather(col_frame_index_ap, 1, dist_ij_min_ap_kmax_idx[:,-1].unsqueeze(1))  ## (#ap, 1)
            row_frame_index_ap_chosen = torch.gather(row_frame_index_ap, 1, dist_ji_min_ap_kmax_idx[:,-1].unsqueeze(1))  ## (#ap, 1)

            row_index_ap = torch.cat([dist_ij_min_ap_kmax_idx[:,[-1]], row_frame_index_ap_chosen], dim=1)    ## (#ap, 2)
            col_index_ap = torch.cat([col_frame_index_ap_chosen, dist_ji_min_ap_kmax_idx[:,[-1]]], dim=1)    ## (#ap, 2)

            row_index_ap = torch.gather(row_index_ap, 1, ij_ji_ap_index.unsqueeze(1))    ## (#ap, 1)
            col_index_ap = torch.gather(col_index_ap, 1, ij_ji_ap_index.unsqueeze(1))    ## (#ap, 1)

            row_index_ap_hard = row_index_ap.reshape(b, -1).gather(1, dist_ap_index.unsqueeze(1))  ## (b, 1)
            col_index_ap_hard = col_index_ap.reshape(b, -1).gather(1, dist_ap_index.unsqueeze(1))  ## (b, 1)

            # for negative
            col_frame_index_an = dist_ij_min_idx[diffs.nonzero()].squeeze()   # (#ap, s)
            row_frame_index_an = dist_ji_min_idx[diffs.nonzero()].squeeze()   # (#ap, s)
            col_frame_index_an_chosen = torch.gather(col_frame_index_an, 1, dist_ij_min_an_kmax_idx[:,-1].unsqueeze(1))  ## (#ap, 1)
            row_frame_index_an_chosen = torch.gather(row_frame_index_an, 1, dist_ji_min_an_kmax_idx[:,-1].unsqueeze(1))  ## (#ap, 1)

            row_index_an = torch.cat([dist_ij_min_an_kmax_idx[:,[-1]], row_frame_index_an_chosen], dim=1)    ## (#ap, 2)
            col_index_an = torch.cat([col_frame_index_an_chosen, dist_ji_min_an_kmax_idx[:,[-1]]], dim=1)    ## (#ap, 2)

            row_index_an = torch.gather(row_index_an, 1, ij_ji_an_index.unsqueeze(1))    ## (#ap, 1)
            col_index_an = torch.gather(col_index_an, 1, ij_ji_an_index.unsqueeze(1))    ## (#ap, 1)

            row_index_an_hard = row_index_an.reshape(b, -1).gather(1, dist_an_index.unsqueeze(1))  ## (b, 1)
            col_index_an_hard = col_index_an.reshape(b, -1).gather(1, dist_an_index.unsqueeze(1))  ## (b, 1)

            row_frame_index = torch.cat([row_index_ap_hard, row_index_an_hard], dim=1)
            col_frame_index = torch.cat([col_index_ap_hard, col_index_an_hard], dim=1)

            params_for_vis_batch_hard = [p_index, n_index, row_frame_index, col_frame_index]

        return self.ranking_loss(dist_an, dist_ap, y), params_for_vis_batch_hard

if __name__ == '__main__':
    test_tensor = torch.rand(8, 6, 25)
    test_target = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    test_loss = PhdLoss()
    loss = test_loss(test_tensor, test_target)
    print(loss)