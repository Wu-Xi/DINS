'''
Created on October 1, 2023

@author: WuXi
'''
import torch
import torch.nn as nn

class Matrix_Factorization(nn.Module):
    def __init__(self, data_config, args_config):
        super(Matrix_Factorization, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.ns = args_config.ns
        self.alpha = args_config.alpha
        self.decay = args_config.l2
        self.latent_size = args_config.dim
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_size))),
        })
        return embedding_dict

    def generate(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def forward(self,batch):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        rec_user_emb, rec_item_emb = self.generate()
        if self.ns=='rns':
            neg_gcn_emb_ = rec_item_emb[neg_item[:, :1]]
            neg_item_emb = neg_gcn_emb_.view(-1,self.latent_size)
        else:
            neg_item_emb = self.negative_sampling_DINS(rec_user_emb,rec_item_emb,user,neg_item,pos_item)

        user_emb, pos_item_emb = rec_user_emb[user], rec_item_emb[pos_item]
        batch_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb) + self.l2_reg_loss(self.decay, user_emb,pos_item_emb,neg_item_emb)/user.shape[0]
        return batch_loss,batch_loss,batch_loss


    def bpr_loss(self,user_emb_, pos_item_emb_, neg_item_emb_):
        pos_score = torch.mul(user_emb_, pos_item_emb_).sum(dim=1)
        neg_score = torch.mul(user_emb_, neg_item_emb_).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    def l2_reg_loss(self,reg,user_emb_mf,pos_item_emb_mf,neg_item_emb_mf):
        emb_loss = 0
        emb_loss = torch.norm(user_emb_mf, p=2)/user_emb_mf.shape[0] + torch.norm(pos_item_emb_mf, p=2)/pos_item_emb_mf.shape[0] + torch.norm(neg_item_emb_mf, p=2)/neg_item_emb_mf.shape[0]
        return emb_loss * reg
    
    def negative_sampling_DINS(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, channel]
        
        """Hard Boundary Definition"""
        n_e = item_gcn_emb.unsqueeze(dim=0)
        n_e2 = n_e[:,neg_candidates].squeeze(dim=0) # [batch_size, n_negs, channel]
        scores = (s_e.unsqueeze(dim=1) * n_e2).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        h_n_e=n_e2[torch.arange(batch_size), indices]
        
        """Dimension Independent Mixup"""
        neg_scores = torch.exp(s_e *h_n_e)  # [batch_size, channel]
        pos_scores = self.alpha * torch.exp ((s_e * p_e))   # [batch_size, n_hops, channel]
        total_sum=neg_scores+pos_scores
        neg_weight = neg_scores/total_sum     # [batch_size, n_hops, channel]
        pos_weight = 1-neg_weight   # [batch_size, n_hops, channel]

        n_e_ =  pos_weight * p_e + neg_weight * h_n_e  # mixing
        return n_e_


