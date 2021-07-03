import torch
import torch.nn as nn
import torch.nn.functional as functional


class MoCo(nn.Module):
    """
    Build a DataParallel MoCo V2 model with: a query encoder, a key encoder, and a queue.
    Based on official DistributedDataParallel version.
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, k=65536, m=0.999, t=0.07):
        """
        :param dim: feature dimension (default: 128)
        :param k: queue size, number of negative keys (default: 65536)
        :param m: moco momentum of updating key encoder (default: 0.999)
        :param t: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.k, self.m, self.t = k, m, t

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # replace the final fully connected layer
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                          nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                          nn.ReLU(), self.encoder_k.fc)

        # initialize encoder_k parameters with no grad
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer('queue', torch.randn(dim, k))
        self.queue = functional.normalize(self.queue, dim=0)

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.k  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_dp(self, x: torch.Tensor):
        """
        Batch shuffle, for making use of BatchNorm.
        Only Support DataParallel(DP) model.
        :param x: raw data
        :return: shuffled data, unshuffle key
        """
        # random shuffle key
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # unshuffle key
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffle data
        x_shuffle = x[idx_shuffle]

        return x_shuffle, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_dp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        Only Support DataParallel(DP) model.
        :param x: shuffled data
        :param idx_unshuffle: unshuffle key
        :return: raw data
        """
        x_unshuffle = x[idx_unshuffle]
        return x_unshuffle

    def forward(self, im_q, im_k):
        """
        :param im_q: a batch of query images
        :param im_k: a batch of key images
        :return: logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: N*C
        q = functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no grad to keys

            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_dp(im_k)

            # compute keys features
            k = self.encoder_k(im_k)  # keys: N*C
            k = functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_dp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive (He.)
        # positive logits: N*1
        l_pos = torch.einsum('nc, nc -> n', q, k).unsqueeze(-1)
        # negative logits: N*K
        l_neg = torch.einsum('nc, ck -> nk', q, self.queue.clone().detach())

        # logits: N*(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
