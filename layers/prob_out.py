import torch
from torch import nn
import torch.nn.functional as F


class SortMultiOutput(nn.Module):

    def __init__(self, in_features, out_units, num_quantiles=100):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * num_quantiles)
        self.drop = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(out_units * num_quantiles,
                                out_units * num_quantiles)
        self.out_units = out_units
        self.nq = num_quantiles

    def forward(self, x):
        out = self.dense(x)
        out = self.drop(F.relu(out))
        out = self.dense2(out)
        out = out.view(*out.shape[:-1], self.out_units, self.nq)
        out, _ = torch.sort(out, dim=-1)
        return out


class NormalOutput(nn.Module):

    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 2)
        self.out_units = out_units

    def forward(self, x, epsilon=0.1):
        out = self.dense(x)
        mu, log_sigma = torch.split(out, self.out_units, dim=-1)
        sigma = epsilon + (1 - epsilon) * F.softplus(log_sigma)
        return torch.distributions.Normal(mu, sigma)


class GammaOutput(nn.Module):

    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 2)
        self.out_units = out_units

    def forward(self, x, epsilon=0.1):
        out = self.dense(x)
        # Output two parameters: concentration (α) and rate (β).
        log_concentration, log_rate = torch.split(out, self.out_units, dim=-1)

        # Use softplus to ensure positive parameters.
        concentration = epsilon + (1 - epsilon) * F.softplus(log_concentration)
        rate = epsilon + (1 - epsilon) * F.softplus(log_rate)

        return torch.distributions.Gamma(concentration, rate)


class MCDropout(nn.Module):

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class NormalInvGamma(nn.Module):

    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x) + 1e-6

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta
