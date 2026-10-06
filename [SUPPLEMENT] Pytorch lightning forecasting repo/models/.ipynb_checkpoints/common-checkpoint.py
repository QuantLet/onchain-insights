import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, mode = 'revin'):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine == 1:
            self._init_params()
        self.type = mode

    def forward(self, x, mode:str):
            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
            elif mode == 'denorm':
                x = self._denormalize(x)
            elif mode == 'denorm_scale':
                x = self._denormalize_scale(x)
            return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.type == 'revin':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
        elif self.type == 'robust':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.median(x, dim=1, keepdim=True).values.detach()
            x_mad = torch.median(torch.abs(x-self.mean), dim=1, keepdim = True).values.detach()
            stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x_mad_aux = stdev * 0.6744897501960817
            x_mad = x_mad * (x_mad>0) + x_mad_aux * (x_mad==0)
            x_mad[x_mad==0] = 1.0
            x_mad = x_mad + self.eps
            self.stdev = x_mad
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def _denormalize_scale(self, x, eps = 1e-5):  
        x = x * self.stdev
        return x
    def robust_statistics(self, x, dim=-1, eps=1e-6):
        return None

class ShapProbWrapper(nn.Module):
    """Wraps a model that outputs probabilities so SHAP sees an nn.Module."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x.float())  # already probabilities
        if out.ndim == 1:
            out = out.unsqueeze(1)        # (B, 1)
        elif out.ndim == 2 and out.shape[1] != 1:
            out = out[:, :1]
        return out


def chebyshev_lobatto_u(J: int, eps: float = 1e-5, device=None):
    # x_j = cos(pi*j/(J-1)) in [-1,1], dense at endpoints
    j = torch.arange(J, device=device, dtype=torch.float32)
    x = torch.cos(torch.pi * j / (J - 1))              # 1..-1
    u = (x + 1.0) / 2.0                                # in [0,1]
    u = torch.flip(u, dims=[0])                        # increasing
    u = u.clamp(eps, 1 - eps)                          # avoid exact 0/1
    return u

def chebyshev_basis(u: torch.Tensor, K: int):
    """
    u: (J,) in (0,1)
    returns T: (J,K) with T_k(2u-1)
    """
    x = 2*u - 1
    T0 = torch.ones_like(x)
    if K == 1:
        return T0.unsqueeze(-1)
    T1 = x
    Ts = [T0, T1]
    for k in range(2, K):
        Ts.append(2*x*Ts[-1] - Ts[-2])
    return torch.stack(Ts[:K], dim=-1)

class ChebyshevQuantile(nn.Module):
    """
    params -> (Q(u_j), q(u_j)) on a fixed u-grid, monotone by construction.
    params shape: (B, H, 2+K) = [b, log_s, a_0..a_{K-1}]
    """
    def __init__(self, K: int, u_grid: torch.Tensor, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.K = K
        self.eps = eps
        self.normalize = normalize

        # buffers move with .to(device)
        self.register_buffer("u", u_grid)                       # (J,)
        self.register_buffer("du", u_grid[1:] - u_grid[:-1])    # (J-1,)
        self.register_buffer("T", chebyshev_basis(u_grid, K))   # (J,K)

        # trapezoid weights on u for ∫_0^1 f(u) du
        du = self.du
        wu = torch.zeros_like(u_grid)
        wu[0] = du[0] / 2
        wu[-1] = du[-1] / 2
        wu[1:-1] = (du[:-1] + du[1:]) / 2
        self.register_buffer("wu", wu)                          # (J,)

    def forward(self, params: torch.Tensor):
        """
        returns:
          Q: (B,H,J)
          q: (B,H,J)  (quantile density)
        """
        b = params[..., 0]                           # (B,H)
        s = F.softplus(params[..., 1]) + self.eps    # (B,H)
        a = params[..., 2:]                          # (B,H,K)

        # g(u) = sum_k a_k T_k(u)
        g = torch.einsum("bhk,jk->bhj", a, self.T)    # (B,H,J)

        # q(u) > 0
        q = F.softplus(g) + self.eps                 # (B,H,J)

        # integrate q to get Q on the u grid (cumulative trapezoid)
        q_mid = 0.5 * (q[..., 1:] + q[..., :-1])     # (B,H,J-1)
        integ = torch.cumsum(q_mid * self.du, dim=-1) # (B,H,J-1)
        integ = torch.cat([torch.zeros_like(b)[..., None], integ], dim=-1)  # (B,H,J)

        if self.normalize:
            # normalize integral so that Q(0)=b and Q(1)=b+s
            total = integ[..., -1] + self.eps        # (B,H)
            Q = b[..., None] + s[..., None] * integ / total[..., None]
            q = s[..., None] * q / total[..., None]  # adjust density consistently
        else:
            Q = b[..., None] + integ

        return Q, q
    
class ThresholdWeightedCRPS(nn.Module):
    def __init__(self, quantile, threshold_low, threshold_high, side="below", smooth_h=0.0):
        super().__init__()
        self.quantile = quantile
        self.side = side            # "below", "above", "two_sided"
        if self.side == "two_sided":
            self.threshold = (threshold_low, threshold_high)  # float OR (r_low, r_high)
        elif self.side in ["below", "above"]:
            self.threshold = threshold_low if self.side == "below" else threshold_high  # float
        self.smooth_h = smooth_h

    def weight(self, z: torch.Tensor):
        h = self.smooth_h

        if self.side == "two_sided":
            rL, rU = self.threshold  # (negative, positive)
            if h and h > 0:
                wL = torch.sigmoid((rL - z) / h)   # ~1 if z <= rL
                wU = torch.sigmoid((z - rU) / h)   # ~1 if z >= rU
                return wL + wU
            else:
                return (z <= rL).float() + (z >= rU).float()

        r = float(self.threshold)
        if h and h > 0:
            if self.side == "below":
                return torch.sigmoid((r - z) / h)
            else:
                return torch.sigmoid((z - r) / h)
        else:
            if self.side == "below":
                return (z <= r).float()
            else:
                return (z >= r).float()

    def forward(self, params: torch.Tensor, y: torch.Tensor):
        Q, q = self.quantile(params)                      # (B,H,J)
        I = (y.unsqueeze(-1) <= Q).float()                # (B,H,J)
        wQ = self.weight(Q)                               # (B,H,J)
        u = self.quantile.u.view(1, 1, -1)                # (1,1,J)

        integrand = wQ * (u - I).pow(2) * q
        loss_bh = torch.sum(integrand * self.quantile.wu.view(1,1,-1), dim=-1)
        return loss_bh.mean()