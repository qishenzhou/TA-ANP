from model._model_register import ModelRegistry
import torch.nn.functional as F
from layers.attention import MultiHeadQueryAttention, MultiHeadAttention
from torch_geometric.data import Data
from typing import Optional
from einops import rearrange, repeat
from layers.encoders import TemporalEncoder2
import torch
import torch.nn as nn
from layers.prob_out import NormalInvGamma, NormalOutput, SortMultiOutput, MCDropout, GammaOutput


def madrid_x_gather(batch, not_static, poi, not_hit, not_speed, seq_len):
    static = None if not_static else batch.static_aux[:, 0]
    poi_static = None if not poi else batch.poi_aux[:, 0]

    dynamic = batch.dynamic_aux[..., [0, 3]][:, :seq_len]
    hit, speed = dynamic.split(1, dim=-1)
    hit = None if not_hit else batch.dynamic_aux[..., 0:1][:, :seq_len]
    speed = None if not_speed else batch.dynamic_aux[..., 3:4][:, :seq_len]
    time_enc = batch.time_enc
    return static, poi_static, hit, speed, time_enc


class TomEncoder(nn.Module):
    """Encode each road segment's floating-car features with residual linear layers."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = rearrange(x, "B t c -> B (t c)")
        out = F.relu(self.norm1(self.linear1(x)))

        res1 = out
        out = self.norm2(self.linear2(out))
        out = F.relu(out + res1)

        res2 = out
        out = self.norm3(self.linear3(out))
        out = F.relu(out + res2)
        return self.linear4(out)


class ContextSetEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        self.lin4 = nn.Linear(hidden_channels, out_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.mc_drop = MCDropout(p=0.1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.mc_drop(out)
        res1 = out
        out = self.lin2(out)
        out = self.norm2(out)
        out = out + res1
        out = F.relu(out)
        out = self.mc_drop(out)
        res2 = out
        out = self.lin3(out)
        out = self.norm3(out)
        out = out + res2
        out = F.relu(out)
        out = self.lin4(out)
        return out


@ModelRegistry.register(name="TAANP")
class TAANP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.without_tamqm = getattr(args, 'without_tamqm', False)
        self.poi = args.use_poi
        self.not_static = not args.use_static
        self.not_hit = not args.use_hit
        self.not_speed = not args.use_speed

        self.tom_dim = 0 if (self.not_hit and self.not_speed) else 8
        self.temporal_coord_dim = 4
        self.static_dim = 0 if self.not_static else 12
        self.poi_dim = 10 if self.poi else 0

        self.y_dim = 1
        self.x_dim = 32
        self.spatial_coord_dim = self.tom_dim
        self.coord_dim = self.temporal_coord_dim + self.tom_dim + self.static_dim + self.poi_dim
        self.input_window = args.input_window_size
        self.pred_window = args.pred_window_size
        self.h_dim = args.hidden_dim
        self.z_dim = args.z_dim
        self.num_heads = args.num_heads
        self.epsilon = args.epsilon
        self.dataset = args.dataset_class.lower()
        self.output_path = args.output_path

        self.temporal_encoder = TemporalEncoder2(d_phi=self.temporal_coord_dim,
                                                 hidden=self.h_dim)
        if not (self.not_hit and self.not_speed):
            self.tom_encoder = TomEncoder(
                (int(not self.not_hit) + int(not self.not_speed)) * self.input_window,
                self.tom_dim,
                self.tom_dim)

        # Deterministic path
        self.encoder_determin = ContextSetEncoder(self.coord_dim + self.y_dim,
                                                  self.h_dim // 2, self.h_dim)
        if self.without_tamqm:
            # Shared-query attention for the TAMQM ablation.
            self.cross_attention = MultiHeadAttention(
                dim_q=self.coord_dim, dim_k=self.coord_dim, dim_v=self.h_dim,
                embed_dim=self.h_dim, num_heads=self.num_heads,
                batch_first=True, dropout=0.1, mc_drop=True)
        else:
            self.cross_attention = MultiHeadQueryAttention(
                dim_q=self.coord_dim, dim_k=self.coord_dim, dim_v=self.h_dim,
                embed_dim=self.h_dim, num_heads=self.num_heads,
                batch_first=True, num_query=4)

        # Latent path
        self.encoder_latent = ContextSetEncoder(self.coord_dim + self.y_dim,
                                               self.h_dim // 2, self.h_dim)
        self.to_distribution = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                             nn.Linear(self.h_dim, self.z_dim * 2))

        decoder_input_dim = self.coord_dim + self.h_dim + self.z_dim

        if self.output_path == 'normal':
            self.decoder = nn.Sequential(nn.Linear(decoder_input_dim, self.h_dim),
                                         nn.ReLU(), MCDropout(p=0.2),
                                         nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                         MCDropout(p=0.1),
                                         NormalOutput(self.h_dim, self.y_dim))
        elif self.output_path == 'nig':
            self.decoder = nn.Sequential(nn.Linear(decoder_input_dim, self.h_dim),
                                         nn.ReLU(), MCDropout(p=0.2),
                                         nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                         MCDropout(p=0.1),
                                         NormalInvGamma(self.h_dim, self.y_dim))
        elif self.output_path == 'sqt':
            self.decoder = nn.Sequential(nn.Linear(decoder_input_dim, self.h_dim),
                                         nn.ReLU(), MCDropout(p=0.2),
                                         nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                         MCDropout(p=0.1),
                                         SortMultiOutput(self.h_dim, self.y_dim))
        elif self.output_path == 'gamma':
            self.decoder = nn.Sequential(nn.Linear(decoder_input_dim, self.h_dim),
                                         nn.ReLU(), MCDropout(p=0.2),
                                         nn.Linear(self.h_dim, self.h_dim), nn.ReLU(),
                                         MCDropout(p=0.1),
                                         GammaOutput(self.h_dim, self.y_dim))
        else:
            raise NotImplementedError(f"Output path {self.output_path} not implemented")

    def deterministic_path(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        context_y_mask: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic encoder.

        Args:
            context_x (torch.Tensor): Context features. [B,Nc, Dx]
            context_y (torch.Tensor): Context labels. [B, Nc, Dy]
            target_x (torch.Tensor): Target features. [B, Nt, Dx]
            attn_mask (torch.Tensor, optional): Attention mask. [Nt, Nc]. Defaults to None.

        Returns:
            tuple: Context representation [B, Nt, h_dim] and attention scores.
        """

        # Encode context points
        context_input = torch.cat([context_x, context_y], dim=-1)
        r = self.encoder_determin(context_input)

        if self.without_tamqm:
            return self.cross_attention(
                query=target_x, key=context_x, value=r,
                attn_mask=attn_mask, return_attn_scores=True,
                value_mask=context_y_mask)

        return self.cross_attention(query=target_x,
                                    key=context_x,
                                    value=r,
                                    attn_mask=attn_mask,
                                    return_attn_scores=True,
                                    value_mask=context_y_mask,
                                    query_type_indices=q_indices)

    def latent_path(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
    ) -> torch.distributions.Normal:
        """Encode a set of points into a Gaussian latent distribution.

        Args:
            context_x (torch.Tensor): Context features. [B, Nc, Dx]
            context_y (torch.Tensor): Context labels. [B, Nc, Dy]
        Returns:
            torch.distributions.Normal: Latent distribution with shape [B, z_dim].
        """

        # Encode context
        context_input = torch.cat([context_x, context_y], dim=-1)
        b, _, _ = context_input.shape
        context_input = rearrange(context_input, 'b n d -> (b n) d')
        s = self.encoder_latent(context_input)
        s = rearrange(s, '(b n) d -> b n d', b=b)

        # Mean pooling
        s = torch.mean(s, dim=1)

        # Get latent distribution parameters through MLP
        latent_params = self.to_distribution(s)
        mu, log_sigma = latent_params.chunk(2, dim=-1)
        sigma = self.epsilon + (1 - self.epsilon) * F.softplus(log_sigma)

        return torch.distributions.Normal(mu, sigma)

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        target_y: Optional[torch.Tensor] = None,
        context_y_mask: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Decode target points using context attention and a sampled latent state.

        Args:
            context_x (torch.Tensor): Context features. [B, Nc, Dx]
            context_y (torch.Tensor): Context labels. [B, Nc, Dy]
            target_x (torch.Tensor): Target features. [B, Nt, Dx]
            target_y (torch.Tensor, optional): Target labels. [B, Nt, Dy]. Defaults to None.
            attn_mask (torch.Tensor, optional): Attention mask. [Nt, Nc]. Defaults to None.

        Returns:
            tuple:
                - Decoder output, with node and time dimensions restored.
                - Prior distribution [B, z_dim].
                - Posterior distribution [B, z_dim].
                - Attention scores.
        """

        r_cross, att_scores = self.deterministic_path(
            context_x, context_y, target_x, attn_mask, context_y_mask, q_indices)
        prior = self.latent_path(context_x, context_y)
        # The target-conditioned posterior is used by the ELBO.
        posterior = self.latent_path(target_x, target_y) if target_y is not None else prior
        z = posterior.rsample() if self.training else prior.rsample()
        z = z.unsqueeze(1).repeat(1, target_x.size(1), 1)

        decoder_input = torch.cat([target_x, r_cross, z], dim=-1)
        decoder_input = rearrange(decoder_input,
                                  'b (n t) d -> (b n) t d',
                                  t=self.pred_window + self.input_window)
        decoder_output = self.decoder(decoder_input)

        return decoder_output, prior, posterior, att_scores

    def sampling(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        target_y: Optional[torch.Tensor] = None,
        context_y_mask: Optional[torch.Tensor] = None,
        q_indices: Optional[torch.Tensor] = None,
        num_sampling: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the predictive mean and aleatoric/epistemic standard deviations."""

        pred_mu_list = []
        pred_sigma_list = []
        for _ in range(num_sampling):
            prior = self.latent_path(context_x, context_y)
            z = prior.rsample()
            z = z.unsqueeze(1).repeat(1, target_x.size(1), 1)
            r_cross, _ = self.deterministic_path(
                context_x, context_y, target_x, attn_mask, context_y_mask, q_indices)
            decoder_input = torch.cat([target_x, r_cross, z], dim=-1)
            decoder_input = rearrange(decoder_input,
                                      'b (n t) d -> (b n) t d',
                                      t=self.pred_window + self.input_window)
            decoder_output = self.decoder(decoder_input)
            pred_mu, pred_sigma = decoder_output.loc, decoder_output.scale
            pred_mu_list.append(pred_mu)
            pred_sigma_list.append(pred_sigma)
        pred_mu = torch.stack(pred_mu_list, dim=0)
        pred_sigma = torch.stack(pred_sigma_list, dim=0)
        mu = pred_mu.mean(dim=0)
        epist = pred_mu.std(dim=0)
        alea = torch.sqrt(
            (pred_sigma**2).mean(dim=0))

        return mu, alea, epist

    def __call__(self,
                 batch: Data,
                 num_sampling: int = 0,
                 return_attn_scores: bool = False):
        first_graph = batch.to_data_list()[0]
        b = len(batch.t_id)
        assert (first_graph.label_unobs_mask == batch.label_unobs_mask.view(
            len(batch.t_id), -1)).all(), "label_unobs_mask is not consistent"
        if 'madrid' in self.dataset:
            static, poi_static, hit, speed, time_enc = madrid_x_gather(
                batch, self.not_static, self.poi, self.not_hit, self.not_speed,
                self.input_window)
            target_x = self.temporal_encoder(time_enc)
            if static is not None:
                static = repeat(static,
                                'n c-> n t c',
                                t=self.input_window + self.pred_window)
                target_x = torch.cat([target_x, static], dim=-1)
            if poi_static is not None:
                poi_static = repeat(poi_static,
                                    'n c-> n t c',
                                    t=self.input_window + self.pred_window)
                target_x = torch.cat([target_x, poi_static], dim=-1)
            if not (self.not_hit and self.not_speed):
                tom = []
                if hit is not None:
                    tom.append(hit)
                if speed is not None:
                    tom.append(speed)
                tom = torch.cat(tom, dim=-1)
                tom = self.tom_encoder(tom)
                tom = repeat(tom, 'n c-> n t c', t=self.input_window + self.pred_window)
                target_x = torch.cat([target_x, tom], dim=-1)
            target_x = rearrange(target_x, '(b n) t c -> b n t c', b=len(batch.t_id))

        else:
            raise ValueError(f"Dataset {self.dataset} not supported")

        context_x = target_x[:, ~first_graph.
                             label_unobs_mask][:, :, :self.
                                               input_window]  # shape (B, N_obs, T_hist, Dx)
        b, n, t, _ = target_x.shape
        q_indices = torch.zeros((b, n, t), dtype=torch.long, device=target_x.device)
        q_indices[:, first_graph.label_unobs_mask, :self.input_window] = 0  # Unobserved history
        q_indices[:, ~first_graph.label_unobs_mask, :self.input_window] = 1  # Observed history
        q_indices[:, first_graph.label_unobs_mask, self.input_window:] = 2  # Unobserved future
        q_indices[:, ~first_graph.label_unobs_mask, self.input_window:] = 3  # Observed future
        q_indices = rearrange(q_indices, 'b n t -> b (n t)')
        target_x = rearrange(target_x, 'b n t d -> b (n t) d')
        context_x = rearrange(context_x, 'b n t d -> b (n t) d')
        target_y = batch.y
        target_y_mask = target_y.isnan()
        target_y = torch.nan_to_num(target_y, nan=0.0)

        context_y = target_y[
            ~batch.label_unobs_mask][:, :
                                     self.input_window]  # shape (B* N_obs, T_hist, Dx)
        context_y_mask = target_y_mask[~batch.label_unobs_mask][:, :self.input_window]
        context_y = rearrange(context_y, '(b n) t d -> b (n t) d', b=b)
        context_y_mask = rearrange(context_y_mask, '(b n) t d -> b (n t) d', b=b)
        target_y = rearrange(target_y, '(b n) t d -> b (n t) d', b=b)
        if num_sampling > 0:
            return self.sampling(
                context_x,
                context_y,
                target_x,
                context_y_mask=context_y_mask,
                q_indices=q_indices,
                num_sampling=num_sampling,
            )  # mu, alea, epist
        pred_dist, prior, posterior, att_scores = self.forward(
            context_x,
            context_y,
            target_x,
            target_y=target_y,
            context_y_mask=context_y_mask,
            q_indices=q_indices)

        return pred_dist, prior, posterior
