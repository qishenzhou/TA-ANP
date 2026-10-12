from ast import Tuple

from math import nan

import torch

from torch import Tensor, nn

from torch.nn import functional as F

from sklearn.metrics import r2_score

from torch.distributions import kl_divergence, Normal

from properscoring import crps_gaussian

from typing import Optional

import numpy as np

from scipy.stats import norm

def mask_gamma_nll(preds_dist: torch.distributions.Distribution,
                   targets: torch.Tensor,
                   mask: torch.Tensor,
                   reduction: str = 'mean',
                   *args,
                   **kwargs) -> torch.Tensor:
    """
    Masked Gamma Negative Log-Likelihood Loss

    Args:
        preds_dist (torch.distributions.Gamma): Gamma distribution with concentration and rate parameters
        targets (torch.Tensor): Target values with shape [batch_size, ...], must be positive
        mask (torch.Tensor): Mask of valid targets with shape [batch_size, ...]
        reduction (str): Specifies the reduction to apply: 'none', 'mean', or 'sum'

    Returns:
        torch.Tensor: Computed loss value according to the specified reduction
    """
    # Ensure positive targets, as required by the Gamma distribution.
    safe_targets = torch.where(mask, targets, torch.ones_like(targets))

    # Compute log probabilities.
    log_prob = preds_dist.log_prob(safe_targets)

    # Apply the mask to log probabilities.
    masked_log_prob = log_prob * mask.float()

    # Count valid samples.
    valid_count = mask.sum().float()

    # Return zero loss if there are no valid samples.
    if valid_count == 0:
        return torch.tensor(0.0, device=targets.device)

    # Compute the negative log likelihood.
    nll_elements = -masked_log_prob

    # Apply the requested reduction.
    if reduction == 'none':
        return nll_elements
    elif reduction == 'sum':
        return nll_elements.sum() * mask.numel() / valid_count
    elif reduction == 'mean0_sum':
        return nll_elements.mean(0).sum()
    elif reduction == 'mean':
        return nll_elements.sum() / valid_count
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def mask_nll(preds_dist: torch.distributions.Distribution,
             targets: torch.Tensor,
             mask: torch.Tensor,
             reduction: str = 'mean',
             *args,
             **kwargs) -> torch.Tensor:
    """
    Masked Gaussian Negative Log-Likelihood Loss

    Args:
        preds (tuple/list): A tuple/list containing two tensors (mean, std_dev) with shapes [batch_size, ...]
        targets (torch.Tensor): Target values with shape [batch_size, ...], may contain NaN values
        nan_mask (torch.Tensor): Mask of valid targets with shape [batch_size, ...]
        reduction (str): Specifies the reduction to apply: 'none', 'mean', or 'sum'

    Returns:
        torch.Tensor: Computed loss value according to the specified reduction
    """

    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))

    # Compute log probability using safe targets
    log_prob = preds_dist.log_prob(safe_targets)

    # Apply mask to log probabilities
    masked_log_prob = log_prob * mask.float()

    # Calculate number of valid samples
    valid_count = mask.sum().float()

    # Return zero loss if no valid samples
    if valid_count == 0:
        return torch.tensor(0.0, device=targets.device)

    # Calculate negative log-likelihoodshi
    nll_elements = -masked_log_prob

    # Apply specified reduction method
    if reduction == 'none':
        return nll_elements
    elif reduction == 'sum':
        return nll_elements.sum() * mask.numel() / valid_count
    elif reduction == 'mean0_sum':
        # nll_sum = nll_elements.sum(dim=tuple(range(1, nll_elements.dim())))
        # valid_sum = mask.sum(dim=tuple(range(1, mask.dim())))
        # mask_count = mask.view(mask.shape[0], -1).shape[1]
        # return (nll_sum * (mask_count / valid_sum)).mean()
        return nll_elements.mean(0).sum()
    elif reduction == 'mean':
        return nll_elements.sum() / valid_count
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def mask_beta_elbo(preds: Tuple,
                   targets,
                   mask,
                   beta,
                   reduction='mean',
                   *args,
                   **kwargs):
    preds_dist, prior, posterior = preds
    if preds_dist.scale.shape[0] != prior.scale.shape[0]:
        B = prior.scale.shape[0]
        preds_mean, preds_scale = preds_dist.mean.reshape(
            B, -1, *preds_dist.mean.shape[1:]), preds_dist.scale.reshape(
                B, -1, *preds_dist.scale.shape[1:])
        preds_dist = Normal(preds_mean, preds_scale)
        targets = targets.reshape(B, -1, *targets.shape[1:])
        mask = mask.reshape(B, -1, *mask.shape[1:])

    nll = mask_nll(preds_dist, targets, mask, reduction='mean0_sum')
    kl = torch.distributions.kl_divergence(posterior, prior).mean(dim=0).sum()
    return nll + kl * beta

def mask_gamma_beta_elbo(preds: Tuple,
                         targets,
                         mask,
                         beta,
                         reduction='mean',
                         *args,
                         **kwargs):
    preds_dist, prior, posterior = preds
    nll = mask_gamma_nll(preds_dist, targets, mask, reduction='mean0_sum')
    kl = torch.distributions.kl_divergence(posterior, prior).mean(dim=0).sum()
    return nll + kl * beta

def mask_mae(preds: torch.Tensor, targets: torch.Tensor,
             nan_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Calculate masked Mean Absolute Error
    """
    if torch.sum(nan_mask) == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    assert preds.shape == targets.shape, f"preds shape {preds.shape} does not match targets shape {targets.shape}"
    targets = torch.nan_to_num(targets, nan=0.0)
    abs_error = torch.abs(preds - targets)
    return torch.sum(abs_error * nan_mask) / torch.sum(nan_mask)

def mask_mse(preds: torch.Tensor,
             targets: torch.Tensor,
             nan_mask: torch.Tensor,
             no_reduce: bool = False,
             *args,
             **kwargs) -> torch.Tensor:
    """Calculate masked Mean Squared Error
    Args:
        preds: Predicted values
        targets: True values
        mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked MSE value
    """
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    # Ensure mask is same shape as inputs
    assert nan_mask.shape == targets.shape, f"Mask shape {nan_mask.shape} does not match targets shape {targets.shape}"
    targets = torch.nan_to_num(targets, nan=0.0)
    # Calculate squared error
    squared_error = torch.square(preds - targets)

    # Apply mask and calculate mean
    masked_squared_error = squared_error * nan_mask

    if no_reduce:
        return masked_squared_error
    else:
        return torch.sum(masked_squared_error) / num_valid

def mask_rmse(preds: torch.Tensor, targets: torch.Tensor,
              nan_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Calculate masked Root Mean Squared Error
    Args:
        preds: Predicted values
        targets: True values
        mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked RMSE value
    """
    return torch.sqrt(mask_mse(preds, targets, nan_mask))

def mask_mape(preds: torch.Tensor,
              targets: torch.Tensor,
              nan_mask: torch.Tensor,
              epsilon: float = 1e-6,
              *args,
              **kwargs) -> torch.Tensor:
    """Calculate masked Mean Absolute Percentage Error
    Args:
        preds: Predicted values
        targets: True values
        nan_mask: Mask values, 1 for valid, 0 for invalid
        epsilon: Small value to avoid division by zero
    Returns:
        Masked MAPE value
    """
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    # Ensure mask is same shape as inputs
    assert nan_mask.shape == targets.shape, f"Mask shape {nan_mask.shape} does not match targets shape {targets.shape}"
    targets = torch.nan_to_num(targets, nan=0.0)
    # Calculate absolute percentage error
    # Add epsilon to denominator to avoid division by zero
    abs_percentage_error = (torch.abs(
        (preds - targets) / (torch.abs(targets) + epsilon)) * 100)

    # Apply mask and calculate mean
    masked_percentage_error = abs_percentage_error * nan_mask

    return torch.sum(masked_percentage_error) / num_valid

def mask_smape(preds: torch.Tensor,
               targets: torch.Tensor,
               nan_mask: torch.Tensor,
               epsilon: float = 1e-6,
               *args,
               **kwargs) -> torch.Tensor:
    """Calculate masked Symmetric Mean Absolute Percentage Error
    Args:
        preds: Predicted values
        targets: True values
        nan_mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked SMAPE value
    """
    more_than_5_mask = targets > 5
    nan_mask = nan_mask & more_than_5_mask
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    targets = torch.nan_to_num(targets, nan=0.0)
    abs_error = torch.abs(preds - targets)
    sum_abs = (torch.abs(preds) + torch.abs(targets)) / 2 + epsilon
    masked_sape = (abs_error / sum_abs) * nan_mask

    return torch.sum(masked_sape) / num_valid

def mask_rrmse(preds: torch.Tensor, targets: torch.Tensor,
               nan_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Calculate masked Root Relative Mean Squared Error
    Args:
        preds: Predicted values
        targets: True values
        nan_mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked RRMSE value
    """
    targets = torch.nan_to_num(targets, nan=0.0)
    rrmse = mask_rmse(preds, targets,
                      nan_mask) / (targets.sum() / torch.sum(nan_mask))
    if rrmse > 100:
        print(mask_rmse(preds, targets, nan_mask))
        print(torch.sum(nan_mask))
        print(targets.sum())
        print((targets.sum() / torch.sum(nan_mask)))

        raise ValueError(f"RRMSE is too large: {rrmse}")
    return rrmse

def masked_huber(preds: torch.Tensor,
                 targets: torch.Tensor,
                 nan_mask: torch.Tensor,
                 delta: float = 1.0,
                 reduction: str = 'mean',
                 *args,
                 **kwargs) -> torch.Tensor:
    """
    Computes the Huber loss with a mask.

    Huber loss combines the advantages of L1 and L2 losses and is robust to outliers.
    The loss is calculated only for elements where the mask is True.

    Args:
        preds (torch.Tensor): The predicted output tensor from the model.
        targets (torch.Tensor): The ground truth tensor.
        mask (torch.Tensor): A boolean mask tensor with the same shape as preds and targets.
                             True indicates that the loss for the corresponding element should be computed,
                             False indicates it should be ignored.
        delta (float): The threshold parameter for the Huber loss. Errors with an absolute
                       value less than delta use the L2 loss; errors greater than delta
                       use a modified L1 loss to ensure continuity. Defaults to 1.0.
        reduction (str): Specifies how to aggregate the computed losses. Options are:
                         'mean': Compute the mean of the losses for the masked elements.
                         'sum': Compute the sum of the losses for the masked elements.
                         'none': Return the element-wise losses (elements outside the mask will be 0).
                         Defaults to 'mean'.

    Returns:
        torch.Tensor: The computed loss value (scalar, unless reduction='none').

    Raises:
        ValueError: If preds, targets, and mask do not have the same shape.
        ValueError: If an unsupported reduction type is provided.
    """
    # Ensure the mask is boolean type
    if not mask.dtype == torch.bool:
        mask = mask.bool()

    # Verify that input tensors have the same shape (as confirmed by user)
    if not (preds.shape == targets.shape == mask.shape):
        raise ValueError("preds, targets, and mask must have the same shape.")
    targets = torch.nan_to_num(targets, nan=0.0)
    # --- Compute element-wise Huber loss using torch.nn.functional ---
    # Calculate Huber loss for all elements without reduction first.
    elementwise_loss = F.huber_loss(preds,
                                    targets,
                                    reduction='none',
                                    delta=delta)

    # Apply the mask: zero out losses for elements where mask is False.
    # Convert boolean mask to float (True->1.0, False->0.0) for multiplication.
    masked_loss = elementwise_loss * nan_mask.float()

    # Aggregate the losses based on the reduction parameter
    if reduction == 'mean':
        # Calculate the mean loss only over the masked elements.
        # Numerator: sum of losses for selected elements.
        # Denominator: number of selected elements.
        num_masked_elements = nan_mask.sum()
        if num_masked_elements == 0:
            # Avoid division by zero if the mask is all False. Return 0 loss.
            return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
        # Compute the mean
        return masked_loss.sum() / num_masked_elements
    elif reduction == 'sum':
        # Compute the sum of losses for selected elements.
        return masked_loss.sum()
    elif reduction == 'none':
        # Return the tensor of element-wise losses (0 where mask is False).
        return masked_loss
    else:
        raise ValueError(
            f"Unsupported reduction type: {reduction}. Choose 'mean', 'sum', or 'none'."
        )

def mask_r2(preds: torch.Tensor, targets: torch.Tensor, nan_mask: torch.Tensor,
            *args, **kwargs) -> float:
    """Calculate masked R-squared score
    Args:
        preds: Predicted values
        targets: True values
        nan_mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked R-squared score
    """
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    # Ensure mask is same shape as inputs
    assert nan_mask.shape == targets.shape, f"Mask shape {nan_mask.shape} does not match targets shape {targets.shape}"

    # Calculate R-squared score
    r2 = r2_score(y_pred=preds[nan_mask].cpu().numpy(),
                  y_true=targets[nan_mask].cpu().numpy())
    return torch.tensor(r2)

def mask_crps(preds: torch.Tensor, targets: torch.Tensor,
              nan_mask: torch.Tensor, preds_sigma: torch.Tensor, *args,
              **kwargs) -> torch.Tensor:
    """Calculate masked CRPS
    Args:
        preds: Predicted values
        targets: True values
        nan_mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked CRPS value
    """
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    if preds.shape != preds_sigma.shape:
        return mask_sort_crps(
            preds_sigma, targets, nan_mask
        )  # here preds_sigma is assumes to save all quantiles of preds
    targets = torch.nan_to_num(targets, nan=0.0)
    crps = torch.from_numpy(
        crps_gaussian(targets.cpu().numpy(),
                      preds.cpu().numpy(),
                      preds_sigma.cpu().numpy())).to(preds.device)
    crps = crps * nan_mask
    return torch.sum(crps) / num_valid

def mask_gamma_crps_analytic(preds_dist: torch.distributions.Distribution,
                             targets: torch.Tensor, nan_mask: torch.Tensor,
                             *args, **kwargs) -> torch.Tensor:
    """Calculate masked CRPS for Gamma distribution using analytic formula

    Args:
        preds_dist: Gamma distribution object
        targets: True values
        nan_mask: Mask values, 1 for valid, 0 for invalid
    Returns:
        Masked CRPS value
    """
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

    # Handle NaN values.
    targets = torch.nan_to_num(targets, nan=1.0)

    # Extract distribution parameters.
    alpha = preds_dist.concentration
    beta = preds_dist.rate
    y = targets

    # Apply the mask.
    valid_alpha = alpha[nan_mask]
    valid_beta = beta[nan_mask]
    valid_y = y[nan_mask]

    if len(valid_y) == 0:
        return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

    # Construct a valid Gamma distribution.
    valid_gamma_dist = torch.distributions.Gamma(valid_alpha, valid_beta)

    # Compute the CDF.
    F_y = valid_gamma_dist.cdf(valid_y)

    # Compute F_{α+1,β}(y).
    gamma_alpha_plus_1 = torch.distributions.Gamma(valid_alpha + 1, valid_beta)
    F_alpha_plus_1 = gamma_alpha_plus_1.cdf(valid_y)

    # Analytical CRPS formula.
    term1 = valid_y * (2 * F_y - 1)
    term2 = (valid_alpha / valid_beta) * (2 * F_alpha_plus_1 - 1)

    # Constant term.
    from torch import lgamma
    log_term3 = lgamma(valid_alpha + 0.5) - lgamma(
        valid_alpha) - 0.5 * torch.log(torch.tensor(torch.pi))
    term3 = torch.exp(log_term3) / valid_beta

    crps_values = term1 - term2 - term3

    return torch.mean(crps_values)

def mask_picp_discrete(preds: torch.Tensor, targets: torch.Tensor,
                       nan_mask: torch.Tensor, confidence_level):

    alpha = 1 - confidence_level
    low, high = alpha / 2, (1 - alpha / 2)
    lower_bound, upper_bound = torch.quantile(preds,
                                              torch.tensor(
                                                  [low, high],
                                                  device=preds.device,
                                                  dtype=preds.dtype),
                                              dim=-1)

    # Check interval coverage.
    in_interval = (targets >= lower_bound) & (targets <= upper_bound)
    valid_in_interval = in_interval & nan_mask

    # Compute PICP.
    picp = valid_in_interval.sum() / nan_mask.sum()
    return picp

def mask_gamma_pred_interval_coverage_prob(
        preds_dist: torch.distributions.Distribution,
        targets: torch.Tensor,
        nan_mask: torch.Tensor,
        confidence_level=0.95,
        *args,
        **kwargs) -> torch.Tensor:
    """
    Calculate Prediction Interval Coverage Probability (PICP) for Gamma distribution

    Args:
        preds_dist (torch.distributions.Gamma): Gamma distribution with concentration and rate parameters
        targets (torch.Tensor): Target values with shape [batch_size, ...], must be positive
        nan_mask (torch.Tensor): Mask of valid targets with shape [batch_size, ...]
        confidence_level (float): Confidence level for prediction interval (default: 0.95)

    Returns:
        torch.Tensor: PICP value
    """
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

    # Extract valid targets.
    valid_targets = targets[nan_mask]

    # Compute the confidence interval of the Gamma distribution.
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    # Obtain the quantiles.
    interval_lower = preds_dist.icdf(
        torch.tensor(lower_quantile, device=targets.device))
    interval_upper = preds_dist.icdf(
        torch.tensor(upper_quantile, device=targets.device))

    # Apply the mask to the quantiles.
    valid_interval_lower = interval_lower[nan_mask]
    valid_interval_upper = interval_upper[nan_mask]

    # Compute interval coverage.
    interval_coverage = (valid_targets >= valid_interval_lower) & (
        valid_targets <= valid_interval_upper)

    return torch.sum(interval_coverage) / num_valid

def mask_pred_interval_coverage_prob(preds: torch.Tensor,
                                     targets: torch.Tensor,
                                     nan_mask: torch.Tensor,
                                     preds_sigma: torch.Tensor, *args,
                                     **kwargs) -> torch.Tensor:
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    if preds.shape != preds_sigma.shape:
        # here preds_sigma is assumes to save all quantiles of preds
        return mask_picp_discrete(preds_sigma,
                                  targets,
                                  nan_mask,
                                  confidence_level=0.95)
    targets = targets[nan_mask]
    preds = preds[nan_mask]
    preds_sigma = preds_sigma[nan_mask]
    interval_lower = preds - 1.96 * preds_sigma
    interval_upper = preds + 1.96 * preds_sigma
    interval_coverage = (targets >= interval_lower) & (targets
                                                       <= interval_upper)
    return torch.sum(interval_coverage) / num_valid

def nig_nll(gamma, v, alpha, beta, y, y_mask, *args, **kwargs):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma)**2 * y_mask + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll

def nig_reg(gamma, v, alpha, _beta, y, y_mask, *args, **kwargs):
    reg = (y - gamma).abs() * (2 * v + alpha) * y_mask
    return reg

def ker_reg(v, attn_score, *args, **kwargs):
    # att_score: [B, Head, Nt, Nc]
    return v * attn_score.detach()

def mask_evidential(dist_params,
                    y,
                    mask=1,
                    lamb1=0.01,
                    lamb2=0.18,
                    reduction=True,
                    *args,
                    **kwargs):
    y = torch.nan_to_num(y, nan=0.0)
    if len(dist_params) == 2:
        dist_params, attn_score = dist_params
    else:
        attn_score = None
    loss = nig_nll(*dist_params, y,
                   mask) + lamb1 * nig_reg(*dist_params, y, mask)
    if attn_score is not None:
        loss += lamb2 * ker_reg(dist_params[1], attn_score)
    if reduction:
        return loss.mean(dim=0).sum()
    else:
        return loss

def mask_sort_crps(preds: torch.Tensor, targets: torch.Tensor,
                   nan_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    num_valid = torch.sum(nan_mask)
    if num_valid == 0:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

    targets = torch.nan_to_num(targets, nan=0.0)
    targets = targets.unsqueeze(-1)

    sorted_preds, _ = torch.sort(preds, dim=-1)
    M = sorted_preds.shape[-1]

    # Compute the first term: E|X - y|.
    # torch.abs(sorted_preds - targets) has shape (B, T, F, M).
    # Average over dimension M to obtain a tensor of shape (B, T, F).
    term1 = torch.abs(sorted_preds - targets).mean(dim=-1)

    # Compute the second term: (1/M^2) * Σ (2i - M - 1) * x_i.
    # First construct the coefficients (2i - M - 1).
    # PyTorch uses zero-based indexing, so the formula's one-based 'i' becomes 'i_torch + 1'.
    # The coefficient becomes 2*(i_torch + 1) - M - 1 = 2*i_torch + 2 - M - 1 = 2*i_torch - M + 1.
    i = torch.arange(M, device=preds.device, dtype=preds.dtype)
    coeffs = (2 * i - M + 1)

    num_dims = sorted_preds.dim()

    # 2. Construct the new view shape.
    view_shape = [1] * (num_dims - 1) + [M]

    # 3. Apply this dynamic view.
    coeffs = coeffs.view(*view_shape)
    # Multiply by the sorted predictions and sum.
    # (B, T, F, M) * (1, 1, 1, M) -> (B, T, F, M)
    # Sum over dimension M to obtain a tensor of shape (B, T, F).
    sum_val = torch.sum(coeffs * sorted_preds, dim=-1)

    # Divide by M^2.
    term2 = sum_val / (M**2)

    # Obtain the CRPS value for each point.
    crps_per_element = term1 - term2  # Shape: (B, T, F)

    # --- 4. Apply the mask and compute the final loss ---
    # Set the CRPS of invalid points to zero.
    masked_crps = crps_per_element * nan_mask

    # Count valid points.
    num_valid_points = torch.sum(nan_mask)
    mean_loss = torch.sum(masked_crps) / num_valid_points
    return mean_loss

def mask_qice_discrete(preds, targets, nan_mask, M):
    targets_expanded = targets.unsqueeze(0)
    q_list = torch.linspace(0,
                            100,
                            steps=M + 1,
                            device=preds.device,
                            dtype=preds.dtype) / 100
    preds_quantiles = torch.quantile(preds, q_list, dim=-1)

    lower_bounds, upper_bounds = preds_quantiles[:-1], preds_quantiles[1:]
    in_intervals = (targets_expanded >= lower_bounds) & (targets_expanded
                                                         < upper_bounds)
    nan_mask_expanded = nan_mask.unsqueeze(0)
    valid_in_intervals = in_intervals & nan_mask_expanded

    coverage_counts = valid_in_intervals.reshape(M, -1).sum(dim=-1)
    below_lower_bound = (targets < preds_quantiles[0]) & nan_mask
    above_upper_bound = (targets >= preds_quantiles[-1]) & nan_mask
    coverage_counts[0] += below_lower_bound.sum()
    coverage_counts[-1] += above_upper_bound.sum()
    valid_totals = nan_mask.sum()
    coverage_rates = coverage_counts / valid_totals

    # Expected coverage under a uniform distribution.
    expected_coverage = 1.0 / M

    # QICe = mean absolute deviation from the expected coverage.
    qice = torch.abs(coverage_rates - expected_coverage).mean(dim=-1)

    return qice

def mask_qice(preds: torch.Tensor,
              targets: torch.Tensor,
              nan_mask: torch.Tensor,
              y_pred_sigma: torch.Tensor,
              M=20):
    """
    Calculate the QICE score
    Args:
        p_values_1d (np.array): 1D array containing all p-values
        M (int): Number of bins.

    Returns:
        float: The QICE score
    """
    if torch.isnan(targets).all():
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)
    if preds.shape != y_pred_sigma.shape:
        # here preds_sigma is assumes to save all quantiles of preds
        return mask_qice_discrete(y_pred_sigma, targets, nan_mask, M)

    p_values = norm.cdf(targets, loc=preds, scale=y_pred_sigma)
    counts, _ = np.histogram(p_values[nan_mask], bins=M, range=(0, 1))

    proportions = counts / len(p_values[nan_mask])

    ideal_proportion = 1.0 / M

    qice_score = np.mean(np.abs(proportions - ideal_proportion))

    return torch.tensor(qice_score, device=preds.device, dtype=preds.dtype)

def mask_gamma_qice(preds_dist: torch.distributions.Distribution,
                    targets: torch.Tensor,
                    nan_mask: torch.Tensor,
                    M=20):
    """
    Calculate the QICE score for Gamma distribution outputs

    Args:
        preds_dist (torch.distributions.Gamma): Gamma distribution with concentration and rate parameters
        targets (torch.Tensor): Target values with shape [batch_size, ...], must be positive
        nan_mask (torch.Tensor): Mask of valid targets with shape [batch_size, ...]
        M (int): Number of bins.

    Returns:
        torch.Tensor: The QICE score
    """
    if torch.isnan(targets).all():
        return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

    # Ensure positive targets.
    safe_targets = torch.where(nan_mask, targets, torch.ones_like(targets))

    # Compute cumulative Gamma probabilities (p-values).
    p_values = preds_dist.cdf(safe_targets)

    # Apply the mask to p-values.
    masked_p_values = p_values[nan_mask]

    # Return zero if there are no valid samples.
    if len(masked_p_values) == 0:
        return torch.tensor(0.0, device=targets.device, dtype=targets.dtype)

    # Convert p-values to NumPy for computation.
    p_values_np = masked_p_values.detach().cpu().numpy()

    # Compute bin statistics.
    counts, _ = np.histogram(p_values_np, bins=M, range=(0, 1))

    # Compute the proportion in each interval.
    proportions = counts / len(p_values_np)

    # Ideal proportions under a uniform distribution.
    ideal_proportion = 1.0 / M

    # Compute the QICE score (mean absolute deviation).
    qice_score = np.mean(np.abs(proportions - ideal_proportion))

    return torch.tensor(qice_score, device=targets.device, dtype=targets.dtype)

def get_loss_function(loss_name):
    loss_functions = {
        "mask_mse": mask_mse,
        "mask_rmse": mask_rmse,
        "mask_mape": mask_mape,
        "mask_smape": mask_smape,
        "mask_mae": mask_mae,
        "mask_huber": masked_huber,
        "mask_r2": mask_r2,
        "mask_crps": mask_crps,
        "mask_picb": mask_pred_interval_coverage_prob,
        # "mask_pred_interval": mask_pred_interval,
        "mask_evidential": mask_evidential,
        "mask_rrmse": mask_rrmse,
        "mask_nll": mask_nll,
        "mask_beta_elbo": mask_beta_elbo,
        "mask_qice": mask_qice,
        "mask_sort_crps": mask_sort_crps,
        "mask_gamma_beta_elbo": mask_gamma_beta_elbo
    }
    return loss_functions[loss_name]

class MadridMetrics:

    def __init__(
        self,
        look_back_window_size,
        pred_window_size,
        have_unobs=True,
        metric_names=[
            "mask_mse", "mask_rmse", "mask_mape", "mask_crps", "mask_qice"
        ],
    ):
        prefixs = []
        self.have_unobs = have_unobs
        if look_back_window_size > 0:
            if have_unobs:
                prefixs += ["unobs_hist", "obs_hist"]
            else:
                prefixs += ["obs_hist"]
            self.hist_slice = slice(None, look_back_window_size)
        if pred_window_size > 0:
            if have_unobs:
                prefixs += ["unobs_future", "obs_future"]
            else:
                prefixs += ["obs_future"]
            self.future_slice = slice(look_back_window_size,
                                      look_back_window_size + pred_window_size)
        self.metrics = {
            metrics: get_loss_function(metrics)
            for metrics in metric_names
        }
        self.log_metrics = [
            f"{prefix}/{metric}" for prefix in prefixs
            for metric in metric_names
        ]
        self.look_back_window_size = look_back_window_size
        self.pred_window_size = pred_window_size

    def __call__(self,
                 preds: Tensor,
                 targets: Tensor,
                 label_unobs_mask: Tensor,
                 preds_sigma: Optional[Tensor] = None,
                 preds_sigma_2: Optional[Tensor] = None):
        label_obs_mask = ~label_unobs_mask
        losses = {}

        if preds_sigma is None:
            self.metrics.pop("mask_crps", None)
            self.metrics.pop("mask_picb", None)
            self.metrics.pop("mask_pred_interval", None)
            self.metrics.pop("mask_qice", None)
        else:
            if preds_sigma_2 is not None:
                preds_sigma = torch.sqrt((preds_sigma**2) + (preds_sigma_2**2))
        obs = self.split_window(preds=preds,
                                targets=targets,
                                mask=label_obs_mask,
                                preds_sigma=preds_sigma)
        unobs = self.split_window(preds=preds,
                                  targets=targets,
                                  mask=label_unobs_mask,
                                  preds_sigma=preds_sigma)

        for metric_name in self.log_metrics:
            prefix, metric = metric_name.split("/")
            if prefix.startswith("unobs"):
                data = unobs[prefix.split("_")[1]]
            elif prefix.startswith("obs"):
                data = obs[prefix.split("_")[1]]
            else:
                raise ValueError(f"Unknown prefix: {prefix}")
            if metric not in [
                    "mask_crps", "mask_picb", "mask_pred_interval", "mask_qice"
            ] and preds_sigma is not None:
                data = data[:-1]
            elif preds_sigma is None and metric in [
                    "mask_crps", "mask_picb", "mask_pred_interval", "mask_qice"
            ]:
                continue
            loss = self.metrics[metric](*data)
            losses[metric_name] = loss.item()
        return losses

    def split_window(self, preds, targets, mask, preds_sigma=None):
        """Split data into input and output windows for observed/unobserved nodes"""
        r = {}
        if self.look_back_window_size > 0:
            r_hist = [
                preds[mask][:, self.hist_slice],
                targets[mask][:, self.hist_slice],
                ~torch.isnan(targets[mask][:, self.hist_slice]),
            ]
            if preds_sigma is not None:
                r_hist.append(preds_sigma[mask][:, self.hist_slice])
            r["hist"] = r_hist
        if self.pred_window_size > 0:
            r_future = [
                preds[mask][:, self.future_slice],
                targets[mask][:, self.future_slice],
                ~torch.isnan(targets[mask][:, self.future_slice]),
            ]
            if preds_sigma is not None:
                r_future.append(preds_sigma[mask][:, self.future_slice])
            r["future"] = r_future
        return r

class MadridMetrics2:

    def __init__(
        self,
        look_back_window_size,
        pred_window_size,
        have_unobs=True,
        metric_names=[
            "mask_mse", "mask_rmse", "mask_mape", "mask_crps", "mask_qice"
        ],
    ):
        prefixs = []
        self.have_unobs = have_unobs
        if look_back_window_size > 0:
            if have_unobs:
                prefixs += ["unobs_hist", "obs_hist"]
            else:
                prefixs += ["obs_hist"]
            self.hist_slice = slice(None, look_back_window_size)
        if pred_window_size > 0:
            if have_unobs:
                prefixs += ["unobs_future", "obs_future"]
            else:
                prefixs += ["obs_future"]
            self.future_slice = slice(look_back_window_size,
                                      look_back_window_size + pred_window_size)
        self.metrics = {
            metrics: get_loss_function(metrics)
            for metrics in metric_names
        }
        self.metrics.update({
            'mask_crps': mask_gamma_crps_analytic,
            'mask_picb': mask_gamma_pred_interval_coverage_prob,
            'mask_qice': mask_gamma_qice
        })
        self.metrics.pop('mask_pred_interval', None)
        self.log_metrics = [
            f"{prefix}/{metric}" for prefix in prefixs
            for metric in metric_names
        ]
        self.look_back_window_size = look_back_window_size
        self.pred_window_size = pred_window_size

    def __call__(self, preds_inverse: Tensor, targets_inverse: Tensor,
                 label_unobs_mask: Tensor, targets_not_inverse: Tensor,
                 gamma_dist: torch.distributions.Distribution):
        label_obs_mask = ~label_unobs_mask
        losses = {}

        obs = self.split_window(preds=preds_inverse,
                                targets=targets_inverse,
                                mask=label_obs_mask,
                                targets_not_inverse=targets_not_inverse,
                                gamma_dist=gamma_dist)
        unobs = self.split_window(preds=preds_inverse,
                                  targets=targets_inverse,
                                  mask=label_unobs_mask,
                                  targets_not_inverse=targets_not_inverse,
                                  gamma_dist=gamma_dist)

        for metric_name in self.log_metrics:
            prefix, metric = metric_name.split("/")
            if prefix.startswith("unobs"):
                data = unobs[prefix.split("_")[1]]
            elif prefix.startswith("obs"):
                data = obs[prefix.split("_")[1]]
            else:
                raise ValueError(f"Unknown prefix: {prefix}")
            if metric not in [
                    "mask_crps", "mask_picb", "mask_pred_interval", "mask_qice"
            ]:
                data = data[:-2]
            else:
                data = data[-2:] + data[-3:-2]  #target, gamma_dist,mask
            loss = self.metrics[metric](*data)
            losses[metric_name] = loss.item()
        return losses

    def split_window(self, preds, targets, mask, targets_not_inverse,
                     gamma_dist):
        """Split data into input and output windows for observed/unobserved nodes"""
        r = {}
        if self.look_back_window_size > 0:
            r_hist = [
                preds[mask][:, self.hist_slice],
                targets[mask][:, self.hist_slice],
                ~torch.isnan(targets[mask][:, self.hist_slice]),
            ]
            alpha = gamma_dist.concentration[mask][:, self.hist_slice]
            beta = gamma_dist.rate[mask][:, self.hist_slice]
            gamma_dist_window = torch.distributions.Gamma(alpha, beta)
            r_hist.append(gamma_dist_window)
            r_hist.append(targets_not_inverse[mask][:, self.hist_slice])
            r["hist"] = r_hist
        if self.pred_window_size > 0:
            r_future = [
                preds[mask][:, self.future_slice],
                targets[mask][:, self.future_slice],
                ~torch.isnan(targets[mask][:, self.future_slice]),
            ]
            alpha = gamma_dist.concentration[mask][:, self.future_slice]
            beta = gamma_dist.rate[mask][:, self.future_slice]
            gamma_dist_window = torch.distributions.Gamma(alpha, beta)
            r_future.append(gamma_dist_window)
            r_future.append(targets_not_inverse[mask][:, self.future_slice])
            r["future"] = r_future
        return r
