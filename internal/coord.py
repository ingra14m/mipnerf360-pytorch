# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools for manipulating coordinate spaces and distances along rays."""

import torch
from internal import math


def contract(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    eps = torch.finfo(torch.float32).eps
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.max(eps, torch.sum(x ** 2, dim=-1, keepdims=True))
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z


def inv_contract(z):
    """The inverse of contract()."""
    eps = torch.finfo(torch.float32).eps
    # Clamping to eps prevents non-finite gradients when z == 0.
    z_mag_sq = torch.max(eps, torch.sum(z ** 2, dim=-1, keepdims=True))
    x = torch.where(z_mag_sq <= 1, z, z / (2 * torch.sqrt(z_mag_sq) - z_mag_sq))
    return x


# def track_linearize(fn, mean, cov):
#   """Apply function `fn` to a set of means and covariances, ala a Kalman filter.
#
#   We can analytically transform a Gaussian parameterized by `mean` and `cov`
#   with a function `fn` by linearizing `fn` around `mean`, and taking advantage
#   of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
#   https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).
#
#   Args:
#     fn: the function applied to the Gaussians parameterized by (mean, cov).
#     mean: a tensor of means, where the last axis is the dimension.
#     cov: a tensor of covariances, where the last two axes are the dimensions.
#
#   Returns:
#     fn_mean: the transformed means.
#     fn_cov: the transformed covariances.
#   """
#   if (len(mean.shape) + 1) != len(cov.shape):
#     raise ValueError('cov must be non-diagonal')
#   fn_mean, lin_fn = jax.linearize(fn, mean)
#
#   fn_cov = torch.vmap(lin_fn, -1, -2)(torch.vmap(lin_fn, -1, -2)(cov))
#   return fn_mean, fn_cov
# TODO may not be correctly running
@torch.no_grad()
def track_linearize(fn, mean, std):
    """Apply function `fn` to a set of means and covariances, ala a Kalman filter.

  We can analytically transform a Gaussian parameterized by `mean` and `cov`
  with a function `fn` by linearizing `fn` around `mean`, and taking advantage
  of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
  https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).

  Args:
    fn: the function applied to the Gaussians parameterized by (mean, cov).
    mean: a tensor of means, where the last axis is the dimension.
    std: a tensor of covariances, where the last two axes are the dimensions.

  Returns:
    fn_mean: the transformed means.
    fn_cov: the transformed covariances.
  """
    # if fn == 'contract':
    #     fn = contract_mean_jacobi
    # else:
    #     raise NotImplementedError

    pre_shape = mean.shape[:-1]
    mean = mean.reshape(-1, 3)
    std = std.reshape(-1)

    def contract_mean_std(x, std):
        eps = torch.finfo(x.dtype).eps
        # eps = 1e-3
        # Clamping to eps prevents non-finite gradients when x == 0.
        x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
        x_mag_sqrt = torch.sqrt(x_mag_sq)
        mask = x_mag_sq <= 1
        z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
        det = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2))[..., 0]

        std = torch.where(mask[..., 0], std, (det ** (1 / x.shape[-1])) * std)
        return z, std

    mean, std = contract_mean_std(mean, std)  # calculate det explicitly by using eigenvalues

    mean = mean.reshape(*pre_shape, 3)
    std = std.reshape(*pre_shape)
    return mean, std


# def track_linearize(fn, mean, cov):
#   """Apply function `fn` to a set of means and covariances, ala a Kalman filter.
#
#   Args:
#       fn: the function applied to the Gaussians parameterized by (mean, cov).
#       mean: a tensor of means, where the last axis is the dimension.
#       cov: a tensor of covariances, where the last two axes are the dimensions.
#
#   Returns:
#       fn_mean: the transformed means.
#       fn_cov: the transformed covariances.
#   """
#   if (len(mean.shape) + 1) != len(cov.shape):
#     raise ValueError('cov must be non-diagonal')
#
#   # Linearize fn around mean
#   mean.requires_grad = True
#   y = fn(mean)
#   jacobian = torch.autograd.functional.jacobian(fn, mean)
#   lin_fn = lambda x: y + torch.matmul(jacobian, (x - mean).unsqueeze(-1)).squeeze(-1)
#
#   # Apply lin_fn to cov
#   fn_mean = lin_fn(mean)
#   fn_cov = torch.matmul(jacobian, torch.matmul(cov, jacobian.transpose(-1, -2)))
#
#   return fn_mean, fn_cov


def construct_ray_warps(fn, t_near, t_far):
    """Construct a bijection between metric distances and normalized distances.

    See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
    detailed explanation.

    Args:
      fn: the function to ray distances.
      t_near: a tensor of near-plane distances.
      t_far: a tensor of far-plane distances.

    Returns:
      t_to_s: a function that maps distances to normalized distances in [0, 1].
      s_to_t: the inverse of t_to_s.
    """
    if fn is None:
        fn_fwd = lambda x: x
        fn_inv = lambda x: x
    elif fn == 'piecewise':
        # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
        fn_fwd = lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x)
        fn_inv = lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x))
    # elif fn == 'power_transformation':
    #     fn_fwd = lambda x: power_transformation(x * 2, lam=lam)
    #     fn_inv = lambda y: inv_power_transformation(y, lam=lam) / 2
    else:
        inv_mapping = {
            'reciprocal': torch.reciprocal,
            'log': torch.exp,
            'exp': torch.log,
            'sqrt': torch.square,
            'square': torch.sqrt
        }
        fn_fwd = fn
        fn_inv = inv_mapping[fn.__name__]

    s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
    t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
    s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t


def expected_sin(mean, var):
    """Compute the mean of sin(x), x ~ N(mean, var)."""
    return torch.exp(-0.5 * var) * math.safe_sin(mean)  # large var -> small value.


def integrated_pos_enc(mean, var, min_deg, max_deg):
    """Encode `x` with sinusoids scaled by 2^[min_deg, max_deg).

    Args:
      mean: tensor, the mean coordinates to be encoded
      var: tensor, the variance of the coordinates to be encoded.
      min_deg: int, the min degree of the encoding.
      max_deg: int, the max degree of the encoding.

    Returns:
      encoded: torch.ndarray, encoded variables.
    """
    scales = 2 ** torch.arange(min_deg, max_deg)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None] ** 2, shape)

    return expected_sin(
        torch.cat([scaled_mean, scaled_mean + 0.5 * torch.pi], dim=-1),
        torch.cat([scaled_var] * 2, dim=-1))


def lift_and_diagonalize(mean, cov, basis):
    """Project `mean` and `cov` onto basis and diagonalize the projected cov."""
    fn_mean = torch.matmul(mean, basis)
    fn_cov_diag = torch.sum(basis * torch.matmul(cov, basis), dim=-2)
    return fn_mean, fn_cov_diag


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg)
    shape = x.shape[:-1] + (-1,)
    scaled_x = torch.reshape((x[..., None, :] * scales[:, None]), shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat
