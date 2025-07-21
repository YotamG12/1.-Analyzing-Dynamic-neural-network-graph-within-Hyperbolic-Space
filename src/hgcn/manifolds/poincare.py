"""Poincare ball manifold."""

import torch
from hgcn.manifolds.base import Manifold
from hgcn.utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoincareBall manifold class for hyperbolic geometry operations.
    Implements all required methods for GNNs in hyperbolic space.
    """

    def __init__(self, ):
        """
        Initialize PoincareBall manifold parameters.
        """
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        """
        Compute squared distance between two points on the Poincare ball.

        Args:
            p1, p2 (torch.Tensor): Points on the manifold.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Squared distance.
        """
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def dist0(self, p1, c, keepdim=False):
        """
        Compute distance from the origin to a point on the Poincare ball.

        Args:
            p1 (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter.
            keepdim (bool): Whether to keep dimensions.
        Returns:
            torch.Tensor: Distance value.
        """
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * p1.norm(dim=-1, p=2, keepdim=keepdim)
        )
        dist = dist_c * 2 / sqrt_c
        return dist

    def _lambda_x(self, x, c):
        """
        Compute the conformal factor lambda_x for point x.

        Args:
            x (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Lambda value.
        """
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        """
        Convert Euclidean gradient to Riemannian gradient at point p.

        Args:
            p (torch.Tensor): Point on the manifold.
            dp (torch.Tensor): Euclidean gradient.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Riemannian gradient.
        """
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        """
        Project point x onto the Poincare ball manifold.

        Args:
            x (torch.Tensor): Point to project.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Projected point.
        """
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        """
        Project vector u onto the tangent space at point p.

        Args:
            u (torch.Tensor): Vector to project.
            p (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Projected vector.
        """
        return u

    def proj_tan0(self, u, c):
        """
        Project vector u onto the tangent space at the origin.

        Args:
            u (torch.Tensor): Vector to project.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Projected vector.
        """
        return u

    def expmap(self, u, p, c):
        """
        Exponential map of vector u at point p on the Poincare ball.

        Args:
            u (torch.Tensor): Vector in tangent space.
            p (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Point on the manifold.
        """
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        """
        Logarithmic map of point p2 at point p1 on the Poincare ball.

        Args:
            p1, p2 (torch.Tensor): Points on the manifold.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Vector in tangent space.
        """
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        """
        Exponential map of vector u at the origin.

        Args:
            u (torch.Tensor): Vector in tangent space.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Point on the manifold.
        """
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        """
        Logarithmic map of point p at the origin.

        Args:
            p (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Vector in tangent space.
        """
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        """
        Mobius addition of points x and y on the Poincare ball.

        Args:
            x, y (torch.Tensor): Points on the manifold.
            c (float): Curvature parameter.
            dim (int): Dimension along which to add.
        Returns:
            torch.Tensor: Result of Mobius addition.
        """
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        """
        Hyperbolic matrix-vector multiplication on the Poincare ball.

        Args:
            m (torch.Tensor): Matrix.
            x (torch.Tensor): Vector.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Result of multiplication.
        """
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        
        # ✅ FIXED: use boolean tensor
        cond = (mx == 0).prod(-1, keepdim=True).bool()
        
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res


    def init_weights(self, w, c, irange=1e-5):
        """
        Initialize random weights on the Poincare ball manifold.

        Args:
            w (torch.Tensor): Weights to initialize.
            c (float): Curvature parameter.
            irange (float): Initialization range.
        Returns:
            torch.Tensor: Initialized weights.
        """
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        """
        Compute the gyration operation for Mobius addition.

        Args:
            u, v, w (torch.Tensor): Points/vectors on the manifold.
            c (float): Curvature parameter.
            dim (int): Dimension along which to operate.
        Returns:
            torch.Tensor: Result of gyration.
        """
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        """
        Compute inner product for tangent vectors at point x on the Poincare ball.

        Args:
            x (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter.
            u, v (torch.Tensor): Tangent vectors.
            keepdim (bool): Whether to keep dimensions.
        Returns:
            torch.Tensor: Inner product value.
        """
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        """
        Parallel transport of vector u from x to y on the Poincare ball.

        Args:
            x, y (torch.Tensor): Points on the manifold.
            u (torch.Tensor): Vector to transport.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Transported vector.
        """
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        """
        Alternate parallel transport of vector u from x to y on the Poincare ball.

        Args:
            x, y (torch.Tensor): Points on the manifold.
            u (torch.Tensor): Vector to transport.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Transported vector.
        """
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        """
        Parallel transport of vector u from the origin to x on the Poincare ball.

        Args:
            x (torch.Tensor): Point on the manifold.
            u (torch.Tensor): Vector to transport.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Transported vector.
        """
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        """
        Map a point from the Poincare ball to the hyperboloid model.

        Args:
            x (torch.Tensor): Point on the Poincare ball.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Point on the hyperboloid.
        """
        K = 1. / c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)
