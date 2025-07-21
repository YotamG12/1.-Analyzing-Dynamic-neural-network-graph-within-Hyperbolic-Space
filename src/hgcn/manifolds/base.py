"""Base manifold."""

from torch.nn import Parameter


class Manifold(object):
    """
    Abstract base class for manifold operations. Defines the interface for manifold geometry used in hyperbolic neural networks.
    """

    def __init__(self):
        """
        Initialize the manifold base class.
        """
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """
        Compute squared distance between pairs of points on the manifold.

        Args:
            p1, p2: Points on the manifold.
            c (float): Curvature parameter.
        Returns:
            float or tensor: Squared distance.
        """
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """
        Convert Euclidean gradient to Riemannian gradient.

        Args:
            p: Point on the manifold.
            dp: Euclidean gradient.
            c (float): Curvature parameter.
        Returns:
            Riemannian gradient.
        """
        raise NotImplementedError

    def proj(self, p, c):
        """
        Project point p onto the manifold.

        Args:
            p: Point to project.
            c (float): Curvature parameter.
        Returns:
            Projected point.
        """
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """
        Project vector u onto the tangent space at point p.

        Args:
            u: Vector to project.
            p: Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            Projected vector.
        """
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """
        Project vector u onto the tangent space at the origin.

        Args:
            u: Vector to project.
            c (float): Curvature parameter.
        Returns:
            Projected vector.
        """
        raise NotImplementedError

    def expmap(self, u, p, c):
        """
        Exponential map of vector u at point p.

        Args:
            u: Vector in tangent space.
            p: Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            Point on the manifold.
        """
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """
        Logarithmic map of point p1 at point p2.

        Args:
            p1, p2: Points on the manifold.
            c (float): Curvature parameter.
        Returns:
            Vector in tangent space.
        """
        raise NotImplementedError

    def expmap0(self, u, c):
        """
        Exponential map of vector u at the origin.

        Args:
            u: Vector in tangent space.
            c (float): Curvature parameter.
        Returns:
            Point on the manifold.
        """
        raise NotImplementedError

    def logmap0(self, p, c):
        """
        Logarithmic map of point p at the origin.

        Args:
            p: Point on the manifold.
            c (float): Curvature parameter.
        Returns:
            Vector in tangent space.
        """
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """
        Mobius addition of points x and y on the manifold.

        Args:
            x, y: Points on the manifold.
            c (float): Curvature parameter.
            dim (int): Dimension along which to add.
        Returns:
            Point on the manifold.
        """
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """
        Hyperbolic matrix-vector multiplication.

        Args:
            m: Matrix.
            x: Vector.
            c (float): Curvature parameter.
        Returns:
            Result of multiplication.
        """
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """
        Initialize random weights on the manifold.

        Args:
            w: Weights to initialize.
            c (float): Curvature parameter.
            irange (float): Initialization range.
        Returns:
            Initialized weights.
        """
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """
        Compute inner product for tangent vectors at point p.

        Args:
            p: Point on the manifold.
            c (float): Curvature parameter.
            u, v: Tangent vectors.
            keepdim (bool): Whether to keep dimensions.
        Returns:
            Inner product value.
        """
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """
        Parallel transport of vector u from x to y on the manifold.

        Args:
            x, y: Points on the manifold.
            u: Vector to transport.
            c (float): Curvature parameter.
        Returns:
            Transported vector.
        """
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """
        Parallel transport of vector u from the origin to x on the manifold.

        Args:
            x: Point on the manifold.
            u: Vector to transport.
            c (float): Curvature parameter.
        Returns:
            Transported vector.
        """
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization. Stores manifold and curvature info.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        """
        Create a new ManifoldParameter instance.

        Args:
            data: Parameter data.
            requires_grad (bool): Whether to compute gradients.
            manifold: Manifold object.
            c (float): Curvature parameter.
        Returns:
            ManifoldParameter instance.
        """
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        """
        Initialize ManifoldParameter with manifold and curvature info.

        Args:
            data: Parameter data.
            requires_grad (bool): Whether to compute gradients.
            manifold: Manifold object.
            c (float): Curvature parameter.
        """
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        """
        String representation of the ManifoldParameter.
        """
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
