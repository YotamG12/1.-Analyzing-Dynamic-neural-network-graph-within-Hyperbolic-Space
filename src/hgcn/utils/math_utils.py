"""Math utils functions."""

import torch


def cosh(x, clamp=15):
    """
    Compute the hyperbolic cosine of x with clamping for numerical stability.

    Args:
        x (torch.Tensor): Input tensor.
        clamp (float): Clamp value for input.
    Returns:
        torch.Tensor: cosh(x) with clamped input.
    """
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    """
    Compute the hyperbolic sine of x with clamping for numerical stability.

    Args:
        x (torch.Tensor): Input tensor.
        clamp (float): Clamp value for input.
    Returns:
        torch.Tensor: sinh(x) with clamped input.
    """
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    """
    Compute the hyperbolic tangent of x with clamping for numerical stability.

    Args:
        x (torch.Tensor): Input tensor.
        clamp (float): Clamp value for input.
    Returns:
        torch.Tensor: tanh(x) with clamped input.
    """
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    """
    Compute the inverse hyperbolic cosine of x.

    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: arcosh(x).
    """
    return Arcosh.apply(x)


def arsinh(x):
    """
    Compute the inverse hyperbolic sine of x.

    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: arsinh(x).
    """
    return Arsinh.apply(x)


def artanh(x):
    """
    Compute the inverse hyperbolic tangent of x.

    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: artanh(x).
    """
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    """
    Custom autograd function for inverse hyperbolic tangent (artanh).
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for artanh.

        Args:
            ctx: Autograd context.
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: artanh(x).
        """
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for artanh.

        Args:
            ctx: Autograd context.
            grad_output (torch.Tensor): Gradient of output.
        Returns:
            torch.Tensor: Gradient of input.
        """
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    """
    Custom autograd function for inverse hyperbolic sine (arsinh).
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for arsinh.

        Args:
            ctx: Autograd context.
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: arsinh(x).
        """
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for arsinh.

        Args:
            ctx: Autograd context.
            grad_output (torch.Tensor): Gradient of output.
        Returns:
            torch.Tensor: Gradient of input.
        """
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    """
    Custom autograd function for inverse hyperbolic cosine (arcosh).
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass for arcosh.

        Args:
            ctx: Autograd context.
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: arcosh(x).
        """
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for arcosh.

        Args:
            ctx: Autograd context.
            grad_output (torch.Tensor): Gradient of output.
        Returns:
            torch.Tensor: Gradient of input.
        """
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
