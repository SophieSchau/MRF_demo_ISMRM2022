"""
Implementations of various layers and networks for 5-D data [N, H, W, D, C].

Written by Christopher M. Sandino (sandino@stanford.edu), 2020.
Modified by Siddharth Srinivasan (ssi@mit.edu), 2021
"""


import torch
from   torch import nn


class Activation(nn.Module):
  """
  A generic class for activation layers.
  """
  def __init__(self, type):
    super(Activation, self).__init__()
    if type == 'none':
      self.activ = None
    elif type == 'relu':
      self.activ = nn.ReLU(inplace=True)
    elif type == 'leaky_relu':
      self.activ = nn.LeakyReLU(inplace=True)
    else:
      raise ValueError('Invalid activation type: %s' % type)

  def forward(self, input):
    if self.activ == None:
      return input
    return self.activ(input)


class Conv3d(nn.Module):
  """
  A simple 3D convolutional operator.
  """
  def __init__(self, in_chans, out_chans, kernel_size):
    super(Conv3d, self).__init__()

    # Force padding such that the shapes of input and output match
    padding = (kernel_size - 1) // 2
    self.conv = nn.Conv3d(in_chans, out_chans, kernel_size, padding=padding)

  def forward(self, input):
    return self.conv(input)


class ConvBlock(nn.Module):
  """
  A 3D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

  Based on implementation described by:
    K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
  """

  def __init__(self, in_chans, out_chans, kernel_size, act_type='relu'):
    """
    Args:
      in_chans (int): Number of channels in the input.
      out_chans (int): Number of channels in the output.
    """
    super(ConvBlock, self).__init__()

    self.name = 'Conv3D'

    self.in_chans  = in_chans
    self.out_chans = out_chans

    # Define normalization and activation layers
    activation  = Activation(act_type)
    convolution = Conv3d(in_chans, out_chans, kernel_size=kernel_size)

    # Define forward pass (pre-activation)
    self.layers = nn.Sequential(activation, convolution)


  def forward(self, input):
    """
    Args:
      input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

    Returns:
      (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
    """
    return self.layers(input)

  def __repr__(self):
    return f'{self.name}(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ResBlock(nn.Module):
  """
  A ResNet block that consists of two convolutional layers followed by a residual connection.
  """

  def __init__(self, chans, kernel_size, act_type='relu'):
    """
    Args:
      chans (int): Number of channels.
    """
    super(ResBlock, self).__init__()

    self.layers = nn.Sequential(
      ConvBlock(chans, chans, kernel_size, act_type=act_type),
      ConvBlock(chans, chans, kernel_size, act_type=act_type)
    )

  def forward(self, input):
    """
    Args:
      input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

    Returns:
      (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
    """
    return self.layers(input) + input


class ResNet(nn.Module):
  """
  Prototype for 3D ResNet architecture.
  """
  def __init__(self, num_resblocks, in_chans, out_chans, num_features, kernel_size, act_type='relu'):
    """
    Args:
      num_resblocks (int): Number of residual blocks.
      in_chans (int): Input number of channels.
      num_features (int): in_chans gets expanded to this.
      kernel_size (int): Convolution kernel size.
      act_type (str): Activation type.
    """
    super(ResNet, self).__init__()

    # Declare initial conv layer
    self.init_layer = ConvBlock(in_chans, num_features, kernel_size, act_type='none')

    # Declare ResBlock layers
    self.res_blocks = nn.ModuleList([])
    for _ in range(num_resblocks):
      self.res_blocks += [ResBlock(num_features, kernel_size, act_type=act_type)]

    # Declare final conv layer
    self.final_layer = ConvBlock(num_features, out_chans, kernel_size, act_type=act_type)


  def forward(self, input):
    """
    Args:
      input (torch.Tensor): Input tensor of shape [N, C, T, Y, X]

    Returns:
      (torch.Tensor): Output tensor of shape [N, C, T, Y, X]
    """
    # Perform forward pass through the network
    x = self.init_layer(input)
    y = 1 * x
    for res_block in self.res_blocks:
      y = res_block(y)
    output = self.final_layer(y + x)

    return output
