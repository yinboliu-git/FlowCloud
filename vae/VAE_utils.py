import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
import torch
import numpy as np
import harmonypy as hm
import scanpy as sc
import anndata as ad
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np

from math import exp

class PIDControl():
    """incremental PID controller"""
    def __init__(self, Kp, Ki, init_beta, min_beta, max_beta):
        """define them out of loop"""
        self.W_k1 = init_beta
        self.W_min = min_beta
        self.W_max = max_beta
        self.e_k1 = 0.0
        self.Kp = Kp
        self.Ki = Ki

    def _Kp_fun(self, Err, scale=1):
        return 1.0/(1.0 + float(scale)*exp(Err))

    def pid(self, exp_KL, kl_loss):
        """
        Incremental PID algorithm
        Input: KL_loss
        return: weight for KL divergence, beta
        """
        error_k = (exp_KL - kl_loss) * 5.   # we enlarge the error 5 times to allow faster tuning of beta
        ## comput U as the control factor
        # print(f'error_k={error_k}, self.e_k1={self.e_k1}')
        dP = self.Kp * (self._Kp_fun(error_k) - self._Kp_fun(self.e_k1))
        dI = self.Ki * error_k

        if self.W_k1 < self.W_min:
            dI = 0
        dW = dP + dI
        ## update with previous W_k1
        Wk = dW + self.W_k1
        self.W_k1 = Wk
        self.e_k1 = error_k

        ## min and max value
        if Wk < self.W_min:
            Wk = self.W_min
        if Wk > self.W_max:
            Wk = self.W_max

        return Wk, error_k


# class ImageEncoder(nn.Module):
#     def __init__(self, input_channels, hidden_dims, output_dim, activation="elu", dropout=0.):
#         super(ImageEncoder, self).__init__()
#         # hidden_dims += [4]
#         self.conv_layers = self.build_conv_layers(input_channels, hidden_dims, activation, dropout)
#         self.flatten = nn.Flatten()
#         self.fc_mu = None  # Lazy initialization
#         self.fc_var = None  # Lazy initialization
#         self.output_dim = output_dim
#
#     def build_conv_layers(self, input_channels, hidden_dims, activation, dropout):
#         layers = []
#         in_channels = input_channels
#         for out_channels in hidden_dims:
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             if activation == "relu":
#                 layers.append(nn.ReLU())
#             elif activation == "elu":
#                 layers.append(nn.ELU())
#             layers.append(nn.Dropout(p=dropout))
#             in_channels = out_channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # Through convolutional layer
#         h = self.conv_layers(x)
#         h_flat = self.flatten(h)
#
#         # Dynamically initialize fully connected layers
#         if self.fc_mu is None or self.fc_var is None:
#             # Get the size of the flattened feature map
#             input_dim = h_flat.size(1)
#             self.fc_mu = nn.Linear(input_dim, self.output_dim).to(h_flat.device)  # Ensure correct device
#             self.fc_var = nn.Linear(input_dim, self.output_dim).to(h_flat.device)
#
#         # Compute mu and var through fully connected layers
#         mu = self.fc_mu(h_flat)
#         logvar = self.fc_var(h_flat).clamp(-15, 15)
#         return mu, logvar
#
# # ImageDecoder implementation
# class ImageDecoder(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_channels, img_shape=(3, 64, 64), activation="elu", dropout=0.):
#         super(ImageDecoder, self).__init__()
#         self.img_shape = img_shape
#         self.fc = nn.Linear(input_dim, hidden_dims[0] * 8 * 8)  # Assume we restore the decoder's initial feature map to 8x8
#         self.deconv_layers = self.build_deconv_layers(hidden_dims, output_channels, activation, dropout)
#
#         # Module for upsampling to target size
#         self.upsample = nn.Upsample(size=(img_shape[1], img_shape[2]), mode='bilinear', align_corners=False)
#
#         self.downsample = nn.AdaptiveAvgPool2d(output_size=(img_shape[1], img_shape[2]))
#
#     def build_deconv_layers(self, hidden_dims, output_channels, activation, dropout):
#         layers = []
#         in_channels = hidden_dims[0]
#         for out_channels in hidden_dims[1:]:
#             layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             if activation == "relu":
#                 layers.append(nn.ReLU())
#             elif activation == "elu":
#                 layers.append(nn.ELU())
#             layers.append(nn.Dropout(p=dropout))
#             in_channels = out_channels
#         layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size=4, stride=2, padding=1))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.size(0), -1, 8, 8)  # Restore to 8x8 feature map
#         x = self.deconv_layers(x)
#         # If feature map size is larger than target size, downsample
#         if x.size(2) > self.img_shape[1] or x.size(3) > self.img_shape[2]:
#             x = self.downsample(x)
#         else:
#             x = self.upsample(x)
#         assert [i for i in x.shape[1:]] == self.img_shape
#         return torch.sigmoid(x)  # Limit output image values to [0, 1]


import torch
from torch import nn
import math
import torch.nn.functional as F
from timm.layers import DropPath  # you may need to pip install timm

def adata_img_sort_indices(adata_img):
    try:
        pixel_numbers = [int(name.split('_')[1]) for name in adata_img.var_names]
    except (ValueError, IndexError):
        raise ValueError("Unable to parse 'pixel_NUMBER' format from var_names. Please check your variable names.")

    # 2. Get indices that would sort these numbers in ascending order
    # np.argsort() returns the indices of elements in the original array after sorting
    sort_indices = np.argsort(pixel_numbers)

    # 3. Use sorting indices to slice the AnnData object
    # adata[:, sort_indices] creates a new AnnData object,
    # with all column-related data (.var, .X, .layers, etc.) arranged in the new order.
    return adata_img[:, sort_indices]

# --- 1. core modules extracted from new code ---
# ConvMLP is ConvBlock a dependency of
class ConvMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(ConvMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# this is what we will use to replace ResidualBlock new building block
class ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super(ConvBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # depthwise convolution simulating attention
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        # inject position information
        x = x + self.pos_embed(x)
        # core "attention" + residual connection
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x))))
        )
        # MLP + residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- 2. constructusenewbuilding block ModernizedVAE ---
import torch
from torch import nn
import math
import torch.nn.functional as F


# assume ConvBlock and ConvMLP classes are defined
# from timm.layers import DropPath

# class ConvMLP(nn.Module):
#     ...
# class ConvBlock(nn.Module):
#     ...
#
class ResidualVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128, img_size=32, hidden_dims=None):
        # ... (previous code unchanged) ...
        super(ResidualVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        encoder_dims = [input_channels] + hidden_dims
        self.encoder_dims = encoder_dims  # <--- 1. add a new line here，save encoder_dims
        decoder_dims = hidden_dims[::-1]

        # --- encoder (Encoder) ---
        # ... (encoder construction code unchanged) ...
        self.encoder_stages = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            stage = nn.Sequential(
                ConvBlock(dim=encoder_dims[i], mlp_ratio=4.),
                nn.Conv2d(encoder_dims[i], encoder_dims[i + 1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(encoder_dims[i + 1]),
                nn.LeakyReLU(inplace=True)
            )
            self.encoder_stages.append(stage)

        # ... (subsequent __init__ code unchanged) ...
        self.feature_map_size = img_size // (2 ** len(hidden_dims))
        self.fc_input_size = encoder_dims[-1] * (self.feature_map_size ** 2)
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_log_var = nn.Linear(self.fc_input_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.fc_input_size)
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            stage = nn.Sequential(
                nn.ConvTranspose2d(decoder_dims[i], decoder_dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(decoder_dims[i + 1]),
                nn.LeakyReLU(inplace=True),
                ConvBlock(dim=decoder_dims[i + 1], mlp_ratio=4.)
            )
            self.decoder_stages.append(stage)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(decoder_dims[-1], input_channels, kernel_size=2, stride=2),
            nn.Tanh()
        )

    # ... (encode and reparameterize method unchanged) ...
    def encode(self, x):
        for stage in self.encoder_stages:
            x = stage(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x).clamp(-15, 15)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)

        # <--- 2. modify this line of code ---
        deepest_dim = self.encoder_dims[-1]  # get deepest dimension directly from saved list
        x = x.view(-1, deepest_dim, self.feature_map_size, self.feature_map_size)

        for stage in self.decoder_stages:
            x = stage(x)
        reconstructed_x = self.final_layer(x)
        return reconstructed_x

    # ... (forward method unchanged) ...
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var

#
# import torch
# from torch import nn
# import math
# import torch.nn.functional as F
#
#
# class ResidualVAE(nn.Module):
#     def __init__(self, input_channels=3, latent_dim=128, img_size=32, hidden_dims=None):
#         """
#         apurefully connected layer (MLP) VAEversion，used fortestshowsavebaseline。
#         allconvolutional layerallalreadybyremove。
#         externalinterfacewithbeforeversioncompletelyallcompatible。
#
#         Args:
#             input_channels (int): inputgraphpixelchannelnumber。
#             latent_dim (int): latent spacedimension。
#             img_size (int): inputgraphpixeledgegrow。
#             hidden_dims (list, optional):
#                 aintegerlist，used fordefinitionMLPhiddenlayerdimension。
#                 ifas None，thendefaultas [512]。
#         """
#         super(ResidualVAE, self).__init__()
#         self.latent_dim = latent_dim
#         self.img_size = img_size
#         self.input_channels = input_channels
#
#         if hidden_dims is None:
#             hidden_dims = [512]  # asMLPdefinitionadefaulthiddenlayerstructure
#
#         # --- 1. computeflattenafterinputdimension ---
#         flattened_size = input_channels * img_size * img_size
#
#         # --- 2. constructencoder (pureMLP) ---
#         encoder_layer_dims = [flattened_size] + hidden_dims
#
#         encoder_layers = []
#         for i in range(len(encoder_layer_dims) - 1):
#             encoder_layers.extend([
#                 nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1]),
#                 nn.LeakyReLU(inplace=True)
#             ])
#         self.encoder_mlp = nn.Sequential(*encoder_layers)
#
#         # finaloutputmuandlog_varlayer
#         self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
#         self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)
#
#         # --- 3. constructdecoder (pureMLP) ---
#         decoder_layer_dims = [latent_dim] + hidden_dims[::-1]
#
#         decoder_layers = []
#         for i in range(len(decoder_layer_dims) - 1):
#             decoder_layers.extend([
#                 nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1]),
#                 nn.LeakyReLU(inplace=True)
#             ])
#         # finallyonelayerwilloutputrestoretoflattengraphpixeldimension
#         decoder_layers.append(nn.Linear(hidden_dims[0], flattened_size))
#         self.decoder_mlp = nn.Sequential(*decoder_layers)
#
#         # finalactivationfunction
#         self.final_activation = nn.Tanh()
#
#     def encode(self, x):
#         # 1. willinputgraphpixel (B, C, H, W) flattenasvector (B, C*H*W)
#         x = torch.flatten(x, start_dim=1)
#         # 2. throughMLP
#         x = self.encoder_mlp(x)
#         # 3. compute mu and log_var
#         mu = self.fc_mu(x)
#         log_var = self.fc_log_var(x).clamp(-15, 15)
#         return mu, log_var
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z):
#         # 1. throughMLP，willdiveinvectorzreconstructionasflattengraphpixelvector
#         x = self.decoder_mlp(z)
#         # 2. willvector reshape returngraphpixelshape (B, C, H, W)
#         reconstructed_x = x.view(-1, self.input_channels, self.img_size, self.img_size)
#         # 3. applicationfinalactivationfunction
#         reconstructed_x = self.final_activation(reconstructed_x)
#         return reconstructed_x
#
#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         reconstructed_x = self.decode(z)
#         return reconstructed_x, mu, log_var

class ImageEncoder(nn.Module):
    """
    An improved Image Encoder that uses strided convolutions for downsampling
    and adaptive pooling for a robust connection to the fully-connected layers.
    """
    def __init__(self, input_channels=3, latent_dim=8, img_size=32,devcie=None, dtype=None, **kwargs):
        super().__init__()
        self.model = ResidualVAE(input_channels=input_channels, latent_dim=latent_dim, img_size=img_size).to(dtype=dtype)
        self.to(devcie)
    def forward(self, x):
        mu, logvar = self.model.encode(x)
        return mu, logvar

    def get_vae_model(self):
        return self.model

class ImageDecoder(nn.Module):
    """
    An improved Image Decoder that mirrors the encoder's architecture, using
    Upsampling + Conv2d to avoid checkerboard artifacts. The final layer has no
    activation function, outputting raw pixel values (logits).
    """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, rmu):
        reconstructed_x = self.model.decode(rmu)
        return reconstructed_x


import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Modules for Cleaner Code ---
#
# class ConvBlock(nn.Module):
#     """A standard convolutional block: Conv -> BatchNorm -> Activation."""
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, activation="elu"):
#         super().__init__()
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels)
#         ]
#         if activation == "relu":
#             layers.append(nn.ReLU(inplace=True))
#         elif activation == "elu":
#             layers.append(nn.ELU(inplace=True))
#         self.block = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.block(x)
#
#
# class DeconvBlock(nn.Module):
#     """A modern deconvolutional block: Upsample -> Conv -> BatchNorm -> Activation."""
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2, activation="elu"):
#         super().__init__()
#         layers = [
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_channels)
#         ]
#         if activation == "relu":
#             layers.append(nn.ReLU(inplace=True))
#         elif activation == "elu":
#             layers.append(nn.ELU(inplace=True))
#         self.block = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.block(x)
#
#
# # --- Modernized VAE Architecture ---
# class ImageEncoder(nn.Module):
#     """
#     An improved Image Encoder that uses strided convolutions for downsampling
#     and adaptive pooling for a robust connection to the fully-connected layers.
#     """
#
#     def __init__(self, input_channels, hidden_dims, output_dim, activation="elu"):
#         super().__init__()
#
#         # Build the convolutional backbone
#         layers = []
#         in_channels = input_channels
#         for h_dim in hidden_dims:
#             layers.append(ConvBlock(in_channels, h_dim, stride=2, activation=activation))
#             in_channels = h_dim
#         self.conv_layers = nn.Sequential(*layers)
#
#         # Adaptive pooling makes the model robust to input image size
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#
#         # Fully-connected layers are now initialized statically in __init__
#         self.fc_mu = nn.Linear(hidden_dims[-1], output_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1], output_dim)
#         self.output_dim = output_dim
#
#     def forward(self, x):
#         # Pass through convolutional layers to extract features and downsample
#         h = self.conv_layers(x)
#
#         # Pool features to a fixed size and flatten
#         h = self.adaptive_pool(h)
#         h_flat = self.flatten(h)
#
#         # Calculate mu and logvar
#         mu = self.fc_mu(h_flat)
#         logvar = self.fc_var(h_flat).clamp(-15, 15)
#         return mu, logvar
#
#
# class ImageDecoder(nn.Module):
#     """
#     An improved Image Decoder that mirrors the encoder's architecture, using
#     Upsampling + Conv2d to avoid checkerboard artifacts. The final layer has no
#     activation function, outputting raw pixel values (logits).
#     """
#
#     def __init__(self, input_dim, hidden_dims, output_channels, activation="elu", start_resolution=4):
#         super().__init__()
#
#         # hidden_dims should be the reverse of the encoder's hidden_dims
#         self.hidden_dims = hidden_dims
#         self.start_res = start_resolution
#
#         # Project the latent vector `z` and reshape it into a starting feature map
#         self.fc = nn.Linear(input_dim, hidden_dims[0] * (self.start_res ** 2))
#
#         # Build the deconvolutional (upsampling) backbone
#         layers = []
#         in_channels = hidden_dims[0]
#         for h_dim in hidden_dims[1:]:
#             layers.append(DeconvBlock(in_channels, h_dim, activation=activation))
#             in_channels = h_dim
#         self.deconv_layers = nn.Sequential(*layers)
#
#         # Final convolutional layer to produce the output image
#         # This layer has no activation function.
#         self.final_conv = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(hidden_dims[-1], output_channels, kernel_size=3, stride=1, padding=1),
#             # NO FINAL ACTIVATION (e.g., Sigmoid or Tanh)
#         )
#
#     def forward(self, x):
#         # Project and reshape latent vector
#         x = self.fc(x)
#         x = x.view(x.size(0), self.hidden_dims[0], self.start_res, self.start_res)
#         # Pass through upsampling layers
#         x = self.deconv_layers(x)
#         # Generate the final image
#         reconstructed_image = self.final_conv(x)
#         return F.sigmoid(reconstructed_image)

class DenseEncoder(nn.Module):
    def __init__(self,hidden_dims, output_dim, input_dim=None, activation="relu", dropout=0, dtype=torch.float32, norm="batchnorm"):
        super(DenseEncoder, self).__init__()
        if input_dim is not None:
            input_d = [input_dim] + hidden_dims
        else:
            input_d = hidden_dims
        self.layers = buildNetwork(input_d, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], output_dim)
        self.enc_var = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        logvar = self.enc_var(h).clamp(-15, 15)
        return mu, logvar

import torch
from torch import nn


class Pos_GP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Pos_GP, self).__init__()
        output_dim = input_dim
        self.f1 = nn.Linear(input_dim, hidden_dims)
        self.enc_mu = nn.Linear(hidden_dims, output_dim)
        self.enc_var = nn.Linear(hidden_dims, output_dim)
        self.initialize_weights_to_zero()

    def initialize_weights_to_zero(self):
        """willf1andf2weightandbiasplaceallinitializeas0"""
        print("Initializing Pos_GP residual layers to zero.")
        nn.init.zeros_(self.f1.weight)
        if self.f1.bias is not None:
            nn.init.zeros_(self.f1.bias)

        nn.init.zeros_(self.enc_mu.weight)
        if self.enc_mu.bias is not None:
            nn.init.zeros_(self.enc_mu.bias)

        nn.init.zeros_(self.enc_var.weight)
        if self.enc_var.bias is not None:
            nn.init.zeros_(self.enc_var.bias)

    def forward(self, x):
        h = F.softmax(self.f1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_var(h).clamp(-15, 15)
        return mu, logvar


def buildNetwork(layers, network="decoder", activation="relu", dropout=0., dtype=torch.float32, norm="batchnorm"):
    net = []
    if network == "encoder" and dropout > 0:
        net.append(nn.Dropout(p=dropout))
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i]))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        elif activation == "softmax":
            net.append(nn.Softmax(dim=layers[i]))
        elif activation == "softplus":
            net.append(nn.Softplus())  # add Softplus activationfunction
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.exp(x).clamp(min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return F.softplus(x).clamp(min=1e-4, max=1e4)

'''
NBLoss classimplementationbased onnegativeitemdistributionlossfunction，
thiskindlossfunctionespeciallysuitableused forhaveoverdisperseddiscrete（variancegreater thanmean）countnumberdata。
thiskindlossfunctioninRNAorderdataanalyzeetcapplicationincommon，becauseasthissomedatamayshowexitheightendifferent。
'''
class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=None):
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        log_nb = t1 + t2
#        result = torch.mean(torch.sum(result, dim=1))
        result = torch.sum(log_nb)
        return result

class MixtureNBLoss(nn.Module):
    def __init__(self):
        super(MixtureNBLoss, self).__init__()

    def forward(self, x, mean1, mean2, disp, pi_logits, scale_factor=None):
        eps = 1e-10
        if scale_factor is not None:
            scale_factor = scale_factor[:, None]
            mean1 = mean1 * scale_factor
            mean2 = mean2 * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2_1 = (disp+x) * torch.log(1.0 + (mean1/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean1+eps)))
        log_nb_1 = t1 + t2_1

        t2_2 = (disp+x) * torch.log(1.0 + (mean2/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean2+eps)))
        log_nb_2 = t1 + t2_2

        logsumexp = torch.logsumexp(torch.stack((- log_nb_1, - log_nb_2 - pi_logits)), dim=0)
        softplus_pi = F.softplus(-pi_logits)

        log_mixture_nb = logsumexp - softplus_pi
        result = torch.sum(-log_mixture_nb)
        return result


class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, x, mean, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        result = mean - x * torch.log(mean+eps) + torch.lgamma(x+eps)
        result = torch.sum(result)
        return result

'''
 gauss_cross_entropy used forcomputetwopositivestatedistributionbetweenhandcrossentropy。inmachinelearningandstatisticalmodelin，
 handcrossentropy常used formetrictwoprobabilitydistributionbetweendifference，individeinferenceandoptimizationalgorithminevaluationmodelnearquantity.
'''
def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable
    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy


def gmm_fit(data: np.ndarray, mode_coeff=0.6, min_thres=0.3):
    """Returns delta estimate using GMM technique"""
    # Custom definition
    gmm = GaussianMixture(n_components=3)
    gmm.fit(data[:, None])
    vals = np.sort(gmm.means_.squeeze())
    res = mode_coeff * np.abs(vals[[0, -1]]).mean()
    res = np.maximum(min_thres, res)
    return res



from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.sparse import issparse
def PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, k_list=[]):
    """
    Calculate homogeneous transformation matrix, dynamically adjusting return values based on input dimensions.

    Args:
        src_cor (numpy.ndarray): Source point coordinate array (N_src, D), where N_src is the number of points, D is the dimension.
        tgt_cor (numpy.ndarray): Target point coordinate array (N_tgt, D).
        src_exp (numpy.ndarray): Source point expression feature array (N_src, F).
        tgt_exp (numpy.ndarray): Target point expression feature array (N_tgt, F).
        k_list (list[int]): List of k values for nearest neighbor smoothing (optional).

    Returns:
        numpy.ndarray: Homogeneous transformation matrix ((D+1) x (D+1)).
    """
    num_dims = src_cor.shape[1]  # Automatically detect dimensions
    if len(k_list) != 0:
        # Process source point data
        knn_src_exp = src_exp.copy()
        kd_tree = cKDTree(src_cor)
        for k in k_list:
            distances, indices = kd_tree.query(src_cor, k=k)
            src_exp = src_exp + np.mean(knn_src_exp[indices], axis=1)

        # Process target point data
        knn_tgt_exp = tgt_exp.copy()
        kd_tree = cKDTree(tgt_cor)
        for k in k_list:
            distances, indices = kd_tree.query(tgt_cor, k=k)
            tgt_exp = tgt_exp + np.mean(knn_tgt_exp[indices], axis=1)

    # Calculate correlation and match points
    corr = np.corrcoef(src_exp, tgt_exp)[:src_exp.shape[0], src_exp.shape[0]:]
    matched_src_cor = src_cor[np.argmax(corr, axis=0), :]

    # Calculate translation and rotation matrices
    mean_source = np.mean(matched_src_cor, axis=0)
    mean_target = np.mean(tgt_cor, axis=0)
    centered_source = matched_src_cor - mean_source
    centered_target = tgt_cor - mean_target
    rotation_matrix = np.dot(centered_source.T, centered_target)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(vt.T, u.T)

    # If dimension is 3 and mirror flip is detected, correct it
    if num_dims == 3 and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = np.dot(vt.T, u.T)

    translation = mean_target - mean_source

    # Build homogeneous transformation matrix
    M = np.eye(num_dims + 1)
    M[:num_dims, :num_dims] = rotation
    M[:num_dims, -1] = translation
    M[-1, 0] = 1

    return M


def find_best_matching(src_cor, tgt_cor, src_exp, tgt_exp, k_list=[3, 10, 40]):
    """
    Find the best matching point pairs, and extract possible overlapping region point pairs based on correlation analysis and dynamic segmentation.

    Args:
        src_cor (numpy.ndarray): Spatial coordinates of source points (N_src, D).
        tgt_cor (numpy.ndarray): Spatial coordinates of target points (N_tgt, D).
        src_exp (numpy.ndarray): Expression data of source points (N_src, F).
        tgt_exp (numpy.ndarray): Expression data of target points (N_tgt, F).
        k_list (list[int]): List of k values for nearest neighbor smoothing.

    Returns:
        numpy.ndarray: Filtered subset of source point coordinates.
        numpy.ndarray: Filtered subset of target point coordinates.
        pd.DataFrame: Matching result DataFrame containing matched point indices and correlations.
    """
    # Process source point data
    knn_src_exp = src_exp.copy()
    if issparse(knn_src_exp):
        knn_src_exp = knn_src_exp.todense()
    kd_tree = cKDTree(src_cor)
    for k in k_list:
        distances, indices = kd_tree.query(src_cor, k=k)
        knn_src_exp += np.mean(knn_src_exp[indices, :], axis=1)

    # handlegoalpointdata
    knn_tgt_exp = tgt_exp.copy()
    if issparse(knn_tgt_exp):
        knn_tgt_exp = knn_tgt_exp.todense()
    kd_tree = cKDTree(tgt_cor)
    for k in k_list:
        distances, indices = kd_tree.query(tgt_cor, k=k)
        knn_tgt_exp += np.mean(knn_tgt_exp[indices, :], axis=1)

    # computecorrelationmatrix
    corr = np.corrcoef(knn_src_exp, knn_tgt_exp)[:src_exp.shape[0], src_exp.shape[0]:]

    # usecorrelationsortanddynamicsegmentationmethodfindheavypoint
    def detect_inflection_point(corr_vector):
        y = np.sort(np.max(corr_vector, axis=0))[::-1]
        data = np.array(y).reshape(-1, 1)
        algo = rpt.Dynp(model="l1").fit(data)
        result = algo.predict(n_bkps=1)
        return result[0]

    first_inflection_point_src = detect_inflection_point(corr)
    first_inflection_point_tgt = detect_inflection_point(corr.T)

    # extractionmatchpoint
    set1 = np.array([[index, value] for index, value in enumerate(np.argmax(corr, axis=0))])
    set1 = np.column_stack((set1, np.max(corr, axis=0)))
    set1 = pd.DataFrame(set1, columns=['tgt_index', 'src_index', 'corr'])
    set1.sort_values(by='corr', ascending=False, inplace=True)
    set1 = set1.iloc[:first_inflection_point_tgt, :]

    set2 = np.array([[index, value] for index, value in enumerate(np.argmax(corr, axis=1))])
    set2 = np.column_stack((set2, np.max(corr, axis=1)))
    set2 = pd.DataFrame(set2, columns=['src_index', 'tgt_index', 'corr'])
    set2.sort_values(by='corr', ascending=False, inplace=True)
    set2 = set2.iloc[:first_inflection_point_src, :]

    # mergematchpoint
    result = pd.merge(set1, set2, on=['tgt_index', 'src_index'], how='inner')
    matched_src_cor = src_cor[result['src_index'].to_numpy().astype(int), :]
    matched_tgt_cor = tgt_cor[result['tgt_index'].to_numpy().astype(int), :]

    return matched_src_cor, matched_tgt_cor, result


def calculate_transformation_matrix(src_cor, tgt_cor):
    """
    Calculate homogeneous transformation matrix based on best matching point pairs.

    Args:
        src_cor (numpy.ndarray): Matched subset coordinates of source points (N, D).
        tgt_cor (numpy.ndarray): Matched subset coordinates of target points (N, D).

    Returns:
        numpy.ndarray: Homogeneous transformation matrix ((D+1) x (D+1)).
    """
    num_dims = src_cor.shape[1]

    # Calculate mean and perform centering
    mean_src = np.mean(src_cor, axis=0)
    mean_tgt = np.mean(tgt_cor, axis=0)
    centered_src = src_cor - mean_src
    centered_tgt = tgt_cor - mean_tgt

    # Calculate rotation matrix
    cov_matrix = np.dot(centered_src.T, centered_tgt)
    u, _, vt = np.linalg.svd(cov_matrix)
    rotation = np.dot(vt.T, u.T)

    # Prevent mirror flip in 3D case
    if num_dims == 3 and np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = np.dot(vt.T, u.T)

    translation = mean_tgt - mean_src

    # Build homogeneous transformation matrix
    M = np.eye(num_dims + 1)
    M[:num_dims, :num_dims] = rotation
    M[:num_dims, -1] = translation
    M[-1, 0] = 1

    return M

# Main workflow
def match_and_transform(src_cor, tgt_cor, src_exp, tgt_exp, k_list=[], sample_match=True):
    """
    Complete matching point filtering and homogeneous transformation matrix calculation.

    Args:
        src_cor, tgt_cor, src_exp, tgt_exp: Dataset coordinates and expression features.
        k_list: Nearest neighbor smoothing parameters.

    Returns:
        numpy.ndarray: Homogeneous transformation matrix.
    """
    if sample_match:
        transformation_matrix = PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, k_list)
    else:
        matched_src, matched_tgt, _ = find_best_matching(src_cor, tgt_cor, src_exp, tgt_exp, k_list)
        transformation_matrix = calculate_transformation_matrix(matched_src, matched_tgt)
    return transformation_matrix


def initialization_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, method='PCC', sample_match=True):
    if method=='PCC':
        return match_and_transform(src_cor, tgt_cor, src_exp, tgt_exp,sample_match=sample_match)




from scipy.spatial import cKDTree
def PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, k_list = []):
    if len(k_list) != 0:
        # process source slice
        knn_src_exp = src_exp.copy()
        kd_tree = cKDTree(src_cor)
        for k in k_list:
            distances, indices = kd_tree.query(src_cor, k=k)  # (source_num_points, k)
            src_exp = src_exp + np.array(np.mean(knn_src_exp[indices, :], axis=1))

        # process target slice
        knn_tgt_exp = tgt_exp.copy()
        kd_tree = cKDTree(tgt_cor)
        for k in k_list:
            distances, indices = kd_tree.query(tgt_cor, k=k)  # (source_num_points, k)
            tgt_exp = tgt_exp + np.array(np.mean(knn_tgt_exp[indices, :], axis=1))

    corr = np.corrcoef(src_exp, tgt_exp)[:src_exp.shape[0],src_exp.shape[0]:]  # (src_points, tgt_points)
    matched_src_cor = src_cor[np.argmax(corr, axis=0), :]

    # Calculate transformation: translation and rotation
    mean_source = np.mean(matched_src_cor, axis=0)
    mean_target = np.mean(tgt_cor, axis=0)
    centered_source = matched_src_cor - mean_source
    centered_target = tgt_cor - mean_target
    rotation_matrix = np.dot(centered_source.T, centered_target)
    u, _, vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(vt.T, u.T)
    translation = mean_target - mean_source
    M = np.zeros((3, 3))
    M[:2, :2] = rotation
    M[:2, 2] = translation
    M[2, 2] = 1
    M[2, 0] =  1 # Scaling factor temporary storage

    return M


import umap
import numpy as np

def apply_umap_to_image(img, n_components=3):
    """
    Apply UMAP dimensionality reduction to 100-dimensional channel data for each pixel, reducing to 3 dimensions (RGB channels).
    img: Input hyperspectral image with shape (height, width, 100)
    n_components: Number of dimensions after UMAP reduction, default is 3 (RGB)
    """
    height, width, channels = img.shape
    umap_model = umap.UMAP(n_components=n_components)

    # Flatten 100-dimensional data for each pixel for dimensionality reduction
    img_reshaped = img.reshape(-1, channels)  # Flatten to (height * width, 100)
    img_reduced = umap_model.fit_transform(img_reshaped)  # Use UMAP to reduce to 3 dimensions

    # Restore to image shape (height, width, 3)     # Normalize to [0, 1]
    img_reduced = img_reduced.reshape(height, width, n_components)
    img_reduced_min = img_reduced.min()
    img_reduced_max = img_reduced.max()
    imced = (img_reduced - img_reduced_min) / (img_reduced_max - img_reduced_min)

    img_reduced = (img_reduced * 255).astype(np.uint8)  # Map to [0, 255] and convert to uint8 type
    return img_reduced

def convert_to_image(position, xdata,):
        """
        Convert data in (x, y, R, G, B) format to an image.

        data: Data in DataFrame format, containing columns [x, y, R, G, B]
        Returns: Converted image (NumPy array)
        """
        # Assume image dimensions are the maximum x and y coordinates
        height = int(position[:,0].max())
        width = int(position[:, 1].max())
        beta = xdata.shape[0] / (height*width)

        # Initialize an empty RGB image with shape (height, width, 3)
        position = position * beta
        height = int(position[:, 0].max()) +1
        width = int(position[:, 1].max()) +1
        img = np.zeros((height, width, xdata.shape[-1]), dtype=np.uint8)
        # Fill RGB values from data into corresponding (x, y) coordinate positions
        for index, rowdata in zip(position, xdata):
            img[int(index[0]), int(index[1])] = rowdata  # Note y, x order, conforming to image matrix coordinate system
        return img

def feature_matching_SIGT(imgA, imgB):
    """
    Perform feature matching on dimensionality-reduced images and calculate rotation, scaling, and translation transformations.
    imgA, imgB: Input dimensionality-reduced images with shape (height, width, 3)
    """
    # Convert to grayscale images, as feature matching is typically performed on grayscale images
    # Apply PCA dimensionality reduction to images A and B to 3 dimensions (RGB)
    grayA = cv2.cvtColor(imgA.astype(np.float32), cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imgB.astype(np.float32), cv2.COLOR_RGB2GRAY)

    grayA = np.uint8(grayA)
    grayB = np.uint8(grayB)
    # Use SIFT to detect feature points and descriptors
    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(grayA, None)
    kpB, desB = sift.detectAndCompute(grayB, None)

    # Use brute-force matcher for matching, and apply ratio test to filter incorrect matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desA, desB, k=2)

    # Use ratio test to filter matches
    matches = []
    for m, n in raw_matches:
        if m.distance < 0.9 * n.distance:  # 0.75 is a common threshold for SIFT ratio test
            matches.append(m)

    # Sort matches
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoint coordinates
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate affine transformation matrix
    M, mask = cv2.estimateAffine2D(ptsA, ptsB)

    # Rotation matrix and translation vector
    rotation_matrix = M[:2, :2]  # Rotation matrix is the first 2x2 part of the affine matrix
    translation_vector = M[:, 2]  # Translation vector is the last column of the affine matrix

    # Rotation angle
    a, b, c, d = M.flatten()[:4]  # Rotation and scaling elements in the affine matrix

    # Scaling factor
    scale = np.sqrt(a ** 2 + b ** 2)

    return rotation_matrix, translation_vector, scale


def SIGT_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp):
    imgA = convert_to_image(src_cor, src_exp)
    imgB = convert_to_image(tgt_cor, tgt_exp)
    imgA = apply_umap_to_image(imgA, n_components=3)
    imgB = apply_umap_to_image(imgB, n_components=3)
    rotation_matrix, translation_vector, rotation_angle, scale = feature_matching_SIGT(imgA, imgB)
    M = np.zeros((3, 3))
    M[:2, :2] = rotation_matrix
    M[:2, 2] = translation_vector
    M[2, 2] = 1
    M[2, 0] =  scale
    return M


def initialization_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp, method='PCC', **kwargs):
    if method=='PCC':
        return PCC_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp)
    elif method=='SIGT':
        return SIGT_trans_matrix(src_cor, tgt_cor, src_exp, tgt_exp)

    def voxel_data(
            coords: np.ndarray,
            gene_exp: np.ndarray,
            voxel_size: float = None,
            voxel_num: int = 10000,
    ):
        """
        Voxelization of the data.
        Parameters
        ----------
        coords: np.ndarray
            The coordinates of the data points. Shape (N, D)
        gene_exp: np.ndarray
            The gene expression of the data points. Shape (N, G)
        voxel_size: float
            The size of the voxel.
        voxel_num: int
            The number of voxels.
        Returns
        -------
        voxel_coords: np.ndarray
            The coordinates of the voxels.
        voxel_gene_exp: np.ndarray
            The gene expression of the voxels.
        """
        N, D = coords.shape

        # ensureinputis numpy array
        coords = np.asarray(coords)
        gene_exp = np.asarray(gene_exp)

        # createbodyelementgrid
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)

        # computebodyelementsize
        if voxel_size is None:
            voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)

        # computeeachdimensionbodyelementquantity
        grid_size = int(np.sqrt(voxel_num))
        voxel_steps = (max_coords - min_coords) / grid_size

        # generatebodyelementgridcoordinates
        voxel_coords_list = [
            np.arange(min_coord, max_coord, voxel_step)
            for min_coord, max_coord, voxel_step in zip(min_coords, max_coords, voxel_steps)
        ]

        # creategridcoordinates
        voxel_coords = np.stack(np.meshgrid(*voxel_coords_list), axis=-1).reshape(-1, D)
        voxel_gene_exps = np.zeros((voxel_coords.shape[0], gene_exp.shape[1]))
        is_voxels = np.zeros(voxel_coords.shape[0], dtype=bool)

        # willdatapointallocationtobodyelement
        for i, voxel_coord in enumerate(voxel_coords):
            # computetowhenbeforebodyelementdistance
            dists = np.sqrt(np.sum((coords - voxel_coord) ** 2, axis=1))
            # findtodistanceless thanbodyelementhalfpathpoint
            mask = dists < voxel_size / 2
            if np.any(mask):
                # computethissomepointaveragegeneexpression
                voxel_gene_exps[i] = np.mean(gene_exp[mask], axis=0)
                is_voxels[i] = True

        # onlyretainhavedatabodyelement
        voxel_coords = voxel_coords[is_voxels]
        voxel_gene_exps = voxel_gene_exps[is_voxels]

        return voxel_coords, voxel_gene_exps

def normalize_coordinates(coordsA, coordsB, separate_mean=True, separate_scale=True, verbose=False):
    """
    Spatial coordinate normalization function implemented using PyTorch

    Parameters:
    coordsA (torch.Tensor): Coordinate matrix of the first sample, shape [N, D]
    coordsB (torch.Tensor): Coordinate matrix of the second sample, shape [M, D]
    separate_mean (bool): Whether to calculate means separately, default is True
    separate_scale (bool): Whether to calculate scaling factors separately, default is True
    verbose (bool): Whether to print normalization parameter information

    Returns:
    tuple: Normalized coordinates (coordsA_norm, coordsB_norm), scaling factors, means

    Raises:
    AssertionError: Raised when the dimensions of the two coordinate matrices are inconsistent
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(coordsA, torch.Tensor):
        coordsA = torch.tensor(coordsA)
    if not isinstance(coordsB, torch.Tensor):
        coordsB = torch.tensor(coordsB)

    # Check if dimensions are consistent
    assert coordsA.shape[1] == coordsB.shape[1], "The dimensions of the two coordinate matrices must be consistent"

    D = coordsA.shape[1]  # Coordinate dimension
    coords = [coordsA, coordsB]
    normalize_scales = torch.zeros(2, dtype=coordsA.dtype, device=coordsA.device)
    normalize_means = torch.zeros(2, D, dtype=coordsA.dtype, device=coordsA.device)

    # Calculate mean of each coordinate matrix
    for i in range(len(coords)):
        normalize_mean = torch.mean(coords[i], dim=0)
        normalize_means[i] = normalize_mean

    # If not calculating means separately, use global mean
    if not separate_mean:
        global_mean = torch.mean(normalize_means, dim=0)
        normalize_means = global_mean.expand(2, -1)

    # Center coordinates and calculate scaling factors
    for i in range(len(coords)):
        coords[i] -= normalize_means[i]
        squared_sum = torch.sum(coords[i] * coords[i]) / coords[i].shape[0]
        normalize_scale = torch.sqrt(squared_sum)
        normalize_scales[i] = normalize_scale

    # If not calculating scaling factors separately, use global scaling factor
    if not separate_scale:
        global_scale = torch.mean(normalize_scales)
        normalize_scales = torch.full_like(normalize_scales, global_scale)

    # Check if normalization factors are valid, ensuring consistent data types
    for i in range(len(normalize_scales)):
        if torch.isclose(normalize_scales[i],
                         torch.tensor(0.0, dtype=normalize_scales.dtype, device=normalize_scales.device)):
            raise ValueError(f"The {i + 1}th normalization factor is close to zero, unable to normalize")

    # Apply scaling factors
    for i in range(len(coords)):
        coords[i] /= normalize_scales[i]

    # Print normalization information (if needed)
    if verbose:
        print(f"Spatial coordinate normalization parameters:")
        print(f"Scaling factors: {normalize_scales}")
        print(f"Means: {normalize_means}")

    return coords[0], coords[1], normalize_scales, normalize_means

def normalize_expression_matrices(XA, XB, verbose=False):
    """
    Gene expression matrix normalization function implemented using PyTorch

    Parameters:
    XA (torch.Tensor): Gene expression matrix of the first sample
    XB (torch.Tensor): Gene expression matrix of the second sample
    verbose (bool): Whether to print normalization parameter information

    Returns:
    tuple: Normalized matrices (XA_norm, XB_norm)

    Raises:
    ValueError: Raised when normalization factor cannot be calculated
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(XA, torch.Tensor):
        XA = torch.tensor(XA)
    if not isinstance(XB, torch.Tensor):
        XB = torch.tensor(XB)

    # Calculate normalization factor
    normalize_scale = 0

    # Calculate Frobenius norm of each matrix and accumulate
    for X in [XA, XB]:
        # Calculate mean of squared sum of matrix
        squared_sum = torch.sum(X * X) / X.shape[0]
        # Take square root and accumulate to normalization factor
        normalize_scale += torch.sqrt(squared_sum)

    # Calculate average normalization factor
    normalize_scale /= 2.0

    # Check if normalization factor is valid, ensuring consistent data types
    if torch.isclose(normalize_scale,
                     torch.tensor(0.0, dtype=normalize_scale.dtype, device=normalize_scale.device)):
        raise ValueError("Normalization factor is close to zero, unable to normalize")

    # Apply normalization
    XA_norm = XA / normalize_scale
    X= XB / normalize_scale

    # Print normalization information (if needed)
    if verbose:
        print(f"Gene expression normalization parameters:")
        print(f"Scaling factor: {normalize_scale.item():.6f}")

    return XA_norm, XB_norm

def torch_corrcoef(src_exp, tgt_exp):
    """
    Calculate Pearson correlation coefficient matrix between two matrices
    """
    # Ensure each matrix column is mean-centered
    src_exp_centered = src_exp - src_exp.mean(dim=0, keepdim=True)  # Mean-center each column
    tgt_exp_centered = tgt_exp - tgt_exp.mean(dim=0, keepdim=True)  # Mean-center each column

    # Calculate covariance matrix
    cov_matrix = torch.mm(src_exp_centered, tgt_exp_centered.T) / src_exp_centered.shape[-1]

    # Calculate norm (standard deviation) of each column
    src_norm = torch.norm(src_exp_centered, dim=1)  # [n_features]
    tgt_norm = torch.norm(tgt_exp_centered, dim=1)  # [m_features]
    # Normalize covariance matrix
    corr_matrix = cov_matrix / (src_norm.view(-1, 1) * tgt_norm.view(1, -1))  # Broadcast to get [n_features, m_features]
    positive_corr = torch.clamp(corr_matrix, min=1e-8)

    return positive_corr

def harmony_integration(exp_A, exp_B):
    """Use Harmony for batch correction and return dimensionality-reduced features

    Parameters:
        exp_A : Sample count × gene count matrix (numpy array or sparse matrix)
        exp_B : Sample count × gene count matrix (numpy array or sparse matrix)

    Returns:
        corrected_A : Corrected feature matrix for sample A (numpy array)
        corrected_B : Corrected feature matrix for sample B (numpy array)
    """
    # Create AnnData objects and add batch information
    adata_A = ad.AnnData(exp_A)
    adata_A.obs['batch'] = 'batch_0'

    adata_B = ad.AnnData(exp_B)
    adata_B.obs['batch'] = 'batch_1'

    # Merge datasets
    adata = adata_A.concatenate(adata_B, join='outer', batch_key='batch')

    # Preprocessing workflow
    sc.pp.normalize_total(adata, target_sum=1e4)  # CPM normalization
    sc.pp.log1p(adata)  # log1p transformation
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # Select highly variable genes
    adata = adata[:, adata.var.highly_variable]  # Keep highly variable genes

    # PCA dimensionality reduction
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)

    # Harmony batch correction
    harmony_emb = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'batch')
    adata.obsm['X_pca_harmony'] = harmony_emb.Z_corr.T

    # Split data in original sample order
    orig_order = np.concatenate([
        np.arange(exp_A.shape[0]),
        exp_A.shape[0] + np.arange(exp_B.shape[0])
    ])
    adata = adata[orig_order].copy()  # Maintain original sample order

    # Split corrected features
    corrected_A = adata[:exp_A.shape[0]].obsm['X_pca_harmony']
    corrected_B = adata[exp_].obsm['X_pca_harmony']

    return corrected_A, corrected_B

# Calculate correlation (P) between gene expression matrices
def kl_distance_backend(
        X,
        Y,
        probabilistic: bool = True,
        eps: float = 1e-8,
        device=None,
):
    """
    Compute the pairwise KL divergence between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    probabilistic : bool, optional
        If True, normalize the rows of X and Y to sum to 1 (to interpret them as probabilities).
        Default is True.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray
        Pairwise KL divergence matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X).to(device)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y).to(device)
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    X = (X - X.min()) / (X.max() - X.min())
    Y = (Y - Y.min()) / (Y.max() - Y.min())
    X = X + 0.01
    Y = Y + 0.01
    # Normalize rows to sum to 1 if probabilistic is True
    if probabilistic:
        X = X / torch.sum(X, dim=1, keepdims=True)
        Y = Y / torch.sum(Y, dim=1, keepdims=True)

    # Compute log of X and Y
    log_X = torch.log(X + eps)  # Adding epsilon to avoid log(0)
    log_Y = torch.log(Y + eps)  # Adding epsilon to avoid log(0)

    # Compute X log X and the pairwise KL divergence
    X_log_X = torch.sum(X * log_X, dim=1, keepdims=True)

    D = X_log_X - torch.matmul(X, log_Y.T)

    return D


def get_rotation_angle(R, reture_axis=False):
    """Get rotation axis and angle, supporting 2D and 3D"""
    D = R.shape[0]
    device = R.device

    if D == 2:
        # 2Dsituation：rotateaxisfixedasZaxis(0,0,1)，onlyneedcomputeangle
        cos_theta = (R[0, 0] + R[1, 1]) / 2
        sin_theta = (R[1, 0] - R[0, 1]) / 2
        theta = torch.atan2(sin_theta, cos_theta)
        angle_deg = torch.rad2deg(theta)
        axis = torch.tensor([0, 0, 1], device=device)  # 2DrotateaxisasZaxis

    elif D == 3:
        # 3Dsituation：computeaxisandangle
        trace = torch.trace(R)
        cos_theta = (trace - 1) / 2
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

        # handlespecialsituation：rotateangleconnectnear0（thistimeaxisvectornotstablestable）
        if torch.abs(theta) < 1e-6:
            # zerorotate，returnanymeaningsinglepositionaxis（examplesuch asXaxis）
            return torch.tensor([1, 0, 0], device=device), torch.tensor(0.0, device=device)

        # handlespecialsituation：rotateangleconnectnear180degree（thistimeneedspecialhandle）
        elif torch.abs(theta - torch.pi) < 1e-6:
            # whentheta=πtime，directlyfrommatrixelementcomputeaxisvector
            xx = (R[0, 0] + 1) / 2
            yy = (R[1, 1] + 1) / 2
            zz = (R[2, 2] + 1) / 2
            xy = (R[0, 1] + R[1, 0]) / 4
            xz = (R[0, 2] + R[2, 0]) / 4
            yz = (R[1, 2] + R[2, 1]) / 4

            # choosemaximumforangleelementtoobtainnumericstablestable
            max_val = max(xx, yy, zz)

            if max_val == xx:
                x = torch.sqrt(xx)
                y = xy / x
                z = xz / x
            elif max_val == yy:
                y = torch.sqrt(yy)
                x = xy / y
                z = yz / y
            else:  # max_val == zz
                z = torch.sqrt(zz)
                x = xz / z
                y = yz / z

            axis = torch.stack([x, y, z])

        else:
            # General case: compute axis through skew-symmetric matrix
            K = (R - R.T) / 2
            axis = torch.stack([
                K[2, 1],  # R[2,1] - R[1,2]
                K[0, 2],  # R[0,2] - R[2,0]
                K[1, 0]  # R[1,0] - R[0,1]
            ])
            axis = axis / (2 * torch.sin(theta))

        # normalizationaxisvector
        axis = axis / torch.norm(axis)
        angle_deg = torch.rad2deg(theta)

    else:
        # no2D/3Dsituationreturnzerorotate
        axis, angle_deg=torch.tensor([1, 0, 0], device=device), torch.tensor(0.0, device=device)
    if reture_axis:
        return f'{angle_deg} axis={axis}'
    else:
        return angle_deg

def voxel_data_torch(
        coords: torch.Tensor,
        gene_exp: torch.Tensor,
        voxel_size: float = None,
        voxel_num: int = 10000,
):
    """
    PyTorch version voxelization function (supports GPU acceleration)

    Parameters:
        coords: Spatial coordinate tensor (N, D)
        gene_exp: Gene expression matrix (N, G)
        voxel_size: Voxel edge length (set to None for automatic calculation)
        voxel_num: Target voxel count (effective when voxel_size is None)
        device: Computing device ('cuda' or 'cpu')

    Returns:
        voxel_coords: Valid voxel coordinates (M, D)
        voxel_gene_exp: Voxel gene expression (M, G)
    """
    # Device configuration
    device = coords.device
    dtype = coords.dtype
    N, D = coords.shape

    # Calculate coordinate boundaries
    min_coords = torch.min(coords, dim=0).values
    max_coords = torch.max(coords, dim=0).values
    spatial_range = max_coords - min_coords

    # Automatically calculate voxel size
    if voxel_size is None:
        spatial_range = max_coords - min_coords
        volume = torch.prod(spatial_range)
        points_per_voxel = 5 ** D
        voxel_size = (volume * points_per_voxel / N) ** (1 / D)

    # Generate voxel grid - dynamically calculate grid size based on dimension
    # Key modification: use dimension-related grid size calculation
    if D == 2:
        # 2D: use square root
        grid_size = int(torch.sqrt(torch.tensor(voxel_num)))
    else:
        # 3D+: use cube root
        grid_size = int(round(torch.pow(torch.tensor(voxel_num), 1 / D).item()))

    voxel_steps = spatial_range / grid_size

    # Create multi-dimensional grid
    grid = [torch.arange(min_coords[d], max_coords[d], voxel_steps[d],
                         device=device, dtype=dtype) for d in range(D)]
    mesh = torch.meshgrid(grid, indexing='ij')
    voxel_coords = torch.stack(mesh, dim=-1).reshape(-1, D)  # (M, D)

    # Batch distance calculation
    dists = torch.cdist(coords, voxel_coords, p=2)  # (N, M)
    valid_mask = dists < (voxel_size / 2)  # (N, M)

    # Vectorized aggregation calculation
    mask_sum = valid_mask.sum(dim=0)  # (M,)
    valid_voxels = mask_sum > 0  # (M,)
    valid_mask = valid_mask.float().T.to(dtype=gene_exp.dtype)
    aggregated = torch.mm(valid_mask, gene_exp)  # (M, G)

    # Calculate average expression
    voxel_gene_exp = aggregated / mask_sum[:, None]  # (M, G)
    voxel_gene_exp[~valid_voxels] = 0  # Handle empty voxels

    # Filter valid voxels
    return voxel_coords[valid_voxels], voxel_gene_exp[valid_voxels]


# def inner_NN(exp_A, exp_B, X_A, X_B, max_iter=100, tol=1e-7, exp_P=None):
#     D = X_A.shape[1]
#     device = X_A.device
#     # canadjustmentparameter
#     # computeexpressionsimilarity
#     if exp_P is None:
#         exp_P = cal_exp_P(exp_A, exp_B)
#
#     # initializerotatematrixandtranslatevector
#     R = torch.eye(D, dtype=torch.float64, device=device)
#     t = torch.zeros(D, dtype=torch.float64, device=device)
#
#     prev_R, prev_t = R.clone(), t.clone()
#     decay_factor = 1
#     for iter_idx in range(max_iter):
#         # ----------------------------
#         # 1. computepointfordistancematrix
#         # ----------------------------
#         # computeX_AandX_BbetweenEuclideandistancematrix
#         dist_matrix = torch.cdist(X_A @ R.T + t, X_B, p=2.0)
#         median_dist = torch.median(dist_matrix)
#         threshold = 3 * median_dist
#
#         # willthresholdvaluebreakasthreshold
#         dist_matrix = torch.where(dist_matrix > threshold, threshold, dist_matrix)
#         # convertassimilaritymatrix (distancefar，similaritylower)
#         # useheighten斯核而nolog
#         # ----------------------------
#         # 3. computemeanwithgoinheartcoordinates
#         # ----------------------------
#         sigma = median_dist / 2.0
#         spatial_sim = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
#         spatial_P1 = spatial_sim / (spatial_sim.sum(dim=0, keepdim=True) + 1e-15)
#         spatial_P2 = spatial_sim / (spatial_sim.sum(dim=1, keepdim=True) + 1e-15)
#         spatial_P = (spatial_P1 + spatial_P2) / 2.0
#         eps = 1e-15
#         spatial_P = torch.log(spatial_P + eps) - torch.log(torch.min(spatial_P) + eps)
#         spatial_P = spatial_P / torch.max(spatial_P)
#         exp_P_clamped = exp_P * spatial_P
#         P = exp_P_clamped / (exp_P_clamped.sum(dim=1, keepdim=True) + 1e-15) + 1e-15
#         total_weight = P.sum()
#
#         # goalpointcloudaddmean
#         weights_B = P.sum(dim=0)  # [N_B]
#         mu_XB = (X_B.T @ weights_B) / total_weight  # [D]
#
#         # 源pointcloudaddmean
#         weights_A = P.sum(dim=1)  # [N_A]
#         mu_XA = (X_A.T @ weights_A) / total_weight  # [D]
#
#         # goinheartcoordinates
#         XB_centered = X_B - mu_XB  # [N_B, D]
#         XA_centered = X_A - mu_XA  # [N_A, D]
#
#         A = XB_centered.T @ (P.T @ XA_centered)  # [D, D]
#
#         U, S, Vh = torch.linalg.svd(A)
#         V = Vh.T
#
#         # keeprightcoordinatesrelate
#         C = torch.eye(D, dtype=U.dtype, device=device)
#         if torch.det(U @ V.T) < 0:
#             C[-1, -1] = -1.0
#
#         R = U @ C @ V.T
#
#         # ----------------------------
#         # 6. computetranslatevector
#         # ----------------------------
#         t = mu_XB - R @ mu_XA
#         print(
#             f"\riteration {iter_idx + 1}/{max_iter} - rotateangle: {get_rotation_angle(R):.4f}°, translatevector: [{', '.join([f'{v:.6f}' for v in t])}]",
#             end='')
#         # checkcollect敛
#         if (torch.norm(R - prev_R) < tol) and (torch.norm(t - prev_t) < tol):
#             print(f"iteration {iter_idx + 1}/{max_iter} alreadycollect敛")
#             break
#
#         prev_R, prev_t = R.clone(), t.clone()
#
#     dist_matrix = torch.cdist(X_A @ R.T + t, X_B, p=2.0)
#     spatial_sim = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
#     spatial_P = spatial_sim / (spatial_sim.sum(dim=1, keepdim=True) + 1e-15)
#     P = exp_P * clamp_P(spatial_P)
#
#     # constructneattimetransformmatrix
#     M = torch.eye(D + 1, dtype=R.dtype, device=device)
#     M[:D, :D] = R
#     M[:D, D] = t
#     return M.cpu().detach(), P.cpu().detach()
#
def cal_single_P(A, B, k_beta=None, sigma=None, eps=1e-15, method='kl', use_threshold=False):
    if method == 'kl':
        exp_P = kl_distance_backend(A, B)
    if method == 'dist':
        exp_P = torch.cdist(A, B, p=2.0)

    median_dist = 1.5 * torch.median(exp_P)

    if sigma is None:
        sigma = median_dist / 2.0
    if use_threshold:
        threshold = 3 * median_dist
        exp_P = torch.where(exp_P > threshold, threshold, exp_P)
    exp_P = torch.exp(-exp_P ** 2 / (2 * sigma ** 2))
    exp_P1 = exp_P / (exp_P.sum(dim=1, keepdim=True) + 1e-15)
    exp_P2 = exp_P / (exp_P.sum(dim=0, keepdim=True) + 1e-15)
    if not k_beta is None:
        exp_P1 = clamp_P(exp_P1, k_beta, True)
        exp_P2 = clamp_P(exp_P2.T, k_beta, True).T
    exp_P = (exp_P1 + exp_P2) / 2.0
    exp_P = torch.log(exp_P + eps) - torch.log(torch.min(exp_P) + eps)
    exp_P = exp_P / torch.max(exp_P)
    return exp_P



def cal_all_P(spatial_P, exp_P, alpha=0.5):
    return (1 - alpha) * spatial_P + alpha * exp_P


def inner_NN(exp_A, exp_B, X_A, X_B, max_iter=100, tol=1e-7, alpha=0.5, exp_P=None):
    D = X_A.shape[1]
    device = X_A.device
    dtype = X_A.dtype
    # canadjustmentparameter
    # computeexpressionsimilarity
    if exp_P is None:
        exp_P = cal_single_P(exp_A, exp_B, k_beta=0.1)

    # initializerotatematrixandtranslatevector
    R = torch.eye(D, dtype=dtype, device=device)
    t = torch.zeros(D, dtype=dtype, device=device)

    prev_R, prev_t = R.clone(), t.clone()
    decay_factor = 1
    for iter_idx in range(max_iter):
        # ----------------------------
        # 1. computepointfordistancematrix
        # ----------------------------
        # computeX_AandX_BbetweenEuclideandistancematrix
        spatial_P = cal_single_P(X_A @ R.T + t, X_B, method='dist', use_threshold=True)

        P = cal_all_P(spatial_P, exp_P, alpha=alpha)  # exp_P * spatial_P #  ((1-alpha) * spatial_P.T + alpha * exp_P)
        # P = exp_P_clamped / (exp_P_clamped.sum(dim=1, keepdim=True) + 1e-15) + 1e-15
        total_weight = P.sum()

        # goalpointcloudaddmean
        weights_B = P.sum(dim=0)  # [N_B]
        mu_XB = (X_B.T @ weights_B) / total_weight  # [D]

        # 源pointcloudaddmean
        weights_A = P.sum(dim=1)  # [N_A]
        mu_XA = (X_A.T @ weights_A) / total_weight  # [D]

        # goinheartcoordinates
        XB_centered = X_B - mu_XB  # [N_B, D]
        XA_centered = X_A - mu_XA  # [N_A, D]

        A = XB_centered.T @ (P.T @ XA_centered)  # [D, D]

        U, S, Vh = torch.linalg.svd(A)
        V = Vh.T

        # keeprightcoordinatesrelate
        C = torch.eye(D, dtype=U.dtype, device=device)
        if torch.det(U @ V.T) < 0:
            C[-1, -1] = -1.0

        R = U @ C @ V.T

        # ----------------------------
        # 6. computetranslatevector
        # ----------------------------
        t = mu_XB - R @ mu_XA
        alpha = alpha * 0.9
        print(
            f"\riteration {iter_idx + 1}/{max_iter} - rotateangle: {get_rotation_angle(R):.4f}°, translatevector: [{', '.join([f'{v:.6f}' for v in t])}]",
            end='')
        # checkcollect敛
        if (torch.norm(R - prev_R) < tol) and (torch.norm(t - prev_t) < tol):
            print(f"iteration {iter_idx + 1}/{max_iter} alreadycollect敛")
            break

        prev_R, prev_t = R.clone(), t.clone()

    spatial_P = cal_single_P(X_A @ R.T + t, X_B, method='dist', use_threshold=True)
    P = exp_P * clamp_P(spatial_P)  # connectcombineprobabilitydistribution

    # constructneattimetransformmatrix
    M = torch.eye(D + 1, dtype=R.dtype, device=device)
    M[:D, :D] = R
    M[:D, D] = t
    return M.cpu().detach(), P.cpu().detach()
#
#
# def cal_single_P(A, B, k_beta=None, sigma=None, eps=1e-15, method='kl', use_threshold=False):
#     """computepointforpointsimilaritymatrix，supportanymeaningdimension"""
#     if method == 'kl':
#         exp_P = kl_distance_backend(A, B)  # ensurethisfunctionsupportheighten维
#     elif method == 'dist':
#         # supportanymeaningdimensiondistancecompute
#         exp_P = torch.cdist(A, B, p=2.0)
#     else:
#         raise ValueError(f"未knowmethod: {method}")
#
#     # automaticdeterministicsigmaparameter
#     median_dist = torch.median(exp_P[exp_P > 0]) if exp_P.numel() > 0 else 1.0
#     median_dist = 1.5 * median_dist
#
#     if sigma is None:
#         sigma = median_dist / 2.0
#
#     # optionaldistancebreak
#     if use_threshold:
#         threshold = 3 * median_dist
#         exp_P = torch.where(exp_P > threshold, threshold, exp_P)
#
#     # computeheightensimilarity
#     exp_P = torch.exp(-exp_P ** 2 / (2 * sigma ** 2))
#
#     # linenormalizationandcolumnnormalization
#     exp_P1 = exp_P / (exp_P.sum(dim=1, keepdim=True) + eps
#                       exp_P2 = exp_P/ (exp_P.sum(dim=0, keepdim=True) + eps
#
#     # optionaldoubletowardbreak
#     if k_beta is not None:
#         exp_P1 = clamp_P(exp_P1, k_beta, clamp_rows=True)
#     exp_P2 = clamp_P(exp_P2.T, k_beta, clamp_rows=True).T
#
#     # fornameprobabilitymatrix
#     exp_P = (exp_P1 + exp_P2) / 2.0
#
#     # numericstablehandle
#     min_val = torch.min(exp_P) + eps
#     exp_P = torch.log(exp_P + eps) - torch.log(min_val)
#     exp_P = exp_P / torch.max(exp_P)
#
#     return exp_P

#
# def cal_all_P(spatial_P, exp_P, alpha=0.5):
#     """compositespaceandexpressionsimilarity"""
#     return (1 - alpha) * spatial_P + alpha * exp_P
#
#
# def inner_NN(exp_A, exp_B, X_A, X_B, max_iter=100, tol=1e-7, alpha=0.5, exp_P=None):
#     """
#     pointcloudregistrationmethod，support2D、3Dandmoreheightendimension
#     parameter:
#         exp_A, exp_B: expressionfeaturematrix [N_A, F], [N_B, F]
#         X_A, X_B: spacecoordinatesmatrix [N_A, D], [N_B, D] (D=2,3,4...)
#     """
#     # checkdimensionone致
#     assert X_A.shape[1] == X_B.shape[1], "spacedimensionmustone致"
#     D = X_A.shape[1]  # spacedimension (2,3,4...)
#     device = X_A.device
#     dtype = X_A.dtype
#
#     # computeexpressionsimilaritymatrix
#     if exp_P is None:
#         exp_P = cal_single_P(exp_A, exp_B, k_beta=0.1)
#
#     # initializetransformparameter
#     R = torch.eye(D, dtype=dtype, device=device)  # rotatematrix
#     t = torch.zeros(D, dtype=dtype, device=device)  # translatevector
#
#     # iterationoptimization
#     prev_loss = float('inf')
#     for iter_idx in range(max_iter):
#         # 1. applicationwhenbeforetransform
#         transformed_A = X_A @ R.T + t
#
#         # 2. computespacesimilarity
#         spatial_P = cal_single_P(transformed_A, X_B, method='dist', use_threshold=True)
#
#         # 3. compositespaceandexpressionsimilarity
#         P = cal_all_P(spatial_P, exp_P, alpha=alpha)
#
#         # 4. computeaddweight质heart
#         total_weight = P.sum()
#         weights_B = P.sum(dim=0)  # [N_B]
#         weights_A = P.sum(dim=1)  # [N_A]
#
#         mu_B = (X_B.T @ weights_B) / total_weight  # [D]
#         mu_A = (X_A.T @ weights_A) / total_weight  # [D]
#
#         # 5. goinheart
#         XB_centered = X_B - mu_B
#         XA_centered = X_A - mu_A
#
#         # 6. computecovariancematrix
#         A = XB_centered.T @ (P.T @ XA_centered)  # [D, D]
#
#         # 7. SVDdividesolveseeksolvemostexcellentrotate
#         U, S, Vh = torch.linalg.svd(A)
#         V = Vh.T
#
#         # 8. handleshootsituation (ensureispurerotate)
#         det_UV = torch.det(U @ V.T)
#         correction = torch.eye(D, device=device, dtype=dtype)
#         if det_UV < 0:
#             correction[-1, -1] = -1.0
#
#         R = U @ correction @ V.T
#
#         # 9. computetranslate
#         t = mu_B - R @ mu_A
#
#         # 10. computelossandcheckcollect敛
#         current_loss = torch.norm(transformed_A - X_B, p='fro').item()
#         loss_delta = abs(prev_loss - current_loss)
#
#         # hitprintiterationinformation (dimensionnoclose)
#         print(f"Iter {iter_idx + 1}/{max_iter}: Loss={current_loss:.6f}, Δ={loss_delta:.6f}, "
#               f"Trans={' '.join([f'{x:.4f}' for x in t])}")
#
#         # checkcollectcondition
#         if loss_delta < tol:
#             print(f"Converged at iteration {iter_idx + 1}")
#             break
#
#         prev_loss = current_loss
#         alpha *= 0.95  # 衰subtractexpressionsimilarityweight
#
#     # finaltransformandprobabilitymatrix
#     final_transformed = X_A @ R.T + t
#     spatial_P = cal_single_P(final_transformed, X_B, method='dist', use_threshold=True)
#     P = exp_P * clamp_P(spatial_P)  # connectcombineprobabilitydistribution
#
#     # constructneattimetransformmatrix [D+1, D+1]
#     M = torch.eye(D + 1, dtype=dtype, device=device)
#     M[:D, :D] = R
#     M[:D, D] = t
#
#     return M.cpu().detach(), P.cpu().detach()

def clamp_P(P_matrix, k_beta=0.1, min_k=5, reture_values=True):
    """
    pressline筛选matrix，retaineachlineinmaximumk_betaratioexampleelement（tofewmin_k），itsremaining设as0

    parameter:
    P_matrix: inputmatrix
    k_beta: eachlineretainelementratioexample (default0.1，即10%)
    min_k: eachlinetofewretainelementquantity (default5)

    return:
    筛选aftermatrix
    """
    # ensurematrixelementnonegative
    P_clamped = torch.clamp(P_matrix, 0, float('inf'))

    # getmatrixshape
    rows, cols = P_clamped.shape

    # computeeachlineneedretainelementquantity（tofewmin_k）
    k_per_row = max(int(cols * k_beta), min_k)

    # ensurek_per_rownotcolumnnumber
    k_per_row = min(k_per_row, cols)

    # createcodematrix，used formarkereachlineinneedretainelement
    mask = torch.zeros_like(P_clamped, dtype=torch.bool)
    result = torch.zeros_like(P_clamped)

    # foreachlinefindtomaximumk_per_rowelementindex
    for i in range(rows):
        # getwhenbeforelineinmaximumk_per_rowelementindex
        _, indices = torch.topk(P_clamped[i], k_per_row)
        # incodeinmarkerthissomepositionplaceasTrue
        if reture_values:
            mask[i, indices] = True
        else:
            result[i, indices] = 1
    # applicationcode，willnomarkerpositionplaceas0
    if reture_values:
        result = torch.where(mask, P_clamped, torch.zeros_like(P_clamped))
        return result
    else:
        return result

def calculate_data_scale(X_A, X_B):
    """computedatascalefeature，used forselfsuitableshouldparameterset"""
    # mergeallpoint
    all_points = torch.cat([X_A, X_B])

    # computepointcloudspacerange
    min_vals = all_points.min(dim=0).values
    max_vals = all_points.max(dim=0).values
    spatial_range = max_vals - min_vals

    # computepointclouddegreefeature
    dists = torch.cdist(X_A, X_A)
    dists.fill_diagonal_(float('inf'))  # 忽略selfdistance
    min_dists = dists.min(dim=1).values
    median_dist = torch.median(min_dists).item()

    # computespacescalefeature
    spatial_scale = torch.norm(spatial_range).item()

    return {
        'spatial_scale': spatial_scale,  # pointcloudwholescale
        'median_dist': median_dist,  # pointcloudinpositiondistance
        'dims': X_A.shape[1]  # spacedimension
    }


# auxiliaryfunction: applicationneattimetransformmatrix
def transform_points(points, M):
    n = points.shape[0]
    points_homo = torch.cat([points, torch.ones(n, 1, device=points.device)], dim=1)
    transformed = torch.mm(points_homo, M.t())
    return transformed[:, :points.shape[1]]


# auxiliaryfunction: createperturbationtransformmatrix
# auxiliaryfunction - dimensionnocloseperturbationmatrixcreate
# def create_perturbation_matrix(rot_params, trans_vector, D, device, dtype):
#     """
#     based ondimensioncreateperturbationtransformmatrix
#     parameter:
#         rot_params: rotateparameterlist
#         trans_vector: translatevector (D,)
#         D: spacedimension
#     """
#     # createfoundationsinglepositionmatrix
#     M = torch.eye(D + 1, dtype=dtype, device=device)
#
#     # applicationtranslate
#     M[:D, D] = trans_vector
#
#     # based ondimensionapplicationrotate
#     if D == 2:
#         # 2D: 绕Zaxisrotate
#         angle = torch.deg2rad(torch.tensor(rot_params[0], device=device))
#         cos_a = torch.cos(angle)
#         sin_a = torch.sin(angle)
#         R = torch.tensor([
#             [cos_a, -sin_a],
#             [sin_a, cos_a]
#         ], device=device, dtype=dtype)
#         M[:2, :2] = R
#
#     elif D == 3:
#         # 3D: 欧pullangle (ZYXsequential)
#         angles = torch.deg2rad(torch.tensor(rot_params, device=device))
#
#         # 绕Zaxisrotate
#         cos_z = torch.cos(angles[2])
#         sin_z = torch.sin(angles[2])
#         Rz = torch.tensor([
#             [cos_z, -sin_z, 0],
#             [sin_z, cos_z, 0],
#             [0, 0, 1]
#         ], device=device, dtype=dtype)
#
#         # 绕Yaxisrotate
#         cos_y = torch.cos(angles[1])
#         sin_y = torch.sin(angles[1])
#         Ry = torch.tensor([
#             [cos_y, 0, sin_y],
#             [0, 1, 0],
#             [-sin_y, 0, cos_y]
#         ], device=device, dtype=dtype)
#
#         # 绕Xaxisrotate
#         cos_x = torch.cos(angles[0])
#         sin_x = torch.sin(angles[0])
#         Rx = torch.tensor([
#             [1, 0, 0],
#             [0, cos_x, -sin_x],
#             [0, sin_x, cos_x]
#         ], device=device, dtype=dtype)
#
#         # compositerotate R = Rz * Ry * Rx
#         R = Rz @ Ry @ Rx
#         M[:3, :3] = R
#
#     elif D >= 4:
#         # 4D+: useaxisangleexpressmethod
#         angle = torch.deg2rad(torch.tensor(rot_params[0], device=device))
#         # randomchooserotatesurface (actualapplicationinmayneedmorecomplexhandle)
#         axis1 = torch.zeros(D, device=device)
#         axis2 = torch.zeros(D, device=device)
#         axis1[0] = 1
#         axis2[1] = 1
#
#         # createrotatematrix (simplifyversion)
#         cos_a = torch.cos(angle)
#         sin_a = torch.sin(angle)
#
#         R = torch.eye(D, device=device, dtype=dtype)
#         R[0, 0] = cos_a
#         R[0, 1] = -sin_a
#         R[1, 0] = sin_a
#         R[1, 1] = cos_a
#
#         M[:D, :D] = R
#     return M


def create_perturbation_matrix(rot_params, trans_vector, D, device, dtype, deg=False):
    """
    based ondimensioncreateperturbationtransformmatrix，maintaingradientpropagation
    parameter:
        rot_params: rotateparametertensor（canmicrodivide）
        trans_vector: translatevectortensor (D,)（canmicrodivide）
        D: spacedimension
    """
    # createfoundationsinglepositionmatrix（maintaingradient）
    M = torch.eye(D + 1, dtype=dtype, device=device)

    # applicationtranslate（maintaingradient）
    M = M.clone()  # createcanmodify副this
    M[:D, D] = trans_vector[:D]

    # based ondimensionapplicationrotate
    if D == 2:
        # 2D: 绕Zaxisrotate
        if deg == False:
            angle = torch.deg2rad(rot_params[0])
        else:
            angle = rot_params[0]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # directlyconstructrotatematrix（maintaingradient）
        M[0, 0] = cos_a
        M[0, 1] = -sin_a
        M[1, 0] = sin_a
        M[1, 1] = cos_a

    elif D == 3:
        # 3D: 欧pullangle (ZYXsequential)
        if deg == False:
            angles = torch.deg2rad(rot_params)
        else:
            angles = rot_params
        # directlyconstructrotatematrix（avoidcreatenewtensor）
        cos_z = torch.cos(angles[2])
        sin_z = torch.sin(angles[2])
        cos_y = torch.cos(angles[1])
        sin_y = torch.sin(angles[1])
        cos_x = torch.cos(angles[0])
        sin_x = torch.sin(angles[0])

        # constructrotatematrixelement（maintaingradient）
        M[0, 0] = cos_z * cos_y
        M[0, 1] = cos_z * sin_y * sin_x - sin_z * cos_x
        M[0, 2] = cos_z * sin_y * cos_x + sin_z * sin_x

        M[1, 0] = sin_z * cos_y
        M[1, 1] = sin_z * sin_y * sin_x + cos_z * cos_x
        M[1, 2] = sin_z * sin_y * cos_x - cos_z * sin_x

        M[2, 0] = -sin_y
        M[2, 1] = cos_y * sin_x
        M[2, 2] = cos_y * cos_x

    elif D >= 4:
        # 4D+: useaxisangleexpressmethod（simplifyversion）
        angle = torch.deg2rad(rot_params[0])
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # directlymodifyrotatematrixpart（maintaingradient）
        M[0, 0] = cos_a
        M[0, 1] = -sin_a
        M[1, 0] = sin_a
        M[1, 1] = cos_a

    return M


def create_affine_matrix(rot_params, trans_vector, scale_params, D, device, dtype, deg=False):
    """
    based ondimensioncreatecompletelyneatimitateshoottransformmatrix（translate、rotate、wrongcut、scale）。

    parameter:
        rot_params: rotateparameter
        trans_vector: translatevector (D,)
        scale_params: scalebecause子 (D,)
        shear_params: wrongcutbecause子
        D: spacedimension (2 or 3)
    """
    # final (D+1)x(D+1) neattimetransformmatrix
    M = torch.eye(D + 1, dtype=dtype, device=device)

    # 1. 构造 DxD linetransformpart (A = R * H * S)
    # R: Rotation, H: Shear, S: Scale

    # -- scalematrix S --
    S = torch.diag(scale_params[:D])
    # -- rotatematrix R --
    R = torch.eye(D, dtype=dtype, device=device)
    if D == 2:
        angle = rot_params[0] if deg else torch.deg2rad(rot_params[0])
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        R[0, 0], R[0, 1] = cos_a, -sin_a
        R[1, 0], R[1, 1] = sin_a, cos_a
    elif D == 3:
        angles = rot_params if deg else torch.deg2rad(rot_params)
        cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])
        cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
        cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])

        # ZYX sequentialpullangle
        Rx = torch.eye(3, dtype=dtype, device=device)
        Rx[1, 1], Rx[1, 2] = cos_x, -sin_x
        Rx[2, 1], Rx[2, 2] = sin_x, cos_x

        Ry = torch.eye(3, dtype=dtype, device=device)
        Ry[0, 0], Ry[0, 2] = cos_y, sin_y
        Ry[2, 0], Ry[2, 2] = -sin_y, cos_y

        Rz = torch.eye(3, dtype=dtype, device=device)
        Rz[0, 0], Rz[0, 1] = cos_z, -sin_z
        Rz[1, 0], Rz[1, 1] = sin_z, cos_z

        R = Rz @ Ry @ Rx

    # 2. compositelinetransform: A = Rotation * Scale
    # note：matrixmultiplymethodsequentialwillaffectfinaleffect，R*H*S isause约stable
    A = R @ S

    # 3. willlinepartandtransformpartputenterneattimematrix M
    M[:D, :D] = A
    M[:D, D] = trans_vector[:D]

    return M

def compute_matching_score(X_A_trans, X_B, match_threshold, exp_P, alpha, similarity_mode):
    """computewhenbeforetransformbelowmatchscore"""
    eps = 1e-15
    sigma = match_threshold/2.0
    spatial_P = cal_single_P(X_A_trans, X_B, sigma=sigma, method='dist', use_threshold=False)

    # 结combineexpressioninformation (ifprovide)
    if exp_P is not None:
        exp_P_normalized = torch.clamp(exp_P, 0, 1)

        if similarity_mode == "probabilistic":
            combined_prob = cal_all_P(spatial_P, exp_P_normalized, alpha=alpha)
        else:
            exp_weight = 1 - alpha * exp_P_normalized
            combined_prob = spatial_P * exp_weight
    else:
        combined_prob = spatial_P

    # computeeachpointmaximummatchprobability
    combined_prob = torch.nan_to_num(combined_prob, nan=eps)
    max_prob1, _ = combined_prob.max(dim=0)
    max_prob2, _ = combined_prob.max(dim=1)
    max_prob = torch.cat([max_prob1, max_prob2], dim=0)
    valid_mask = max_prob >= max_prob.mean()/100.0
    valid_probs = max_prob[valid_mask]
    # returnmatchscore (allpointmaximummatchprobabilityand)
    score = valid_probs.mean()
    if valid_probs.numel() == 0 or torch.isnan(score):
        return torch.tensor(0.0, device=max_prob.device)
    return score


def maximize_matches(X_A, X_B, M_init,method='Bayesian',**kwargs):
    if method == 'Bayesian':
        return maximize_matches_Bayesian(X_A, X_B, M_init,**kwargs)
    else:
        return maximize_matches_RandomPerturbation(X_A, X_B, M_init,**kwargs)


def maximize_matches_RandomPerturbation(X_A, X_B, M_init,
                     match_threshold=None, angle_range=(-25, 25),
                     trans_range=None, num_samples=2048, exp_P=None,
                     alpha=0.5, similarity_mode="probabilistic"):
    """
    ininitialforneatfoundationon，throughmicrotransformmaximummatchpointforprobability

    parameter:
        X_A, X_B: spacecoordinatesmatrix
        M_init: initialtransformmatrix (D+1 x D+1)
        match_threshold: matchpointfordistancethreshold (automaticcompute)
        angle_range: rotateanglesamplingrange (degree)
        trans_range: translatedistancesamplingrange (automaticcompute)
        num_samples: samplingtimenumber
        exp_P: expressionsimilaritymatrix
        alpha: expressioninformationweightintensity (0-1)
        similarity_mode: similarityfusionpattern ("weighted" or "probabilistic")
    """
    eps = 1e-15

    device = X_A.device
    D = X_A.shape[1]
    dtype = X_A.dtype
    M_init = M_init.to(device)

    # automaticcomputedatascalefeature
    data_scale = calculate_data_scale(X_A, X_B)

    # setdefaultmatchthreshold (based onpointclouddegree)
    if match_threshold is None:
        match_threshold = 8.0 * data_scale['median_dist']

    # setdefaulttranslaterange (based onspacescale)
    if trans_range is None:
        scale_factor = 0.05 * data_scale['spatial_scale']
        trans_range = (-scale_factor, scale_factor)

    print(f"selfsuitableshouldparameterset: matchthreshold={match_threshold:.4f}, translaterange=[{trans_range[0]:.4f}, {trans_range[1]:.4f}]")

    # initialtransformafterpointcloud
    X_A_trans_init = transform_points(X_A, M_init)

    init_score = compute_matching_score(
        X_A_trans_init, X_B,
        match_threshold,
        exp_P, alpha, similarity_mode
    )
    best_score = init_score
    M_best = M_init.clone()
    # ininitialtransformnearsamplingmicroparameter
    for i in range(num_samples):
        # generaterandomperturbation (rotateangle + translatevector)
        angle = torch.FloatTensor(1).uniform_(angle_range[0], angle_range[1]).item()

        # aseachdimensiongenerateindependenttranslateperturbation
        trans = torch.FloatTensor(D).uniform_(trans_range[0], trans_range[1]).to(device)

        # constructperturbationtransformmatrix
        M_perturb = create_perturbation_matrix(angle, trans, D, device, dtype)

        # compositetransform: M_perturb * M_init
        M_current = M_perturb @ M_init

        # applicationtransform
        X_A_trans = transform_points(X_A, M_current)
        current_score = compute_matching_score(
            X_A_trans, X_B,
            match_threshold,
            exp_P, alpha, similarity_mode
        )
        # updatemostexcellentresult
        if current_score > best_score:
            improvement = current_score - best_score
            best_score = current_score
            M_best = M_current.clone()

    return M_best, best_score

#
#
# def maximize_matches_Bayesian(X_A, X_B, M_init,
#                               match_threshold=None, angle_range=(-25, 25),
#                               trans_range=None, n_calls=100, exp_P=None,
#                               alpha=0.5, similarity_mode="probabilistic"):
#     """
#     useBayesianoptimizationininitialforneatfoundationonmicrotransform，maximummatchpointforprobability
#
#     parameter:
#         X_A, X_B: spacecoordinatesmatrix (N x D)
#         M_init: initialtransformmatrix (D+1 x D+1)
#         match_threshold: matchpointfordistancethreshold (automaticcompute)
#         angle_range: rotateanglerange (degree)
#         trans_range: translatedistancerange (automaticcompute)
#         n_calls: Bayesianoptimizationevaluationtimenumber
#         exp_P: expressionsimilaritymatrix (N x M)
#         alpha: expressioninformationweightintensity (0-1)
#         similarity_mode: similarityfusionpattern ("weighted" or "probabilistic")
#     """
#     device = X_A.device
#     dtype = X_A.dtype
#     D = X_A.shape[1]
#     M_init = M_init.to(device)
#
#     # automaticcomputedatascalefeature
#     data_scale = calculate_data_scale(X_A, X_B)
#
#     # setdefaultmatchthreshold (based onpointclouddegree)
#     if match_threshold is None:
#         match_threshold = 1.5 * data_scale['median_dist']
#
#     # setdefaulttranslaterange (based onspacescale)
#     if trans_range is None:
#         scale_factor = 0.05 * data_scale['spatial_scale']
#         trans_range = (-scale_factor, scale_factor)
#
#     print(f"selfsuitableshouldparameterset: matchthreshold={match_threshold:.4f}, translaterange=[{trans_range[0]:.4f}, {trans_range[1]:.4f}]")
#
#     # definitionparameterspace (angle + Dtranslatedimension)
#     dimensions = [
#         Real(low=angle_range[0], high=angle_range[1], name='angle')
#     ]
#     for i in range(D):
#         dimensions.append(Real(low=trans_range[0], high=trans_range[1], name=f'trans_{i}'))
#
#     # cache机制avoidrepeatedcompute
#     cache = {}
#
#     def objective(params):
#         """goalfunction: computewhenbeforetransformbelowmatchscore"""
#         # tryfromcacheingetresult
#         params_key = tuple(params)
#         if params_key in cache:
#             return cache[params_key]
#
#         # solveparameter
#         angle = params[0]
#         trans = torch.tensor(params[1:], device=device, dtype=dtype)
#
#         # constructperturbationtransformmatrix
#         M_perturb = create_perturbation_matrix(angle, trans, D, device, dtype)
#
#         # compositetransform: M_perturb * M_init
#         M_current = M_perturb @ M_init
#
#         # applicationtransform
#         X_A_trans = transform_points(X_A, M_current)
#
#         # computematchscore
#         score = compute_matching_score(
#             X_A_trans, X_B,
#             match_threshold,
#             exp_P, alpha, similarity_mode
#         )
#
#         # Bayesianoptimizationminimumgoal，sowereturnnegativescore
#         result = -score.item()
#
#         # cacheresult
#         cache[params_key] = result
#         return result
#
#     # initialpoint (zeroperturbation)
#     x0 = [0.0] + [0.0] * D
#
#     # runBayesianoptimization
#     res = gp_minimize(
#         func=objective,
#         dimensions=dimensions,
#         n_calls=n_calls,
#         x0=x0,
#         random_state=42,
#         n_jobs=1,  # singlelinetoavoidGPU冲突
#         verbose=False
#     )
#
#     # extractionmostexcellentparameter
#     best_params = res.x
#     best_angle = best_params[0]
#     best_trans = torch.tensor(best_params[1:], device=device, dtype=dtype)
#
#     # constructmostexcellenttransformmatrix
#     M_perturb = create_perturbation_matrix(best_angle, best_trans, D, device, dtype)
#     M_best = M_perturb @ M_init
#
#     # computemostexcellentscore
#     X_A_trans_best = transform_points(X_A, M_best)
#     best_score = compute_matching_score(
#         X_A_trans_best, X_B,
#         match_threshold,
#         exp_P, alpha, similarity_mode
#     ).item()
#     return M_best, best_score
#
# def maximize_matches_Bayesian(X_A, X_B, M_init,
#                               match_threshold=None, angle_range=(-25, 25),
#                               trans_range=None, n_calls=100, exp_P=None,
#                               alpha=0.5, similarity_mode="probabilistic"):
#     """
#     useBayesianoptimizationininitialforneatfoundationonmicrotransform，maximummatchpointforprobability
#     support2D、3D、4Dandmoreheightendimensionspacedata
#
#     parameter:
#         X_A, X_B: spacecoordinatesmatrix (N x D)
#         M_init: initialtransformmatrix (D+1 x D+1)
#         match_threshold: matchpointfordistancethreshold (automaticcompute)
#         angle_range: rotateanglerange (degree) - 仅suitableused for2D/3D
#         trans_range: translatedistancerange (automaticcompute)
#         n_calls: Bayesianoptimizationevaluationtimenumber
#         exp_P: expressionsimilaritymatrix (N x M)
#         alpha: expressioninformationweightintensity (0-1)
#         similarity_mode: similarityfusionpattern ("weighted" or "probabilistic")
#     """
#     device = X_A.device
#     dtype = X_A.dtype
#     M_init = M_init.to(device)
#     D = X_A.shape[1]  # spacedimension
#
#     # automaticcomputedatascalefeature
#     data_scale = calculate_data_scale(X_A, X_B)
#
#     # setdefaultmatchthreshold (based onpointclouddegree)
#     if match_threshold is None:
#         match_threshold = 1.5 * data_scale['median_dist']
#
#     # setdefaulttranslaterange (based onspacescale)
#     if trans_range is None:
#         scale_factor = 0.05 * data_scale['spatial_scale']
#         trans_range = (-scale_factor, scale_factor)
#
#     print(f"selfsuitableshouldparameterset: matchthreshold={match_threshold:.4f}, translaterange=[{trans_range[0]:.4f}, {trans_range[1]:.4f}]")
#
#     # dynamiccreateparameterspace
#     dimensions = []
#
#     # based ondimensionaddrotateparameter
#     if D == 2:
#         # 2D: singlerotateangle
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
#     elif D == 3:
#         # 3D: three欧pullangle
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_x'))
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_y'))
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
#     elif D >= 4:
#         # 4D+: userotatevector (angle+axis) oronlyoptimizationtranslate
#         print(f"warning: {D}Dspaceusesimplifyrotateexpress")
#         # addaglobalrotateangle
#         dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rotation_angle'))
#
#     # addtranslateparameter (eachdimensiona)
#     for i in range(D):
#         dimensions.append(Real(low=trans_range[0], high=trans_range[1], name=f'trans_{i}'))
#
#     # cache机制avoidrepeatedcompute
#     cache = {}
#
#     def objective(params):
#         """goalfunction: computewhenbeforetransformbelowmatchscore"""
#         # tryfromcacheingetresult
#         params_key = tuple(params)
#         if params_key in cache:
#             return cache[params_key]
#
#         # solveparameter - based ondimensiondeterministicrotateparameterquantity
#         if D == 2:
#             rot_params = [params[0]]
#             trans_params = params[1:]
#         elif D == 3:
#             rot_params = params[:3]
#             trans_params = params[3:]
#         else:  # D >= 4
#             rot_params = [params[0]]  # onlyusearotateangle
#             trans_params = params[1:]
#
#         # constructperturbationtransformmatrix
#         M_perturb = create_perturbation_matrix(
#             torch.tensor(rot_params, device=device, dtype=dtype),
#             torch.tensor(trans_params, device=device, dtype=dtype),
#             D,
#             device,
#             dtype
#         )
#
#         # compositetransform: M_perturb * M_init
#         M_current = M_perturb @ M_init
#
#         # applicationtransform
#         X_A_trans = transform_points(X_A, M_current)
#
#         # computematchscore
#         score = compute_matching_score(
#             X_A_trans, X_B,
#             match_threshold,
#             exp_P, alpha, similarity_mode
#         )
#
#         # Bayesianoptimizationminimumgoal，sowereturnnegativescore
#         result = -score.item()
#
#         # cacheresult
#         cache[params_key] = result
#         return result
#
#     # initialpoint (zeroperturbation)
#     x0 = [0.0] * len(dimensions)
#
#     # runBayesianoptimization
#     res = gp_minimize(
#         func=objective,
#         dimensions=dimensions,
#         n_calls=n_calls,
#         x0=x0,
#         random_state=42,
#         n_jobs=1,  # singlelinetoavoidGPU冲突
#         verbose=False
#     )
#
#     # extractionmostexcellentparameter
#     best_params = res.x
#
#     # solvemostexcellentparameter - withobjectivefunctionone致
#     if D == 2:
#         best_rot = [best_params[0]]
#         best_trans = torch.tensor(best_params[1:], device=device, dtype=dtype)
#     elif D == 3:
#         best_rot = best_params[:3]
#         best_trans = torch.tensor(best_params[3:], device=device, dtype=dtype)
#     else:  # D >= 4
#         best_rot = [best_params[0]]
#         best_trans = torch.tensor(best_params[1:], device=device, dtype=dtype)
#
#     # constructmostexcellenttransformmatrix
#     M_perturb = create_perturbation_matrix(
#         torch.tensor(best_rot, device=device, dtype=dtype),
#         best_trans,
#         D,
#         device,
#         dtype
#     )
#     M_best = M_perturb @ M_init
#
#     # computemostexcellentscore
#     X_A_trans_best = transform_points(X_A, M_best)
#     best_score = compute_matching_score(
#         X_A_trans_best, X_B,
#         match_threshold,
#         exp_P, alpha, similarity_mode
#     ).item()
#
#     return M_best, best_score

# assumebelowlibraryandauxiliaryfunctionalreadysavein
# from skopt import gp_minimize
# from skopt.space import Real
# import torch
# def calculate_data_scale(X_A, X_B): ...
# def create_perturbation_matrix(rot_params, trans_params, D, device, dtype): ...
# def transform_points(X_A, M): ...
# def compute_matching_score(X_A_trans, X_B, threshold, exp_P, alpha, mode): ...


def maximize_matches_Bayesian(X_A, X_B, M_init,
                                        match_threshold=None, angle_range=(-25, 25),
                                        trans_range=None, n_calls=100, exp_P=None,
                                        alpha=0.5, similarity_mode="probabilistic",
                                        # --- used foriterationoptimizationnewparameter ---
                                        max_rounds=2, edge_threshold=0.95):
    """
    useiterationstyleBayesianoptimizationcomemicrotransform，maximummatchpointforprobability。
    ifsomeoneroundoptimizationmostexcellentsolvepositionatsearchspaceboundarynear，thenwilltoshouldsolveasfoundation，
    startnewoneroundoptimization。
    support2D、3D、4Dandmoreheightendimensionspacedata。

    parameter:
        (originalbeginparameterremains unchanged)
        ...
        max_rounds: optimizationmaximumroundnumber。
        edge_threshold: threshold (0-1)，used forbreakaparameterisno“inboundaryon”。
                        examplesuch as，0.95expressifmostexcellentvalueplaceatitsrangepart5%orpart5%inside，
                        thenasinboundaryon。
    """
    device = X_A.device
    dtype = X_A.dtype
    D = X_A.shape[1]  # spacedimension

    # --- asiterationloopperforminitialize ---
    M_current_init = M_init.to(device)
    current_round = 0
    best_score_overall = -float('inf')
    M_best_overall = M_current_init

    while current_round < max_rounds:
        print(f"\n--- startoptimization {current_round + 1}/{max_rounds} round ---")

        # whenbeforeroundtimegoalfunctionuse M_init isononeroundoptimizationresult，
        # orinoneroundtimeisuserspreadenterinitialmatrix。
        _M_init_round = M_current_init

        # inallroundtimeinmaintainparameter（such asthresholdandrange）one致
        # alsocanheavynewcompute，butmaintainonemoresimple、morestablehealthy。
        data_scale = calculate_data_scale(X_A, X_B)
        if match_threshold is None:
            _match_threshold = 1.5 * data_scale['median_dist']
        else:
            _match_threshold = match_threshold

        if trans_range is None:
            scale_factor = 0.05 * data_scale['spatial_scale']
            _trans_range = (-scale_factor, scale_factor)
        else:
            _trans_range = trans_range

        if current_round == 0:
            print(
                f"selfsuitableshouldparameterset: matchthreshold={_match_threshold:.4f}, translaterange=[{_trans_range[0]:.4f}, {_trans_range[1]:.4f}]")

        # asskoptdefinitionsearchspace (dimensions)
        dimensions = []
        if D == 2:
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
        elif D == 3:
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_x'))
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_y'))
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rot_z'))
        elif D >= 4:
            dimensions.append(Real(low=angle_range[0], high=angle_range[1], name='rotation_angle'))

        for i in range(D):
            dimensions.append(Real(low=_trans_range[0], high=_trans_range[1], name=f'trans_{i}'))

        cache = {}

        def objective(params):
            params_key = tuple(params)
            if params_key in cache:
                return cache[params_key]

            if D == 2:
                rot_params, trans_params = [params[0]], params[1:]
            elif D == 3:
                rot_params, trans_params = params[:3], params[3:]
            else:
                rot_params, trans_params = [params[0]], params[1:]

            M_perturb = create_perturbation_matrix(
                torch.tensor(rot_params, device=device, dtype=dtype),
                torch.tensor(trans_params, device=device, dtype=dtype),
                D, device, dtype
            )
            # inononeroundresultfoundationonapplicationperturbation
            M_current = M_perturb @ _M_init_round
            X_A_trans = transform_points(X_A, M_current)

            score = compute_matching_score(
                X_A_trans, X_B, _match_threshold, exp_P, alpha, similarity_mode
            )
            result = -score.item()
            cache[params_key] = result
            return result

        res = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            x0=[0.0] * len(dimensions),
            random_state=42 + current_round,  # aseachroundoptimizationmorerandomkind子
            n_jobs=1,
            verbose=False
        )

        best_params_round = res.x

        # --- 判breakisnoneedperformbelowoneround决策logic ---
        is_on_edge = False
        for i, param in enumerate(best_params_round):
            dim = dimensions[i]
            low, high = dim.low, dim.high
            # checkparameterisnoinboundarythresholdinside
            if param <= low + (1.0 - edge_threshold) * (high - low) or \
                    param >= low + edge_threshold * (high - low):
                print(f"  -> warning: parameter '{dim.name}' ({param:.4f}) alreadyconnectnearitssearchrange [{low:.4f}, {high:.4f}] boundary。")
                is_on_edge = True
                break

        # constructwhenbeforeroundtimeoptimaltransformmatrix
        if D == 2:
            best_rot, best_trans = [best_params_round[0]], best_params_round[1:]
        elif D == 3:
            best_rot, best_trans = best_params_round[:3], best_params_round[3:]
        else:
            best_rot, best_trans = [best_params_round[0]], best_params_round[1:]

        M_perturb_best = create_perturbation_matrix(
            torch.tensor(best_rot, device=device, dtype=dtype),
            torch.tensor(best_trans, device=device, dtype=dtype),
            D, device, dtype
        )
        M_best_round = M_perturb_best @ _M_init_round

        # goalfunctionreturnisnegativedivide，sothis里取反
        best_score_round = -res.fun

        print(f" {current_round + 1} roundresult: mostheightenscore = {best_score_round:.6f}")

        # updateglobalfindtooptimalresult
        if best_score_round > best_score_overall:
            best_score_overall = best_score_round
            M_best_overall = M_best_round

        current_round += 1

        # ifinboundaryonandreachtomaximumroundnumber，thencontinue
        if is_on_edge and current_round < max_rounds:
            print(f"mostexcellentsolvepositionatboundary。preparenewoneroundoptimization。")
            # belowoneround M_init theniswhenbeforeroundtimeoptimalresult
            M_current_init = M_best_round
        else:
            if is_on_edge:
                print("alreadyreachtomaximumoptimizationroundnumber。optimizationend。")
            else:
                print("mostexcellentsolvealreadyinsearchspaceinsidefindto。optimizationend。")
            break  # retreatexit while loop

    print("\n--- iterationstyleBayesianoptimizationcompleted ---")
    print(f"finalfindtoglobalmostheightenscore: {best_score_overall:.6f}")
    return M_best_overall, best_score_overall

import os
import shutil


def copy_py_files(destination_dir):
    # getwhenbeforedirectory
    current_dir = os.getcwd()

    # ensuregoaldirectorysavein
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"createdirectory: {destination_dir}")

    # timewhenbeforedirectorybelowallfile
    for filename in os.listdir(current_dir):
        # checkisnoas.pyfile
        if filename.endswith('.py') and os.path.isfile(os.path.join(current_dir, filename)):
            # constructfileandgoalfilecompletelyneatpath
            source_path = os.path.join(current_dir, filename)
            dest_path = os.path.join(destination_dir, filename)

            # repeatfile
            shutil.copy2(source_path, dest_path)
            print(f"alreadyrepeat制: {filename} -> {destination_dir}")

    print("repeatcompleted!")

from sklearn.cluster import KMeans

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist


def KMeans_inducing_points(pos, n_inducing_points=100, min_distance=1e-3):
    """useKDtreeoptimization诱导point筛选algorithm（correctionversion）"""
    # ensureinputisNumPyarray
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos)

    # ifpoint countquantityless thanrequire，directlyreturnallpoint
    if len(pos) <= n_inducing_points:
        return pos

    # K-meansclustering
    kmeans = KMeans(n_clusters=min(n_inducing_points, len(pos)), random_state=42)
    kmeans.fit(pos)
    inducing_points = kmeans.cluster_centers_

    # ifonlyapointornotneed筛选，directlyreturn
    if len(inducing_points) <= 1 or min_distance <= 0:
        return inducing_points

    # constructKDtreeperformheighteneffectnearsearch
    tree = KDTree(inducing_points)

    # useKDTreequery_ball_pointmethod替代query_radius
    neighbors = tree.query_ball_point(inducing_points, r=min_distance)

    # 贪heartalgorithm筛选point - moreheighteneffectimplementation
    keep_indices = []
    visited = set()

    for i in range(len(inducing_points)):
        if i not in visited:
            keep_indices.append(i)
            # markerallinwhenbeforepointdomaininsidepointasalreadyaccess
            visited.update(neighbors[i])

    return inducing_points[keep_indices]


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import DeformableRegistration
from scipy.interpolate import griddata
import time


def calculate_displacement_field(source, target, normalize=True, add_noise=True, noise_level=0.005,
                                 max_iterations=100, beta=5, lam=1e-3):
    """
    useCPDalgorithmcomputefrompointcloudtogoalpointcloudpositionshift场

    parameter:
    source: 源pointcloud，shapeas (N, D)
    target: goalpointcloud，shapeas (M, D)
    normalize: isnoforcoordinatesperformnormalizationhandle
    add_noise: isnoaddrandommicro扰
    noise_level: microamplitude，forpointcloudrangeratioexample
    max_iterations: CPDalgorithmmaximumiterationtimenumber
    beta: controlshapesmooth（valuelarge，variationalshapesmooth）
    lam: regularizationrelatenumber，average衡simulatecombineprecisionwithsmoothdegree
    """
    start_time = time.time()

    # saveoriginalbeginpointcloudused forsubsequentrecovery
    # original_source = source.copy()
    # original_target = target.copy()

    # 1. normalizationhandle（optional）
    if normalize:
        # computepointcloudstatisticalinformation
        source_mean = np.mean(source, axis=0)
        source_std = np.std(source, axis=0)

        # normalizationpointcloudandgoalpointcloud
        source = (source - source_mean) / (source_std + 1e-8)
        target = (target - source_mean) / (source_std + 1e-8)

    # 2. addrandommicro扰（optional）
    if add_noise:
        # computepointcloudrange
        source_range = np.max(source, axis=0) - np.min(source, axis=0)

        # addheightennoise
        noise = np.random.normal(0, noise_level * source_range, source.shape)
        source = source + noise

    # 3. createandrunCPDregistration
    reg = DeformableRegistration(X=target, Y=source, max_iterations=max_iterations, beta=beta, lam=lam)
    deformed_source, (_, _) = reg.register()

    # 4. computepositionshiftquantity
    displacement = deformed_source - source

    # 5. ifperformnormalization，recoveryoriginalbegincoordinates
    if normalize:
        deformed_source = deformed_source * (source_std + 1e-8) + source_mean
        displacement = displacement * (source_std + 1e-8)

    # computecomputetime
    elapsed_time = time.time() - start_time
    print(f"CPDregistrationcompleted，耗time: {elapsed_time:.2f}second")

    return displacement, deformed_source


import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import scanpy as sc

import numpy as np
from sklearn.cluster import MiniBatchKMeans  # moreheighteneffectclusteringalgorithm
from scipy.spatial import KDTree
import scanpy as sc
from tqdm import tqdm  # advancedegreeitem


def cluster_downsample_adata(adata, spatial_key='spatial', target_size=10000):
    """
    foradataobjectuseclusteringinheartmethodperformdownsample

    parameter:
        adata: AnnDataobject
        spatial_key: spacecoordinatesinobsminkeyname，defaultas'spatial'
        target_size: goalsamplingpoint count，defaultas10000

    return:
        downsampleafterAnnDataobject
    """
    # 1. inputvalidation
    if spatial_key not in adata.obsm:
        raise ValueError(f"spacecoordinateskey '{spatial_key}' notsaveinat adata.obsm in")

    coords = adata.obsm[spatial_key]
    if coords.ndim != 2:
        raise ValueError("spacecoordinatesshouldastwo维array")

    n_cells = adata.shape[0]
    if n_cells <= target_size:
        print(f"datapoint countquantity ({n_cells}) alreadyless than or equalgoalquantity ({target_size})，noneeddownsample")
        return adata.copy()  # return副thistomaintainoriginalbegindatanotvariational

    print(f"startdownsample: {n_cells} -> {target_size}")

    # 2. usemoreheighteneffectMiniBatchKMeans
    print(f"executeMiniBatchKMeansclustering，clusteringnumber: {target_size}")
    kmeans = MiniBatchKMeans(
        n_clusters=target_size,
        random_state=42,
        batch_size=min(1000, n_cells // 10)  # selfsuitableshouldbatchsize
    )
    kmeans.fit(coords)
    cluster_centers = kmeans.cluster_centers_

    # 3. batchquantityquerynearest neighbor
    print("findnearest neighborpoint...")
    kdtree = KDTree(coords)

    # usebatchquantityqueryimproveefficiency
    _, nearest_indices = kdtree.query(cluster_centers, k=1, workers=-1)  # useallCPUcore

    # 4. Handle duplicate points and supplement
    unique_indices = np.unique(nearest_indices)
    selected_indices = unique_indices

    if len(unique_indices) < target_size:
        print(f"Found {len(unique_indices)} unique indices, supplementing {target_size - len(unique_indices)} points")
        remaining_indices = np.setdiff1d(np.arange(n_cells), unique_indices)

        # Ensure there are enough points to supplement
        n_needed = min(len(remaining_indices), target_size - len(unique_indices))

        additional_indices = np.random.choice(
            remaining_indices,
            size=n_needed,
            replace=False
        )
        selected_indices = np.concatenate([unique_indices, additional_indices])
    elif len(unique_indices) > target_size:
        # Theoretically won't happen, but add protection
        selected_indices = unique_indices[:target_size]

    # 5. Create downsampled adata
    downsampled_adata = adata[selected_indices].copy()
    print(f"Downsampling complete: {n_cells} -> {len(downsampled_adata)}")

    # Optional: add downsampling information to obs
    downsampled_adata.obs['downsampled'] = True

    return downsampled_adata


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scanpy as sc


def fast_spatial_downsample(adata, spatial_key='spatial',spatial_dim=2, target_size=10000, grid_strategy='adaptive'):
    """
    Efficient spatial downsampling method
    Parameters:
        adata: AnnData object
        spatial_key: Key name for spatial coordinates in obsm
        target_size: Target sampling point count
        grid_strategy: Grid strategy ('auto', 'fixed', 'adaptive')

    Returns:
        Downsampled AnnData object
    """
    # 1. Input validation
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial coordinate key '{spatial_key}' does not exist in adata.obsm")

    coords = adata.obsm[spatial_key]
    if coords.shape[-1] != spatial_dim:
        coords_old = coords
        coords = coords[:,:spatial_dim]
    else:
        coords_old = coords
        coords = coords

    n_cells = adata.shape[0]
    if n_cells <= target_size:
        print(f"Data point count ({n_cells}) ≤ target count ({target_size}), no downsampling needed")
        return adata.copy()

    print(f"Starting efficient downsampling: {n_cells} -> {target_size}")

    # 2. Use grid method for initial sampling
    if grid_strategy == 'auto':
        # Automatically select strategy based on data scale
        grid_strategy = 'adaptive' if n_cells > 1000000 else 'fixed'

    if grid_strategy == 'fixed':
        # Fixed grid size
        grid_size = int(np.sqrt(target_size) * 2)
        sampled_indices = grid_downsample(coords, grid_size)
    else:
        # Adaptive grid size
        sampled_indices = adaptive_grid_downsample(coords, target_size)

        # 3. If grid sampling points are insufficient, supplement with random sampling
    if len(sampled_indices) < target_size:
        remaining_indices = np.setdiff1d(np.arange(n_cells), sampled_indices)
        n_needed = min(len(remaining_indices), target_size - len(sampled_indices))
        additional_indices = np.random.choice(remaining_indices, n_needed, replace=False)
        sampled_indices = np.concatenate([sampled_indices, additional_indices])

    # 4. Create downsampled adata
    downsampled_adata = adata[sampled_indices].copy()
    downsampled_adata.obs['downsampled'] = True

    print(f"Downsampling complete: {n_cells} -> {len(downsampled_adata)}")
    return downsampled_adata


def grid_downsample(coords, grid_size):
    """Downsample using fixed grid"""
    # 1. Calculate spatial range
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)

    # 2. creategrid
    grid_x = np.linspace(min_vals[0], max_vals[0], grid_size)
    grid_y = np.linspace(min_vals[1], max_vals[1], grid_size)

    # 3. aseachpointallocationgridID
    grid_ids = pd.cut(coords[:, 0], bins=grid_x, labels=False, include_lowest=True)
    grid_ids = grid_ids.astype(str) + "_" + pd.cut(coords[:, 1], bins=grid_y, labels=False, include_lowest=True).astype(
        str)

    # 4. fromeachgridunitinrandomchooseapoint
    unique_grids = np.unique(grid_ids)
    sampled_indices = []

    for grid_id in unique_grids:
        cell_indices = np.where(grid_ids == grid_id)[0]
        if len(cell_indices) > 0:
            sampled_indices.append(np.random.choice(cell_indices))

    return np.array(sampled_indices)


def adaptive_grid_downsample(coords, target_size):
    """useselfsuitableshouldgridperformdownsample"""
    # 1. computeinitialgridsize
    n_cells = coords.shape[0]
    grid_size = int(np.sqrt(target_size))

    # 2. createinitialgrid
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)

    # 3. passreturnfinedivideheightendegreeareadomain
    sampled_indices = []
    queue = [(min_vals, max_vals)]

    while queue and len(sampled_indices) < target_size:
        current_min, current_max = queue.pop(0)

        # inwhenbeforeareadomaininsidechoosepoint
        in_region = np.all((coords >= current_min) & (coords <= current_max), axis=1)
        region_indices = np.where(in_region)[0]

        if len(region_indices) == 0:
            continue

        # ifareadomaininsidepoint countfew，directlyrandoma
        if len(region_indices) <= max(1, target_size // (grid_size * 2)):
            sampled_indices.append(np.random.choice(region_indices))
            continue

        # otherwisefinedivideareadomain
        mid_x = (current_min[0] + current_max[0]) / 2
        mid_y = (current_min[1] + current_max[1]) / 2

        # createfour子areadomain
        sub_regions = [
            (current_min, [mid_x, mid_y]),
            ([current_min[0], mid_y], [mid_x, current_max[1]]),
            ([mid_x, current_min[1]], [current_max[0], mid_y]),
            ([mid_x, mid_y], current_max)
        ]

        # hitchaosareadomainsequentialtoavoidbiastoward特stableareadomain
        np.random.shuffle(sub_regions)
        queue.extend(sub_regions)

    return np.array(sampled_indices)


# assume clean_adata functionalreadydefinition
# from anndata import AnnData
# def clean_adata(adata: AnnData):
#     # thisisaexample，you may need tobased onactualsituationadjustment
#     if 'highly_variable' in adata.var:
#         del adata.var['highly_variable']
#     return adata

def concat_adata(adata_list):
    """
    mergea AnnData objectlist，at the same timeretainaobjectingeneforsequential。
    """
    if not adata_list:
        raise ValueError("input adata_list notcanasempty")

    # --- MODIFIED: step 1 ---
    # first，仍然use set heighteneffectcomputeexitalltotalsamegenehandset
    # thissequentialwill丢失，butcloserelate，weonlyuseitcomebreakmember资格
    common_genes_set = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        common_genes_set.intersection_update(adata.var_names)

    print(f"findto {len(common_genes_set)} totalsamegene")
    # --- MODIFIED: step 2 ---
    # then，toa AnnData objectgenesequentialasreference，
    # generatearetainoriginalbeginforsequentialfinaltotalsamegenelist。
    reference_var_names = adata_list[0].var_names
    ordered_common_genes = [gene for gene in reference_var_names if gene in common_genes_set]
    # --- MODIFIED: step 3 ---
    # 筛选each AnnData object，makeitsonlyretainrowgoodordertotalsamegene
    # thisoneensureallmerge anndata object .var_names completelyallone致（packagesequential）
    filtered_adata_list = []
    for adata in adata_list:
        # usenewgenerate、haveordergenelistperformcutslice
        filtered_adata = adata[:, ordered_common_genes].copy()
        filtered_adata_list.append(filtered_adata)

    # --- UNCHANGED: step 4 ---
    # subsequentmergestepremains unchanged
    if len(filtered_adata_list) > 1:
        filtered_adata_list = [clean_adata(adata) for adata in filtered_adata_list]
        combined_adata = filtered_adata_list[0].concatenate(
            *filtered_adata_list[1:],
            join='outer',  # becauseaswemovekeepvar_namesone致，'outer'and'inner'effectsame
            index_unique='-',
            fill_value=0,
            batch_categories=[f"batch_{i}" for i in range(len(filtered_adata_list))])
    else:
        combined_adata = filtered_adata_list[0]

    return combined_adata


def clean_adata(adata):
    """clearreasonmaylead tomergeproblemdatastructure"""
    # deleteall邻居graphinformation
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]
    for key in list(adata.uns.keys()):
        del adata.uns[key]

    # checkandclearreasonmayproblemdimension
    for key in list(adata.obsm.keys()):
        if len(adata.obsm[key].shape) < 2:
            print(f"Removing problematic obsm['{key}'] with shape {adata.obsm[key].shape}")
            del adata.obsm[key]

    # clearreason layers
    for key in list(adata.layers.keys()):
        if len(adata.layers[key].shape) < 2:
            print(f"Removing problematic layers['{key}'] with shape {adata.layers[key].shape}")
            del adata.layers[key]

    return adata


import pandas as pd  # needenter pandas


def get_concatenated_tensor(adata_list, slices, dtype, data_extractor):
    """
    aauxiliaryfunction，used forheighteneffectfrom adata_list inextractiondataandconnecta Tensor。
    thisversionalreadyupdate，canautomatichandle pandas.Series type。

    parameter:
        data_extractor: alambdafunction，definitionhowfromsingleadataobjectinextractiondata, e.g.,
                        lambda adata: adata.obsm['spatial']
    """
    tensor_list = []
    for s in slices:
        # from anndata objectinextractiondata
        extracted_data = data_extractor(adata_list[s])

        # --- added改造logic ---
        # checkextractionexitdataisnoas Series type，ifis，thenconvertas numpy array
        if isinstance(extracted_data, pd.Series) or isinstance(extracted_data, pd.DataFrame):
            numpy_data = extracted_data.values
        else:
            numpy_data = extracted_data
        tensor_list.append(torch.from_numpy(numpy_data).to(dtype))
    return torch.cat(tensor_list, dim=0)

def auto_batch_size(N_train_all, dim=2):
    if N_train_all <= 1024 * 1:
        batch_size = 128
    elif N_train_all <= 1024 * 2:
        batch_size = 256
    elif N_train_all <= 1024 * 8:
        batch_size = 512
    elif N_train_all <= 2048 * 16 or dim > 2:
        batch_size = 1024
    else:
        batch_size = 2048
    print(f'Batch size: {batch_size}')
    return batch_size

import math
def get_cosine_schedule_with_warmup(current_epoch, warmup_epochs, total_epochs, start_epoch=0, final_lr_scale=0.05):
    """
    createahavehotperiodremaining弦learningrateschedulingfunction。
    """
    assert final_lr_scale < 1.0
    current_epoch = current_epoch - start_epoch
    total_epochs = total_epochs - start_epoch
    if current_epoch>total_epochs:
        current_epoch = total_epochs
    if current_epoch < 0:
        return 1.0
    if current_epoch < warmup_epochs:
        # linehot
        return float(current_epoch) / float(max(1, warmup_epochs)) * 5
    else:
        # remaining弦衰subtract
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # from1.0衰subtracttofinal_lr_scale
        return final_lr_scale + (1.0 - final_lr_scale) * cosine_decay


def clean_metadata_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    ahealthystrongfunction，used forclearreason AnnData 元data DataFrames (obs or var)。
    - willallnumerictype NaN paddingas 0。
    - willallnonumeric（object、string、classification）type NaN paddingas 'NA'。
    - cancorrecthandle 'Categorical' datatype，avoidwrong。

    Args:
        df: a pandas DataFrame，examplesuch as adata.obs or adata.var。

    Returns:
        clearreasonafter pandas DataFrame。
    """
    print(f"startclearreason DataFrame，totalhave {df.shape[1]} column...")

    # time历 DataFrame eachonecolumn
    for col in df.columns:
        column_series = df[col]

        # 1. use pd.api.types.is_numeric_dtype performstablehealthynumerictypecheck
        # thiscancover float64, float32, int64, int32 etcallnumerictype
        if pd.api.types.is_numeric_dtype(column_series):
            # ifisnumerictype，use 0 padding
            df[col] = column_series.fillna(0)
        else:
            # 2. forallnonumerictype

            # 2a. first，专门handle Categorical typemayemitproblem
            if pd.api.types.is_categorical_dtype(column_series):
                # if 'NA' alsonotisacombinemethodclassother，thenaddit
                if 'NA' not in column_series.cat.categories:
                    # .cat.add_categories() returnanew Series，wewillitsvaluereturngo
                    df[col] = column_series.cat.add_categories(['NA'])

            # 2b. appearin，canallasallnonumerictype（packagealreadyhandleCategorical）padding 'NA'
            df[col] = df[col].fillna('NA')

    print("clearreasoncompleted。")
    return df


import scanpy as sc
import pandas as pd
import numpy as np
import re


def split_rna_atac(adata: sc.AnnData, peak_regex: str = r'^(chr)?[\w]+:\d+-\d+$'):
    """
    based on var_names willcontains RNA and ATAC data AnnData objectsplitdivideastwo。

    thisfunctionassume ATAC peaks namename遵循 'chr:start-end' format，而genenamethennot遵循。

    Args:
        adata (sc.AnnData): containsmixcombinedata AnnData object。
        peak_regex (str): used forrecognition ATAC peak namenamepositivethenexpressionstyle。

    Returns:
        (sc.AnnData, sc.AnnData):
        atuple，containstwo AnnData object：(adata_rna, adata_atac)。
    """
    print(f"originalbegin AnnData objectdimension: {adata.shape}")

    # 1. createabooleancodecomerecognition ATAC peaks
    # .str.match() willfor var_names ineachnamecharacterapplicationpositivethenexpressionstyle
    # returnabooleanvalue Pandas Series
    is_atac_mask = adata.var_names.str.match(peak_regex)

    # checkisnohaveanywhatmatchitem
    n_atac_features = np.sum(is_atac_mask)
    if n_atac_features == 0:
        raise ValueError("error：based onprovidepositivethenexpressionstyle，未in var_names infindtoanywhat ATAC peak。请checkdataorpositivethenexpressionstyle。")

    print(f"recognitionto {n_atac_features}  ATAC peaks。")
    print(f"recognitionto {adata.shape[1] - n_atac_features}  RNA gene。")

    # 2. usecodeperformcutslice
    # anndata[:, is_atac_mask] chooseallcellandallas True feature (ATAC)
    adata_atac = adata[:, is_atac_mask].copy()

    # anndata[:, ~is_atac_mask] chooseallcellandallas False feature (RNA)
    # `~` signisboolean取反operation
    adata_rna = adata[:, ~is_atac_mask].copy()

    # 3. asnew AnnData objectadddescribeinformation (optionalbutpush荐)
    adata_rna.uns['modality'] = 'RNA'
    adata_atac.uns['modality'] = 'ATAC'

    print("-" * 30)
    print(f"splitdivideafter RNA AnnData dimension: {adata_rna.shape}")
    print(f"splitdivideafter ATAC AnnData dimension: {adata_atac.shape}")

    return adata_rna, adata_atac


def preprocessing_atac(
        adata,
        min_genes=None,
        min_cells=0.01,
        n_top_genes=30000,
        target_sum=None,
        log=None
):
    """
    preprocessing
    """
    print('Raw dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
    adata.X[adata.X > 0] = 1
    if log: log.info('Filtering cells')
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if log: log.info('Filtering genes')
    if min_cells:
        if min_cells < 1:
            min_cells = min_cells * adata.shape[0]
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if n_top_genes:
        if log: log.info('Finding variable features')
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
    if log: log.info('Batch specific maxabs scaling')
    print('Processed dataset shape: {}'.format(adata.shape))
    return adata


import numpy as np
import torch
import scanpy as sc
import pandas as pd  # ensureenter pandas
import numpy as np
import torch
import pandas as pd  # ensureenter pandas


def filter_expression_data(exp_A, exp_B, X_A, X_B):
    """
    forinputexpressionandcoordinatesdataperformstrictfilter。
    - cellfilter: retaineachselfsampleinexpressionquantityand > 0 cell。
    - genefilter: 仅retainintwosampleinexpressionquantityandall > 0 totalsamegene。
    """
    print("Pre-filtering data with strict common gene policy...")

    # 1. ensurealldataallas Numpy array (logicnotvariational)
    if isinstance(exp_A, torch.Tensor):
        exp_A = exp_A.cpu().detach().numpy()
        X_A = X_A.cpu().detach().numpy()
    if isinstance(exp_B, torch.Tensor):
        exp_B = exp_B.cpu().detach().numpy()
        X_B = X_B.cpu().detach().numpy()

    # 2. independentfiltereachsamplein“expressionquantityandas0”cell (logicnotvariational)
    sums_per_cell_A = exp_A.sum(axis=1)
    sums_per_cell_B = exp_B.sum(axis=1)
    keep_cells_mask_A = sums_per_cell_A > 0
    keep_cells_mask_B = sums_per_cell_B > 0

    n_obs_A_before, n_obs_B_before = exp_A.shape[0], exp_B.shape[0]

    exp_A = exp_A[keep_cells_mask_A, :]
    X_A = X_A[keep_cells_mask_A, :]
    exp_B = exp_B[keep_cells_mask_B, :]
    X_B = X_B[keep_cells_mask_B, :]

    print(f"Filtered cells in A (sum > 0): {n_obs_A_before} -> {exp_A.shape[0]}")
    print(f"Filtered cells in B (sum > 0): {n_obs_B_before} -> {exp_B.shape[0]}")

    # ensuretwomatrixinperformgenefilterbeforehavesamegenenumber
    if exp_A.shape[1] != exp_B.shape[1]:
        raise ValueError("Expression matrices must have the same number of genes before gene filtering.")

    n_vars_before = exp_A.shape[1]

    # 3. <<< corelogicmodify：findfindtotalsameexcellentgene >>>
    # divideothercomputeeachsampleineachgeneand
    sums_per_gene_A = exp_A.sum(axis=0)
    sums_per_gene_B = exp_B.sum(axis=0)

    # divideothercreatebooleancode
    keep_genes_mask_A = sums_per_gene_A > 0
    keep_genes_mask_B = sums_per_gene_B > 0

    # uselogic“with”(&)operationfindtomustat the same timefullsufficienttwoconditiongene
    # thisthenis“handset”operation，ensuregeneintwosampleinallhaveexpression
    final_keep_genes_mask = keep_genes_mask_A & keep_genes_mask_B

    # 4. usetotalsamegenecodecomefiltertwosampleexpressionmatrix
    exp_A = exp_A[:, final_keep_genes_mask]
    exp_B = exp_B[:, final_keep_genes_mask]

    print(f"Filtered genes (sum > 0 in BOTH samples): {n_vars_before} -> {exp_A.shape[1]}")

    return exp_A, exp_B, X_A, X_B


import numpy as np
import torch
from typing import Tuple, Union


def flattened_to_simg(
        flattened_data: Union[np.ndarray, torch.Tensor],
        shape: Tuple[int, int, int]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Restore flattened 2D image data to 4D simg tensor (N, C, H, W).

    Args:
        flattened_data (Union[np.ndarray, torch.Tensor]):
            Flattened image data with shape (n_samples, n_features), where n_features = C * H * W.
        shape (Tuple[int, int, int]):
            Target image shape in format (C, H, W), e.g., (3, 32, 32).

    Returns:
        Union[np.ndarray, torch.Tensor]:
            Restored 4D image data with shape (n_samples, C, H, W).

    Raises:
        ValueError: If the number of features in flattened data doesn't match the target shape.
    """
    n_features = flattened_data.shape[1]
    expected_features = shape[0] * shape[1] * shape[2]

    if n_features != expected_features:
        raise ValueError(
            f"Feature count mismatch. Flattened data has {n_features} features, "
            f"but target shape {shape} requires {expected_features} features."
        )

    # -1 will automatically infer the number of samples (n_samples)
    return flattened_data.reshape(-1, *shape)


def simg_to_flattened(
        simg_data: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Flatten 4D simg image tensor (N, C, H, W) to 2D data (N, C*H*W).

    Args:
        simg_data (Union[np.ndarray, torch.Tensor]):
            4D image data with shape (n_samples, C, H, W).

    Returns:
        Union[np.ndarray, torch.Tensor]:
            Flattened 2D data with shape (n_samples, C * H * W).

    Raises:
        ValueError: If input data dimension is not 4.
    """
    if simg_data.ndim != 4:
        raise ValueError(
            f"Input data must be 4-dimensional (N, C, H, W), but received {simg_data.ndim}-dimensional data."
        )

    # simg_data.shape[0] is the number of samples (N)
    # -1 will automatically calculate the product of C * H * W
    return simg_data.reshape(simg_data.shape[0], -1)


import pandas as pd
import numpy as np
import anndata as ad
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Union

#
# def _crop_and_flatten_patches(
#         full_img: np.ndarray,
#         coords: np.ndarray,
#         patch_size: int
# ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
#     """
#     aauxiliaryfunction，used forfromallsizegraphpixelincrop、transposeandexpandgraphpixel block。
#
#     Args:
#         full_img (np.ndarray): allsizegroupgraphpixel (H, W, C)。
#         coords (np.ndarray): cell/斑pointinheartcoordinates (N, 2)，formatas (y, x)。
#         patch_size (int): eachgraphpixel blockpositivesquareshapeedgegrow。
#
#     Returns:
#         Tuple[np.ndarray, Tuple[int, int, int]]:
#         - flattened_patches (np.ndarray): expandaftergraphpixel blockdata (N, C*H*W)。
#         - patch_shape (Tuple[int, int, int]): singlegraphpixel blockshape (C, H, W)。
#     """
#     if patch_size % 2 != 0:
#         raise ValueError("patch_size mustisnumber。")
#
#     n_channels = full_img.shape[2]
#     patch_shape = (n_channels, patch_size, patch_size)
#     flattened_dim = n_channels * patch_size * patch_size
#
#     # asallingraphpixeledgeperformcrop，wefirstforgraphpixelperformzeropadding
#     pad_width = patch_size//2
#     img_padded = np.pad(
#         full_img,
#         pad_width=((patch_size, patch_size), (patch_size, patch_size), (0, 0)),
#         mode='constant',
#         constant_values=0
#     )
#
#     # adjustmentcoordinatestosuitableshouldpaddingaftergraphpixel
#     coords_padded = coords + patch_size
#
#     flattened_patches = np.zeros((len(coords), flattened_dim), dtype=np.float32)
#
#     patch_transposed_list = np.zeros((len(coords), 3, patch_size, patch_size), dtype=np.float32)
#
#     print(f"currentlyfrom {len(coords)} coordinatespointcrop {patch_size}x{patch_size} graphpixel block...")
#     for i, (y, x) in tqdm(enumerate(coords_padded), total=len(coords)):
#         # willcoordinatesconvertasintegerindex
#         y, x = int(y), int(x)
#
#         # definitioncropareadomain
#         y_start, y_end = y - pad_width, y + pad_width
#         x_start, x_end = x - pad_width, x + pad_width
#
#         # cropgraphpixel block
#         patch = img_padded[y_start:y_end, x_start:x_end, :]
#
#         # transposetomatch (C, H, W) format
#         patch_transposed = patch.transpose(2, 0, 1)
#         patch_transposed_list[i] = patch_transposed
#         # expandandstorage
#         flattened_patches[i] = patch_transposed.flatten()
#
#     return flattened_patches, coords_padded, patch_shape,img_padded
#
#
# def create_img_adata_from_data(
#         full_img: np.ndarray,
#         barcodes: np.ndarray,
#         img_coordinates: np.ndarray,
#         spatial_coords: np.ndarray = None,
#         patch_size: int = 32
# ) -> ad.AnnData:
#     """
#     fromalreadyreadtomemorygraphpixelandpositionplacedataconstructatographpixelfeatureasXAnnDataobject。
#     Args:
#         full_img (np.ndarray): allsizegroupgraphpixelNumpyarray, shapeas (H, W, C)。
#         positions_df (pd.DataFrame):
#             grouppositionplaceDataFrame，structureshouldwith 'tissue_positions_list.csv' classseem。
#             functionwillfalsestable：
#             - 0columniscell/斑pointbarcode。
#             - 4columnisycoordinates (pixel coordinate)。
#             - 5columnisxcoordinates (pixel coordinate)。
#         patch_size (int): eachgraphpixel blockedgegrow。defaultas32。
#
#     Returns:
#         ad.AnnData: constructgoodAnnDataobject。
#     """
#     assert barcodes.shape[0] == img_coordinates.shape[0]
#     assert img_coordinates.shape[1] == 2
#     if not spatial_coords is None:
#         assert spatial_coords.shape[0] == barcodes.shape[0]
#     print("\n--- step 2/3: cropandexpandgraphpixel block ---")
#     # callauxiliaryfunctiongetfeaturematrix X andgraphpixel blockshape
#     feature_matrix, coords_padded, patch_shape, img_padded = _crop_and_flatten_patches(
#         full_img = full_img,
#         coords=img_coordinates,
#         patch_size=patch_size
#     )
#     print("\n--- step 3/3: constructAnnDataobject ---")
#     # createAnnDataobject
#     adata = ad.AnnData(X=feature_matrix)
#     # padding observation (obs) information
#     adata.obs_names = barcodes
#     # padding variable (var) information
#     adata.var['mode'] = 'IMG'
#     adata.var_names = [f'pixel_{i}' for i in range(feature_matrix.shape[1])]
#     # padding obsm information
#     if not spatial_coords is None:
#         adata.obsm['spatial'] = spatial_coords
#     adata.obsm['img_coordinates'] = coords_padded
#     adata.obsm['IMG_Shape'] = np.tile(patch_shape, (adata.shape[0], 1))
#     print("AnnDataobjectconstructcompleted！")
#     adata.layers['rawX'] = adata.X
#     adata.uns['Original_Image'] = img_padded
#     return adata

import numpy as np
import anndata as ad
from tqdm.auto import tqdm
from typing import Tuple

import numpy as np
from typing import Tuple


def _crop_and_flatten_patches_vectorized_corrected(
        full_img: np.ndarray,
        coords: np.ndarray,
        patch_size: int
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
    """
    acorrectionandoptimizationauxiliaryfunction，useNumPyvectoroperationheighteneffectcropgraphpixel block，
    be able toat the same timehandlenumberandnumbersizepatch_size。
    strictaccording tooriginalbeginfunctionformatreturnallrequiredvalue。

    Args:
        full_img (np.ndarray): allsizegroupgraphpixel (H, W, C)。
        coords (np.ndarray): cell/斑pointinheartcoordinates (N, 2)，formatas (y, x)。
        patch_size (int): eachgraphpixel blockpositivesquareshapeedgegrow (canisnumberornumber)。

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], np.ndarray]:
        - flattened_patches (np.ndarray): expandaftergraphpixel blockdata (N, C*H*W)。
        - patches_transposed (np.ndarray): transposeafter4Dgraphpixel blockdata (N, C, H, W)。
        - patch_shape (Tuple[int, int, int]): singlegraphpixel blockshape (C, H, W)。
        - img_padded (np.ndarray): paddingafteroriginalbegingraphpixel。
    """
    # --- mainlymodification point ---
    # remove'patch_size'mustasnumberlimitation。
    # usemoreaccuratepaddinglogiccomeat the same timesupportnumberandnumbersize。
    # fornumbersize (examplesuch as 7), 7 // 2 = 3。weneedininheartpixelelementaroundeach取3pixelelement (3+1+3=7)。
    # fornumbersize (examplesuch as 6), 6 // 2 = 3。we习惯onininheartbefore取3pixelelement，after取2pixelelement (-3,...,+2)。
    # thiskindnofornameby np.indices indexwayselfhandle。

    n_coords = len(coords)
    n_channels = full_img.shape[2]

    # 1. computeaccuratepaddingwidth
    # pad_before definitioninheartpointbeforeneedhow manypixelelement
    pad_before = patch_size // 2
    # pad_after definitioninheartpointafterneedhow manypixelelement
    # fornumber (such as 7), pad_after = 7 - 1 - 3 = 3
    # fornumber (such as 6), pad_after = 6 - 1 - 3 = 2
    pad_after = patch_size - 1 - pad_before

    print(f"currentlyfrom {n_coords} coordinatespointvectorcrop {patch_size}x{patch_size} graphpixel block...")

    # 2. usecomputeexitaccuratewidthperformzeropadding，moresectionmemory
    img_padded = np.pad(
        full_img,
        pad_width=((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
        mode='constant',
        constant_values=0
    )
    # coordinatesalsoshouldbased on 'pad_before'performbiasshift
    coords_padded = coords + pad_before

    # 3. createindexgridtoperformvectorcutslice
    # delta computewaywith 'pad_before' maintainone致
    # np.indices(...,) - pad_before willgenerateafrom -pad_before to +pad_after indexrange
    delta_y, delta_x = np.indices((patch_size, patch_size)) - pad_before
    all_y_indices = coords_padded[:, 0].reshape(-1, 1, 1) + delta_y
    all_x_indices = coords_padded[:, 1].reshape(-1, 1, 1) + delta_x

    # 4. onetimeextractionallgraphpixel block，andtransposeas (N, C, H, W)
    patches_transposed = img_padded[all_y_indices, all_x_indices].transpose(0, 3, 1, 2)

    # 5. from4Darray派生exititsotherneedreturnvalue
    # 派生exitexpandafter2Darray
    flattened_patches = patches_transposed.reshape(n_coords, -1)

    # 派生exitgraphpixel blockshapetuple
    patch_shape = patches_transposed.shape[1:]

    # 6. strictaccording tooriginalbeginsequentialandtypereturnallvalues
    # note：return coords_padded appearinis coords + pad_before，withpaddingaftergraphpixelaccurateforshould
    return (
        flattened_patches.astype(np.float32),
        coords_padded,
        patch_shape,
        img_padded
    )

# =================================================================================
# youoriginalcome create_img_adata_from_data functionappearincannoforconnectthisoptimizationversion
# noneeddoanywhatmodify
# =================================================================================
def create_img_adata_from_data(
        full_img: np.ndarray,
        barcodes: np.ndarray,
        img_coordinates: np.ndarray,
        spatial_coords: np.ndarray = None,
        patch_size: int = 32
) -> ad.AnnData:
    assert barcodes.shape[0] == img_coordinates.shape[0]
    assert img_coordinates.shape[1] == 2
    if not spatial_coords is None:
        assert spatial_coords.shape[0] == barcodes.shape[0]

    print("\n--- step 2/3: cropandexpandgraphpixel block ---")

    # *** appearincallcorrectionafteroptimizationversion ***
    # returnallvalueallcanbycorrectsolvepackage
    feature_matrix, coords_padded, patch_shape, img_padded = _crop_and_flatten_patches_vectorized_corrected(
        full_img=full_img,
        coords=img_coordinates,
        patch_size=patch_size
    )

    print("\n--- step 3/3: constructAnnDataobject ---")
    adata = ad.AnnData(X=feature_matrix)
    adata.obs_names = barcodes
    adata.var['mode'] = 'IMG'
    adata.var_names = [f'pixel_{i}' for i in range(feature_matrix.shape[1])]
    if not spatial_coords is None:
        adata.obsm['spatial'] = spatial_coords
    adata.obsm['img_coordinates'] = coords_padded
    adata.obsm['IMG_Shape'] = np.tile(patch_shape, (adata.shape[0], 1))
    print("AnnDataobjectconstructcompleted！")
    adata.layers['rawX'] = adata.X
    adata.uns['Original_Image'] = img_padded
    return adata

import anndata
import numpy as np
import pandas as pd
from typing import List, Literal, Tuple


def match_and_filter_adata_lists(
        adata_list1: List[anndata.AnnData],
        adata_list2: List[anndata.AnnData],
        match_by: Literal['obs_names', 'spatial'] = 'obs_names',
        spatial_key: str = 'spatial',
        reference_list: Literal[1, 2] = 1
) -> Tuple[List[anndata.AnnData], List[anndata.AnnData]]:
    """
    成formatch、filterandsynchronizationtwoanndataobjectlist。

    thisfunctionnotwillbased onfingermatchstandardfindtototalsame观sequencingvalueandperformfilter，
    alsowillwillnomatchattributeperformsynchronization，ensurereturnmatchforinobs_namesandspacecoordinateson
    allcompletelyallone致。

    parameter:
    ----------
    adata_list1 : List[anndata.AnnData]
        aanndataobjectlist。

    adata_list2 : List[anndata.AnnData]
        twoanndataobjectlist。lengthmustwithadata_list1etc。

    match_by : Literal['obs_names', 'spatial'], defaultas'obs_names'
        matchpattern。

    spatial_key : str, defaultas'spatial'
        whenmatch_by='spatial'time，spacecoordinateskeyname。

    reference_list : Literal[1, 2], defaultas 1
        fingerwhichlistassynchronizationnomatchattributetime“reference”or“standard”。
        - 1: list1isreference，list2towardlist1seeneat。
        - 2: list2isreference，list1towardlist2seeneat。

    return:
    -------
    Tuple[List[anndata.AnnData], List[anndata.AnnData]]
        atuple，containstwonewlist，itsinanndataobjectalreadybyfilterandcompletelyallsynchronization。
    """
    if len(adata_list1) != len(adata_list2):
        raise ValueError("inputerror: twolistlengthmustetc。")
    if match_by not in ['obs_names', 'spatial']:
        raise ValueError(f"inputerror: `match_by` parametermustis 'obs_names' or 'spatial'。")
    if reference_list not in [1, 2]:
        raise ValueError("inputerror: `reference_list` parametermustis 1 or 2。")

    matched_list1 = []
    matched_list2 = []

    for i, (adata1, adata2) in enumerate(zip(adata_list1, adata_list2)):
        print(f"\n--- currentlyhandle {i + 1} for anndata object ---")
        print(f"originalbeginsize: List1 -> {adata1.shape[0]}, List2 -> {adata2.shape[0]}")

        adata1_filtered, adata2_filtered = None, None

        if match_by == 'obs_names':
            common_obs = sorted(list(set(adata1.obs_names) & set(adata2.obs_names)))
            if not common_obs:
                print("warning: 未findtototalsame obs_names。")
                adata1_filtered, adata2_filtered = adata1[[], :].copy(), adata2[[], :].copy()
            else:
                adata1_filtered = adata1[common_obs, :].copy()
                adata2_filtered = adata2[common_obs, :].copy()

                # added：synchronizationspacecoordinates
                print(f"matchpattern: 'obs_names'。currentlywill spatial coordinateswithreferencelist {reference_list} synchronization...")
                if reference_list == 1:
                    adata2_filtered.obsm[spatial_key] = adata1_filtered.obsm[spatial_key]
                else:  # reference_list == 2
                    adata1_filtered.obsm[spatial_key] = adata2_filtered.obsm[spatial_key]

        elif match_by == 'spatial':
            if spatial_key not in adata1.obsm or spatial_key not in adata2.obsm:
                raise KeyError(f"error: `spatial_key='{spatial_key}'` inobjectinnotsavein。")

            coords1_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata1.obsm[spatial_key]]
            coords2_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata2.obsm[spatial_key]]
            df1 = pd.DataFrame({'obs_name_1': adata1.obs_names, 'coord_str': coords1_str})
            df2 = pd.DataFrame({'obs_name_2': adata2.obs_names, 'coord_str': coords2_str})
            merged_df = pd.merge(df1, df2, on='coord_str', how='inner')

            if merged_df.empty:
                print("warning: 未findtototalsamespacecoordinates。")
                adata1_filtered, adata2_filtered = adata1[[], :].copy(), adata2[[], :].copy()
            else:
                obs_to_keep1 = merged_df['obs_name_1'].values
                obs_to_keep2 = merged_df['obs_name_2'].values
                adata1_filtered = adata1[obs_to_keep1, :].copy()
                adata2_filtered = adata2[obs_to_keep2, :].copy()

                # added：synchronization obs_names
                print(f"matchpattern: 'spatial'。currentlywill obs_names withreferencelist {reference_list} synchronization...")
                if reference_list == 1:
                    adata2_filtered.obs_names = adata1_filtered.obs_names
                else:  # reference_list == 2
                    adata1_filtered.obs_names = adata2_filtered.obs_names

        print(f"matchandsynchronizationaftersize: {adata1_filtered.shape[0]}")
        matched_list1.append(adata1_filtered)
        matched_list2.append(adata2_filtered)

    return matched_list1, matched_list2


import pandas as pd
import anndata as ad
import numpy as np
from natsort import natsorted


def merge_and_rename_peaks_in_adata_list(adata_list):
    """
    inonerelatecolumnAnnDataobjectin，mergeheavypeaks，andbased onmergeaftertotalknowledgepeaks
    directlyinoriginalbeginAnnDataobjectonperformheavynameandgathercombine（in-place modification）。

    warning：thisoperationwillclear .varm and .varp indata。

    Args:
        adata_list (list): AnnDataobjectlist。thislistinobjectwillbydirectlymodify。
    """

    # --- step 1, 2, 3: collectset、merge、createmapping（withbeforeversioncompletelyallsame）---
    # (assimpleclean，this里omitthispartcode，itwithonaversionone致)
    all_peaks_df_list = []
    if not adata_list:
        print("inputlistasempty，nomethodhandle。")
        return
    for i, adata in enumerate(adata_list):
        if adata.n_vars == 0:
            continue
        df = pd.DataFrame({'peak_name': adata.var_names})
        df['source_adata_index'] = i
        all_peaks_df_list.append(df)
    if not all_peaks_df_list:
        print("allAnnDataobjectallnofeature（peaks），nomethodhandle。")
        return
    combined_peaks_df = pd.concat(all_peaks_df_list, ignore_index=True)

    def parse_peak_name(peak_name):
        try:
            parts = peak_name.split(':')
            chrom = parts[0]
            start, end = parts[1].split('-')
            return chrom, int(start), int(end)
        except (ValueError, IndexError):
            return None, None, None

    parsed_coords = combined_peaks_df['peak_name'].apply(parse_peak_name)
    combined_peaks_df[['chromosome', 'start', 'end']] = pd.DataFrame(parsed_coords.tolist(),
                                                                     index=combined_peaks_df.index)
    combined_peaks_df.dropna(subset=['chromosome', 'start', 'end'], inplace=True)
    combined_peaks_df['start'] = combined_peaks_df['start'].astype(int)
    combined_peaks_df['end'] = combined_peaks_df['end'].astype(int)
    unique_chroms = combined_peaks_df['chromosome'].unique()
    combined_peaks_df['chromosome'] = pd.Categorical(
        combined_peaks_df['chromosome'], categories=natsorted(unique_chroms), ordered=True
    )
    combined_peaks_df.sort_values(by=['chromosome', 'start'], inplace=True)
    merged_peaks = []
    for chromosome, group in combined_peaks_df.groupby('chromosome', observed=True):
        if group.empty:
            continue
        current_start, current_end = group.iloc[0]['start'], group.iloc[0]['end']
        for i in range(1, len(group)):
            next_peak = group.iloc[i]
            if next_peak['start'] <= current_end:
                current_end = max(current_end, next_peak['end'])
            else:
                merged_peaks.append({'chromosome': str(chromosome), 'start': current_start, 'end': current_end})
                current_start, current_end = next_peak['start'], next_peak['end']
        merged_peaks.append({'chromosome': str(chromosome), 'start': current_start, 'end': current_end})
    consensus_peaks_df = pd.DataFrame(merged_peaks)
    consensus_peaks_df['consensus_peak_name'] = (
            consensus_peaks_df['chromosome'] + ':' + consensus_peaks_df['start'].astype(str) + '-' + consensus_peaks_df[
        'end'].astype(str)
    )
    print(f"originalbeginpeaksnumber: {len(combined_peaks_df)}")
    print(f"mergeaftertotalknowledgepeaksquantity: {len(consensus_peaks_df)}")
    peak_to_consensus_map = {}
    consensus_intervals = {chrom: [] for chrom in unique_chroms}
    for _, row in consensus_peaks_df.iterrows():
        consensus_intervals[row['chromosome']].append((row['start'], row['end'], row['consensus_peak_name']))
    for _, original_peak in combined_peaks_df.iterrows():
        chrom, start, end = original_peak['chromosome'], original_peak['start'], original_peak['end']
        if chrom in consensus_intervals:
            for c_start, c_end, c_name in consensus_intervals[str(chrom)]:
                if start < c_end and end > c_start:
                    peak_to_consensus_map[original_peak['peak_name']] = c_name
                    break

    # --- step 4: ineachoriginalbeginAnnDataobjectonperformheavynameandgathercombine (In-place) ---
    print("\ncurrentlydirectlymodifyoriginalbeginAnnDataobject...")
    for adata in adata_list:
        if adata.n_vars == 0:
            continue

        # 1. preparegathercombineinformation
        new_var_names = [peak_to_consensus_map.get(name, name) for name in adata.var_names]

        # 2. computegathercombineafter .X and .layers matrix
        # createatemporarycolumnused for groupby
        adata.var['consensus_peak_name'] = new_var_names

        # gathercombinedatamatrix .X
        data_df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=adata.obs_names,
            columns=adata.var['consensus_peak_name']
        )
        aggregated_X_df = data_df.groupby(axis=1, level=0).sum()

        # gathercombine .layers indata
        aggregated_layers = {}
        for layer_name, layer_matrix in adata.layers.items():
            layer_df = pd.DataFrame(
                layer_matrix.toarray() if hasattr(layer_matrix, "toarray") else layer_matrix,
                index=adata.obs_names,
                columns=adata.var['consensus_peak_name']
            )
            aggregated_layer_df = layer_df.groupby(axis=1, level=0).sum()
            aggregated_layers[layer_name] = aggregated_layer_df

        # 3. createnew var DataFrame
        new_var_df = pd.DataFrame(index=aggregated_X_df.columns)

        # 4. **core：executeoriginalmodify**

        # first，clearnotcompatibleattribute
        adata.varm.clear()
        adata.varp.clear()

        # getnewvariablenamelist，ensuresequentialcorrect
        final_var_names = new_var_df.index.tolist()

        # use anndata internalmethod `._inplace_subset_var` ismostallway
        # itwillcorrecthandleallcorrelationattributeslice of。
        # wethroughcreatea "false" indexcomeimplementationreplace。
        # note: AnnData 0.8.0 after，directlyvaluemorepush荐。ascompatibleanddirectly，weusedirectlyvalue。

        # directlyvalue (suitableused for AnnData >= 0.8.0)
        adata._var = new_var_df
        adata._X = aggregated_X_df[final_var_names].values

        # update layers
        adata.layers.clear()
        for layer_name, agg_layer_df in aggregated_layers.items():
            adata.layers[layer_name] = agg_layer_df[final_var_names].values

        # clearreasontemporarycolumn
        # becauseaswereplaceneat .var，sothistemporarycolumnalreadynotsavein

    print("\nsuccessinalloriginalbeginAnnDataobjectoncompletedheavynameandgathercombine。")
    print("warning: eachobject .varm and .varp attributealreadybyclear。")
    return adata_list


import anndata
import numpy as np
import pandas as pd
from typing import List, Literal, Tuple


def match_and_filter_adata_lists(
        adata_list1: List[anndata.AnnData],
        adata_list2: List[anndata.AnnData],
        match_by: Literal['obs_names', 'spatial'] = 'obs_names',
        spatial_key: str = 'spatial',
        reference_list: Literal[1, 2] = 1
) -> Tuple[List[anndata.AnnData], List[anndata.AnnData]]:
    """
    成formatch、filterandsynchronizationtwoanndataobjectlist。

    thisfunctionnotwillbased onfingermatchstandardfindtototalsame观sequencingvalueandperformfilter，
    alsowillwillnomatchattributeperformsynchronization，ensurereturnmatchforinobs_namesandspacecoordinateson
    allcompletelyallone致。

    parameter:
    ----------
    adata_list1 : List[anndata.AnnData]
        aanndataobjectlist。

    adata_list2 : List[anndata.AnnData]
        twoanndataobjectlist。lengthmustwithadata_list1etc。

    match_by : Literal['obs_names', 'spatial'], defaultas'obs_names'
        matchpattern。

    spatial_key : str, defaultas'spatial'
        whenmatch_by='spatial'time，spacecoordinateskeyname。

    reference_list : Literal[1, 2], defaultas 1
        fingerwhichlistassynchronizationnomatchattributetime“reference”or“standard”。
        - 1: list1isreference，list2towardlist1seeneat。
        - 2: list2isreference，list1towardlist2seeneat。

    return:
    -------
    Tuple[List[anndata.AnnData], List[anndata.AnnData]]
        atuple，containstwonewlist，itsinanndataobjectalreadybyfilterandcompletelyallsynchronization。
    """
    if len(adata_list1) != len(adata_list2):
        raise ValueError("inputerror: twolistlengthmustetc。")
    if match_by not in ['obs_names', 'spatial']:
        raise ValueError(f"inputerror: `match_by` parametermustis 'obs_names' or 'spatial'。")
    if reference_list not in [1, 2]:
        raise ValueError("inputerror: `reference_list` parametermustis 1 or 2。")

    matched_list1 = []
    matched_list2 = []

    for i, (adata1, adata2) in enumerate(zip(adata_list1, adata_list2)):
        print(f"\n--- currentlyhandle {i + 1} for anndata object ---")
        print(f"originalbeginsize: List1 -> {adata1.shape[0]}, List2 -> {adata2.shape[0]}")

        adata1_filtered, adata2_filtered = None, None

        if match_by == 'obs_names':
            common_obs = sorted(list(set(adata1.obs_names) & set(adata2.obs_names)))
            if not common_obs:
                print("warning: 未findtototalsame obs_names。")
                adata1_filtered, adata2_filtered = adata1[[], :].copy(), adata2[[], :].copy()
            else:
                adata1_filtered = adata1[common_obs, :].copy()
                adata2_filtered = adata2[common_obs, :].copy()

                # added：synchronizationspacecoordinates
                print(f"matchpattern: 'obs_names'。currentlywill spatial coordinateswithreferencelist {reference_list} synchronization...")
                if reference_list == 1:
                    adata2_filtered.obsm[spatial_key] = adata1_filtered.obsm[spatial_key]
                else:  # reference_list == 2
                    adata1_filtered.obsm[spatial_key] = adata2_filtered.obsm[spatial_key]

        elif match_by == 'spatial':
            if spatial_key not in adata1.obsm or spatial_key not in adata2.obsm:
                raise KeyError(f"error: `spatial_key='{spatial_key}'` inobjectinnotsavein。")

            coords1_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata1.obsm[spatial_key]]
            coords2_str = [f"{c[0]:.6f},{c[1]:.6f}" for c in adata2.obsm[spatial_key]]
            df1 = pd.DataFrame({'obs_name_1': adata1.obs_names, 'coord_str': coords1_str})
            df2 = pd.DataFrame({'obs_name_2': adata2.obs_names, 'coord_str': coords2_str})
            merged_df = pd.merge(df1, df2, on='coord_str', how='inner')

            if merged_df.empty:
                print("warning: 未findtototalsamespacecoordinates。")
                adata1_filtered, adata2_filtered = adata1[[], :].copy(), adata2[[], :].copy()
            else:
                obs_to_keep1 = merged_df['obs_name_1'].values
                obs_to_keep2 = merged_df['obs_name_2'].values
                adata1_filtered = adata1[obs_to_keep1, :].copy()
                adata2_filtered = adata2[obs_to_keep2, :].copy()

                # added：synchronization obs_names
                print(f"matchpattern: 'spatial'。currentlywill obs_names withreferencelist {reference_list} synchronization...")
                if reference_list == 1:
                    adata2_filtered.obs_names = adata1_filtered.obs_names
                else:  # reference_list == 2
                    adata1_filtered.obs_names = adata2_filtered.obs_names

        print(f"matchandsynchronizationaftersize: {adata1_filtered.shape[0]}")
        matched_list1.append(adata1_filtered)
        matched_list2.append(adata2_filtered)

    return matched_list1, matched_list2


