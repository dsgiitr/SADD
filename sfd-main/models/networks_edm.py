"""Model architectures and preconditioning schemes used in the paper."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

# @persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

# @persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

        self.kernel = kernel
        self.init_mode = init_mode
        self.init_weight = init_weight
        self.init_bias = init_bias

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

# @persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

# @persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.affine_step = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        # self.affine_step = Linear(in_features=emb_channels, out_features=out_channels, **init)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb, emb_step=None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if emb_step is not None:
            params_step = self.affine_step(emb_step).unsqueeze(2).unsqueeze(3).to(x.dtype)

        if self.adaptive_scale:
            if emb_step is not None:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = torch.addcmul(shift, self.norm1(x), scale + 1)
                scale_step, shift_step = params_step.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift_step, x, scale_step + 1))
            else:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            # if emb_step is not None:
            #     scale_step, shift_step = params_step.chunk(chunks=2, dim=1)
            #     x = silu(torch.addcmul(shift_step, x, scale_step + 1))
        else:
            if emb_step is not None:
                x = silu(self.norm1((x.add_(params)).add_(params_step)))
            else:
                x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

# @persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

# @persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

# @persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        repeat              = 1
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.repeat = repeat
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.map_step = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_step_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_step_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if level == 0:
                    if decoder_type == 'skip':
                        self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                    self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                    self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels*repeat, kernel=3, **init_zero)
                    self.last_layer = f'model.dec.{res}x{res}_aux_conv'
                    # self.last_layer = f'dec.{res}x{res}_aux_conv'
                else:
                    if decoder_type == 'skip' and level < len(channel_mult) - 1:
                        self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                    self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                    self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None, skip_tuning=False, step_condition=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        emb_step = None
        if step_condition is not None:
            emb_step = self.map_step(step_condition)
            emb_step = emb_step.reshape(emb_step.shape[0], 2, -1).flip(1).reshape(*emb_step.shape) # swap sin/cos
            emb_step = silu(self.map_step_layer0(emb_step))
            emb_step = silu(self.map_step_layer1(emb_step))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb, emb_step) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        if skip_tuning:
            count = 0
            coeff_min = 0.75j
            coeff_max = 1
            interval = (coeff_max - coeff_min) / len(skips)
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    if skip_tuning:
                        coeff = coeff_min + interval * count
                        x = torch.cat([x, coeff * skips.pop()], dim=1)
                        count += 1
                    else:
                        x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb, emb_step)

        # CHANNEL_AXIS = 1
        # if self.repeat > 1:
        #     aux = torch.split(aux, aux.shape[CHANNEL_AXIS], CHANNEL_AXIS)
        
        return aux
    
    # def modify_dict(self, old_dict):
    #     """
    #     Return a new state‐dict where every key starting with "model."
    #     has that prefix removed.
    #     """
    #     new_dict = {}
    #     for key, value in old_dict.items():
    #         if key.startswith("model."):
    #             new_key = key[len("model."):]
    #         else:
    #             new_key = key
    #         new_dict[new_key] = value
    #     return new_dict




import hashlib

def _tensor_hash(tensor):
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()
#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

# @persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
        repeat              = 1,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        self.map_step = PositionalEmbedding(num_channels=model_channels)
        self.map_step_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_step_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels*repeat, kernel=3, **init_zero)
        self.last_layer = 'model.out_conv'

    def forward(self, x, noise_labels, class_labels, augment_labels=None, skip_tuning=False, step_condition=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        emb_step = None
        if step_condition is not None:
            emb_step = self.map_step(step_condition)
            emb_step = silu(self.map_step_layer0(emb_step))
            emb_step = self.map_step_layer1(emb_step)
            # if self.map_label is not None:
                # emb_step = emb_step + tmp
            emb_step = silu(emb_step)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb, emb_step) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        if skip_tuning:
            count = 0
            coeff_min = 0.75
            coeff_max = 1
            interval = (coeff_max - coeff_min) / len(skips)
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                if skip_tuning:
                    coeff = coeff_min + interval * count
                    x = torch.cat([x, coeff * skips.pop()], dim=1)
                    count += 1
                else:
                    x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb, emb_step)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

# @persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0.002,            # Minimum supported noise level.
        sigma_max       = 80.0,             # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        repeat = 1,                         # Number of branches
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.repeat = repeat
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, repeat=repeat, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, step_condition=None, **model_kwargs):
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = x.to(dtype)
        sigma = sigma.to(dtype).reshape(-1, 1, 1, 1)
        if step_condition is not None:
            step_condition = torch.tensor(step_condition).to(dtype).reshape(-1,).to(x.device)
            
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(dtype).reshape(-1, self.label_dim)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, step_condition=step_condition, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x.repeat(1, self.model.repeat, 1, 1) + c_out * F_x
        # print("teacher",D_x.shape)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def load_state_dict(self, state_dict, *args, **kwargs):

        # print("only in original:")
        # print(set(state_dict.keys())-set(self.state_dict().keys()))

        # print("only in loading file:")
        # print(set(self.state_dict().keys())-set(state_dict.keys()))

        # print("in both:")
        # print(set(self.state_dict().keys()).intersection(set(state_dict.keys())))

        if self.repeat == 1:
            return super().load_state_dict(state_dict, *args, **kwargs)
        
        modified_state_dict = {}
        print(self.model.last_layer,"LASTLASYER")
        for key, value in state_dict.items():
            if key.startswith(self.model.last_layer):
                print("calling tilingg")
                modified_state_dict[key] = self._tile_weight_for_repeat(key, value)
            else:
                modified_state_dict[key] = value
        
        return super().load_state_dict(modified_state_dict, *args, **kwargs)
    
    def _tile_weight_for_repeat(self, key, tensor):
        if 'bias' in key:
            return tensor.repeat(self.repeat)
        elif 'weight' in key:
            return tensor.clone().repeat(self.repeat, 1, 1, 1)
        else:
            print("panik")
        return tensor

#----------------------------------------------------------------------------

# # @persistence.persistent_class
class CFGPrecond(torch.nn.Module):
    def __init__(self,
        model,
        guidance_type   = 'classifier-free',
        guidance_rate   = 1.0,
        epsilon_t       = 1e-3,                 # Minimum t-value used during training.
        beta_d          = 9.0420,               # Extent of the noise level schedule.
        beta_min        = 0.8477,               # Initial slope of the noise level schedule.
        img_resolution  = 64,
        img_channels    = 4,
        label_dim       = True,                 # Number of class labels, 0 = unconditional.
        model_type      = 'CFGUNet',            # Class name of the underlying model.
        use_fp16        = False,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.model = model
        self.guidance_rate = guidance_rate
        self.guidance_type = guidance_type
        self.alphas_cumprod = model.alphas_cumprod
        self.get_learned_conditioning = model.get_learned_conditioning
        self.apply_model = model.apply_model
        
        alphas_cumprod = model.alphas_cumprod
        log_alphas = 0.5 * torch.log(alphas_cumprod)
        self.M = len(log_alphas)
        self.t_array = torch.linspace(0., 1., self.M + 1)[1:].reshape((1, -1))
        self.log_alpha_array = log_alphas.reshape((1, -1,))

        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.use_fp16 = use_fp16
        
    def noise_pred_fn(self, x, c_noise, cond=None, step_condition=None):
        if c_noise.reshape((-1,)).shape[0] == 1:
            c_noise = c_noise.expand((x.shape[0]))
        if step_condition is not None and step_condition.reshape((-1,)).shape[0] == 1:
            step_condition = step_condition.expand((x.shape[0]))
        t_input = c_noise
        return self.wrapper_fn(x, t_input, cond, step_condition=step_condition)
        
    def forward(self, x, sigma, condition=None, unconditional_condition=None, force_fp32=False, step_condition=None, **model_kwargs):
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = x.to(dtype)
        sigma = sigma.to(dtype).reshape(-1,)
        if step_condition is not None:
            step_condition = torch.tensor(step_condition).to(dtype).reshape(-1,).to(x.device)

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M * self.sigma_inv(sigma) - 1.

        if c_noise.reshape((-1,)).shape[0] == 1:
            c_noise = c_noise.expand((x.shape[0]))
        if self.guidance_type == "uncond":
            F_x = self.noise_pred_fn(c_in.reshape(-1,1,1,1) * x, c_noise, step_condition=step_condition)
        elif self.guidance_type == "classifier-free":
            if self.guidance_rate == 1. or unconditional_condition is None:
                F_x = self.noise_pred_fn(c_in * x, c_noise, cond=condition, step_condition=step_condition)
            else:
                x_in = torch.cat([c_in.reshape(-1,1,1,1) * x] * 2)
                t_in = torch.cat([c_noise] * 2)
                cond_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = self.noise_pred_fn(x_in, t_in, cond=cond_in, step_condition=step_condition).chunk(2)
                F_x = noise_uncond + self.guidance_rate * (noise - noise_uncond)

        D_x = c_skip * x + c_out.reshape(-1,1,1,1) * F_x

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def marginal_log_mean_coeff(self, t):
        t = torch.tensor(t)
        return self.interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def sigma_inv(self, sigma):
        lamb = -(sigma.log())
        log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
        t = self.interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
        return t.reshape((-1,))
    
    def interpolate_fn(self, x, xp, yp):
        """
        A piecewise linear function y = f(x), using xp and yp as keypoints.
        We implement f(x) in a differentiable way (i.e. applicable for autograd).
        The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

        Args:
            x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
            xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
            yp: PyTorch tensor with shape [C, K].
        Returns:
            The function values f(x), with shape [N, C].
        """
        N, K = x.shape[0], xp.shape[1]
        all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
        sorted_all_x, x_indices = torch.sort(all_x, dim=2)
        x_idx = torch.argmin(x_indices, dim=2)
        cand_start_idx = x_idx - 1
        start_idx = torch.where(
            torch.eq(x_idx, 0),
            torch.tensor(1, device=x.device),
            torch.where(
                torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
            ),
        )
        end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
        start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
        end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
        start_idx2 = torch.where(
            torch.eq(x_idx, 0),
            torch.tensor(0, device=x.device),
            torch.where(
                torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
            ),
        )
        y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
        start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
        end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
        cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
        return cand

    def wrapper_fn(self, x, t, cond=None, step_condition=None):
        return self.model.apply_model(x, t, cond, step_condition=step_condition)
