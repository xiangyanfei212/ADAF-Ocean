import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from icecream import ic
from models import register
from functools import reduce
from torch import Tensor
from thop import profile


class Encoder(nn.Module):

    """The Encoder: Encodes multiple input sources into fixed-size representations."""

    def __init__(
            self,
            num_bg_points: int,
            num_in_situ_obs_points: int,
            num_sate_sss_points: int,
            num_sate_sst_points: int,
            num_sate_ssw_points: int,
            num_sate_sic_points: int,
            num_sate_sla_points: int,
            depth: int,
            skip_layers: list,
            num_latent_size: int,
            in_ch_bg: int,
            in_ch_in_situ_obs: int,
            in_ch_sate_sss: int,
            in_ch_sate_ssw: int,
            in_ch_sate_sst: int,
            in_ch_sate_sla: int,
            in_ch_sate_sic: int,
            hidden_dim: int,
            activation: str = "relu",
        ):
        """
        Initializes the encoder for multiple inputs.
        Each input (background, satellite, radar, surface, balloon) is processed 
        by a separate multi-layer perceptron (MLP).

        Args:
        """
        super(Encoder, self).__init__()

        self.skip_layers = skip_layers

        self.activation = self._get_activation(activation)

        # %% Background
        self.bg_encoders = nn.ModuleList(
            [nn.Linear(in_ch_bg, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_bg if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        # ic(self.bg_encoders)
        self.bg_post_proc = nn.Linear(num_bg_points, num_latent_size)
        
        # %% in-situ observations
        self.in_situ_obs_encoders = nn.ModuleList(
            [nn.Linear(in_ch_in_situ_obs, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_in_situ_obs if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        # ic(self.in_situ_obs_encoders)
        self.in_situ_obs_post_proc = nn.Linear(num_in_situ_obs_points, num_latent_size)

        # %% Satellite SSS
        self.sate_sss_encoders = nn.ModuleList(
            [nn.Linear(in_ch_sate_sss, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_sate_sss if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        self.sate_sss_post_proc = nn.Linear(num_sate_sss_points, num_latent_size)

        # %% Satellite SST
        self.sate_sst_encoders = nn.ModuleList(
            [nn.Linear(in_ch_sate_sst, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_sate_sst if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        self.sate_sst_post_proc = nn.Linear(num_sate_sst_points, num_latent_size)

        self.sate_sla_encoders = nn.ModuleList(
            [nn.Linear(in_ch_sate_sla, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_sate_sla if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        self.sate_sla_post_proc = nn.Linear(num_sate_sla_points, num_latent_size)

        self.sate_ssw_encoders = nn.ModuleList(
            [nn.Linear(in_ch_sate_ssw, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_sate_ssw if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        self.sate_ssw_post_proc = nn.Linear(num_sate_ssw_points, num_latent_size)

        self.sate_sic_encoders = nn.ModuleList(
            [nn.Linear(in_ch_sate_sic, hidden_dim)] + 
            [nn.Linear(hidden_dim + in_ch_sate_sic if i in skip_layers else hidden_dim, hidden_dim) for i in range(1, depth)]
        )
        self.sate_sic_post_proc = nn.Linear(num_sate_sic_points, num_latent_size)

    def _get_activation(self, activation: str):
        """Returns the activation function based on the given name."""
        if activation.lower() == "relu":
            return F.relu
        elif activation.lower() == "leakyrelu":
            return F.leaky_relu
        elif activation.lower() == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward_mlp(self, encoders, post_proc, input_x, input_y, num_points):
        """General function to process inputs through MLP."""
        input_data = torch.cat([input_x, input_y], dim=-1)
        batch_size, _, _ = input_data.shape
        input_data = input_data.view(batch_size * num_points, -1)
        hidden = input_data
        for i, layer in enumerate(encoders[:-1]):
            if i in self.skip_layers:
                hidden = torch.cat([hidden, input_data], dim=-1)
            hidden = self.activation(layer(hidden))
        hidden = encoders[-1](hidden)
        hidden = hidden.view(batch_size, num_points, -1)
        hidden = post_proc(hidden.transpose(1, 2))
        return hidden.transpose(1, 2)

    def forward(
            self, 
            bg_context_x,
            bg_context_y,
            num_bg_context_points,
            in_situ_obs_context_x,
            in_situ_obs_context_y,
            num_in_situ_obs_context_points,
            sate_sss_context_x,
            sate_sss_context_y,
            num_sate_sss_context_points,
            sate_sst_context_x,
            sate_sst_context_y,
            num_sate_sst_context_points,
            sate_ssw_context_x,
            sate_ssw_context_y,
            num_sate_ssw_context_points,
            sate_sla_context_x,
            sate_sla_context_y,
            num_sate_sla_context_points,
            sate_sic_context_x,
            sate_sic_context_y,
            num_sate_sic_context_points, 
        ):

        """
        Forward pass through the encoder. Processes each input source independently,
        encodes them into fixed-size representations, and concatenates the results.

        Args:
            

        Returns:
            representation: A concatenated tensor of shape (batch_size, total_representation_dim),
            representing all input sources.
        """


        """Forward pass through the encoder."""
        bg_hidden = self.forward_mlp(
            self.bg_encoders,
            self.bg_post_proc,
            bg_context_x,
            bg_context_y,
            num_bg_context_points)
        in_situ_hidden = self.forward_mlp(
            self.in_situ_obs_encoders,
            self.in_situ_obs_post_proc,
            in_situ_obs_context_x,
            in_situ_obs_context_y,
            num_in_situ_obs_context_points)
        sate_sss_hidden = self.forward_mlp(
            self.sate_sss_encoders,
            self.sate_sss_post_proc,
            sate_sss_context_x,
            sate_sss_context_y,
            num_sate_sss_context_points)
        sate_sst_hidden = self.forward_mlp(
            self.sate_sst_encoders,
            self.sate_sst_post_proc,
            sate_sst_context_x,
            sate_sst_context_y,
            num_sate_sst_context_points)
        sate_ssw_hidden = self.forward_mlp(
            self.sate_ssw_encoders,
            self.sate_ssw_post_proc,
            sate_ssw_context_x,
            sate_ssw_context_y,
            num_sate_ssw_context_points)
        sate_sla_hidden = self.forward_mlp(
            self.sate_sla_encoders,
            self.sate_sla_post_proc,
            sate_sla_context_x,
            sate_sla_context_y,
            num_sate_sla_context_points)
        sate_sic_hidden = self.forward_mlp(
            self.sate_sic_encoders,
            self.sate_sic_post_proc,
            sate_sic_context_x,
            sate_sic_context_y,
            num_sate_sic_context_points)

        representation = torch.cat([
            bg_hidden,
            in_situ_hidden,
            sate_sss_hidden,
            sate_sst_hidden,
            sate_ssw_hidden,
            sate_sla_hidden,
            sate_sic_hidden],
            dim=-1
        )
        ic(representation.shape)

        return representation


class Decoder(nn.Module):
    """The Decoder: Decodes the encoded representation into predictions."""

    def __init__(self, input_ch, output_ch, W, D, skip_layers, activation="relu"):
        """
        Initializes the decoder, which is an MLP used to map the encoded representation
        and target inputs into the predicted output distribution.

        Args:
            output_sizes: A list of layer sizes for the decoder MLP. The last layer
                          should output twice the target_y dimension (mean and log-variance).
        """ 
        super(Decoder, self).__init__()

        self.skip_layers = skip_layers

        self.activation = self._get_activation(activation)

        # Define the linear layers for point coordinates
        self.layers = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W + input_ch if i in skip_layers else W, W) for i in range(1, D)]
        )
        self.output_layers = nn.Linear(W, output_ch)  # Direct output layer
        # ic(self.layers)
        # ic(self.output_layers)

    def _get_activation(self, activation: str):
        """
        Returns the activation function based on the given name.
        """
        if activation.lower() == "relu":
            return F.relu
        elif activation.lower() == "leakyrelu":
            return F.leaky_relu
        elif activation.lower() == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, representation, target_x, num_total_points):
        """
        Performs the forward pass of the decoder. 
        Combines the encoded representation with the target inputs to predict the mean and variance of the target outputs.

        Args:
            representation: Encoded representation from the encoder, 
                            shape (batch_size, representation_dim).
            target_x: Target inputs, shape (batch_size, num_target_points, x_dim_target).
            num_total_points: Number of target points (int).

        Returns:
            dist: A MultivariateNormal distribution over the target points.
            mu: The predicted mean, shape (batch_size, num_target_points, y_dim_target).
            sigma: The predicted standard deviation, shape (batch_size, num_target_points, y_dim_target).
        """

        # Expand representation to match the number of target points
        # representation = representation.unsqueeze(1).repeat(1, num_total_points, 1)

        repeat = target_x.shape[1] // representation.shape[1] 
        # ic(repeat)
        representation = torch.repeat_interleave(representation, repeat, axis=1)

        # print(f'representaion: {representation.shape}')
        # print(f'target_x: {target_x.shape}')
        decoder_input = torch.cat([representation, target_x], dim=-1)
        # ic(decoder_input.shape)

        # Flatten the input for parallel processing
        batch_size, _, _ = decoder_input.shape
        hidden = decoder_input.view(batch_size * num_total_points, -1)
        # ic(hidden.shape)

        h = hidden
        # print(f'skips: {self.skip_layers}')
        for i, layer in enumerate(self.layers):
            # print(f'layer {i}: {h.shape}')

            if i in self.skip_layers:
                h = torch.cat([hidden, h], dim=-1)
                # print(f'skip: {h.shape}')
            
            h = self.activation(layer(h))

        hidden = self.output_layers(h)

        # Reshape back to original shape
        predictions = hidden.view(batch_size, num_total_points, -1)

        return predictions


@register('CNP_determ_skip_connection_v4')
class CNP(nn.Module):
    """The CNP Model: Combines the encoder and decoder to process inputs and predict target outputs."""

    def __init__(
        self,
        in_ch_bg: int,
        in_ch_in_situ_obs: int,
        in_ch_sate_sss: int,
        in_ch_sate_ssw: int,
        in_ch_sate_sst: int,
        in_ch_sate_sla: int,
        in_ch_sate_sic: int,
        num_bg_points: int,
        num_in_situ_obs_points: int,
        num_sate_sss_points: int,
        num_sate_sst_points: int,
        num_sate_ssw_points: int,
        num_sate_sic_points: int,
        num_sate_sla_points: int,
        encoder_depth: int,
        encoder_hidden_dim: int,
        encoder_skip_layers: list,
        encoder_num_latent_size: int,
        decoder_input_dim: int,
        decoder_output_dim: int,
        decoder_hidden_dim: int,
        decoder_depth: int,
        decoder_skip_layers: list,
        activation: str,
    ):
        """
        Initializes the Conditional Neural Process (CNP) model.

        Args:
            encoder_bg_output_sizes: [8, 16, 32] # len(bg_vars) + 3
            encoder_in_situ_obs_output_sizes: [5, 16, 32] # len(in_situ_obs_vars) + 3
            encoder_sate_sss_output_sizes: [4, 16, 32] # len(sate_vars) + 3
            encoder_sate_sst_output_sizes: [5, 16, 32] # len(sate_vars) + 3
            encoder_sate_ssw_output_sizes: [7, 16, 32] # len(sate_vars) + 3
            encoder_sate_sic_output_sizes: [4, 16, 32] # len(sate_vars) + 3
            encoder_sate_sla_output_sizes: [4, 16, 32] # len(sate_vars) + 3
        """

        super(CNP, self).__init__()
        self.encoder = Encoder(
            num_bg_points=num_bg_points,
            num_in_situ_obs_points=num_in_situ_obs_points,
            num_sate_sss_points=num_sate_sss_points,
            num_sate_sst_points=num_sate_sst_points,
            num_sate_ssw_points=num_sate_ssw_points,
            num_sate_sic_points=num_sate_sic_points,
            num_sate_sla_points=num_sate_sla_points,
            depth=encoder_depth,
            skip_layers=encoder_skip_layers,
            hidden_dim=encoder_hidden_dim,
            num_latent_size=encoder_num_latent_size,
            in_ch_bg=in_ch_bg,
            in_ch_in_situ_obs=in_ch_in_situ_obs,
            in_ch_sate_sss=in_ch_sate_sss,
            in_ch_sate_ssw=in_ch_sate_ssw,
            in_ch_sate_sst=in_ch_sate_sst,
            in_ch_sate_sla=in_ch_sate_sla,
            in_ch_sate_sic=in_ch_sate_sic,
            activation=activation,
        )
        self.decoder = Decoder(
            decoder_input_dim,
            decoder_output_dim,
            decoder_hidden_dim,
            decoder_depth,
            decoder_skip_layers,
            activation=activation,
        )
        ic(self.decoder)

    def forward(self, context_points, num_context_points, num_target_points, target_x):
        """
        Forward pass of the CNP model. Encodes the input context points into a global
        representation, then decodes it with the target inputs to predict the target outputs.

        Args:
            target_x: target's coordinates

        Returns:
            prediction: values on the target's coordinates
        """

        (
            (bg_context_x, bg_context_y),
            (in_situ_obs_context_x, in_situ_obs_context_y),
            (sate_sss_context_x, sate_sss_context_y),
            (sate_sst_context_x, sate_sst_context_y),
            (sate_ssw_context_x, sate_ssw_context_y),
            (sate_sla_context_x, sate_sla_context_y),
            (sate_sic_context_x, sate_sic_context_y)
        ) = context_points
        
        (
            num_bg_points,
            num_in_situ_obs_context_points,
            num_sate_sss_context_points,
            num_sate_sst_context_points,
            num_sate_ssw_context_points,
            num_sate_sla_context_points,
            num_sate_sic_context_points,
        ) = num_context_points

        # Step 1: Encode the context points into a global representation
        # Each input source is processed through its respective MLP, and their
        # representations are concatenated into a single vector.
        representation = self.encoder(
            bg_context_x,
            bg_context_y,
            num_bg_points,
            in_situ_obs_context_x,
            in_situ_obs_context_y,
            num_in_situ_obs_context_points,
            sate_sss_context_x,
            sate_sss_context_y,
            num_sate_sss_context_points,
            sate_sst_context_x,
            sate_sst_context_y,
            num_sate_sst_context_points,
            sate_ssw_context_x,
            sate_ssw_context_y,
            num_sate_ssw_context_points,
            sate_sla_context_x,
            sate_sla_context_y,
            num_sate_sla_context_points,
            sate_sic_context_x,
            sate_sic_context_y,
            num_sate_sic_context_points,
        )
        # Step 2: Decode the global representation and target inputs
        predictions = self.decoder(representation, target_x, num_target_points)

        # Return the log-probability, predicted mean, and standard deviation
        return predictions
    

def multiply_list_elements(lst):
    """
    Multiplies all elements in a list together.

    Args:
        lst (list): A list of numbers, e.g. [150, 150, 4].

    Returns:
        int: The result of multiplying all the elements in the list.
    """
    return reduce(lambda x, y: x * y, lst)


def count_parameters(model):
    count_params = 0
    for p in model.parameters():
        if p.requires_grad:
            count_params += p.numel()
    return count_params


if __name__ == '__main__':


    target_block_shape=[10, 20, 1] # units=grid/degree
    bg_block_shape =[10, 20, 1] # units:degree
    sate_sic_patch_shape =[10, 20] # units=degree/grid
    sate_sla_patch_shape =[10, 20] # units=grid
    sate_ssw_patch_shape =[10, 20] # units=grid, 10/0.25=40, 20/0.25=80, [40, 80]
    sate_sst_patch_shape =[10, 20]   # units=grid, 10/0.04=250, 20/0.04=500, [250, 500], 
    sate_sss_patch_shape =[10, 20]    # units=grid, 10/0.2=50, 20/0.02=100 
    in_situ_obs_patch_range =[10, 20, 20] # units=degree, degree, meter
    in_situ_obs_pad =300

    in_ch_bg=10 # 8 10 
    in_ch_in_situ_obs=7 # 5 7 
    in_ch_sate_sss=6 # 4 6
    in_ch_sate_ssw=7 # 5 7
    in_ch_sate_sst=7 # 5 7
    in_ch_sate_sla=6 # 4 6
    in_ch_sate_sic=6 # 4 6
    encoder_depth=6 
    encoder_skip_layers=[2, 4]
    encoder_hidden_dim=48 # 32
    encoder_num_latent_size=200  # can be divided by num_target_points
    decoder_input_dim=341 # encoder_hidden_dim*7+coords_dim
    decoder_output_dim=5
    decoder_hidden_dim=128
    decoder_depth=8
    decoder_skip_layers=[2, 4, 6]
    activation='relu' # relu, leakyrelu, tanh

    num_target_points = multiply_list_elements(target_block_shape)
    num_bg_points = multiply_list_elements(bg_block_shape)
    num_in_situ_obs_points = in_situ_obs_pad
    num_sate_sss_points = multiply_list_elements(sate_sss_patch_shape)
    num_sate_sst_points = multiply_list_elements(sate_sst_patch_shape)
    num_sate_ssw_points = multiply_list_elements(sate_ssw_patch_shape)
    num_sate_sic_points = multiply_list_elements(sate_sic_patch_shape)
    num_sate_sla_points = multiply_list_elements(sate_sla_patch_shape)

    

    model = CNP(
        in_ch_bg=in_ch_bg,
        in_ch_in_situ_obs=in_ch_in_situ_obs,
        in_ch_sate_sss=in_ch_sate_sss,
        in_ch_sate_ssw=in_ch_sate_ssw,
        in_ch_sate_sst=in_ch_sate_sst,
        in_ch_sate_sla=in_ch_sate_sla,
        in_ch_sate_sic=in_ch_sate_sic,
        num_bg_points=num_bg_points,
        num_in_situ_obs_points=num_in_situ_obs_points,
        num_sate_sss_points=num_sate_sss_points,
        num_sate_sst_points=num_sate_sst_points,
        num_sate_ssw_points=num_sate_ssw_points,
        num_sate_sic_points=num_sate_sic_points,
        num_sate_sla_points=num_sate_sla_points,
        encoder_depth=encoder_depth,
        encoder_skip_layers=encoder_skip_layers,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_latent_size=encoder_num_latent_size, # can be divided by num_target_points
        decoder_input_dim=decoder_input_dim,
        decoder_output_dim=decoder_output_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_depth=decoder_depth,
        decoder_skip_layers=decoder_skip_layers,
        activation="leakyrelu" # relu, leakyrelu, tanh
    )

    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)

    # %% Data
    batch_size = 2
    bg_context_x = torch.randn(batch_size, num_bg_points, 5).to(device)
    bg_context_y = torch.randn(batch_size, num_bg_points, 5).to(device)

    in_situ_obs_context_x = torch.randn(batch_size, num_in_situ_obs_points, 2).to(device)
    in_situ_obs_context_y = torch.randn(batch_size, num_in_situ_obs_points, 5).to(device)

    sate_sss_context_x = torch.randn(batch_size, num_sate_sss_points, 1).to(device)
    sate_sss_context_y = torch.randn(batch_size, num_sate_sss_points, 5).to(device)

    sate_sst_context_x = torch.randn(batch_size, num_sate_sst_points, 2).to(device)
    sate_sst_context_y = torch.randn(batch_size, num_sate_sst_points, 5).to(device)

    sate_ssw_context_x = torch.randn(batch_size, num_sate_ssw_points, 2).to(device)
    sate_ssw_context_y = torch.randn(batch_size, num_sate_ssw_points, 5).to(device)

    sate_sic_context_x = torch.randn(batch_size, num_sate_sic_points, 1).to(device)
    sate_sic_context_y = torch.randn(batch_size, num_sate_sic_points, 5).to(device)

    sate_sla_context_x = torch.randn(batch_size, num_sate_sla_points, 1).to(device)
    sate_sla_context_y = torch.randn(batch_size, num_sate_sla_points, 5).to(device)

    target_x = torch.randn(batch_size, num_target_points, 5).to(device)

    context_points = (
        (bg_context_x, bg_context_y),
        (in_situ_obs_context_x, in_situ_obs_context_y),
        (sate_sss_context_x, sate_sss_context_y),
        (sate_sst_context_x, sate_sst_context_y),
        (sate_ssw_context_x, sate_ssw_context_y),
        (sate_sla_context_x, sate_sla_context_y),
        (sate_sic_context_x, sate_sic_context_y),
    )

    num_context_points = (
        num_bg_points,
        num_in_situ_obs_points,
        num_sate_sss_points,
        num_sate_sst_points,
        num_sate_ssw_points,
        num_sate_sla_points,
        num_sate_sic_points,
    )

    target_points = (
        target_x,
        # target_y,
    )

    pre = model(
        context_points,
        num_context_points,
        num_target_points,
        target_x,
        # target_y
    )
    print("pre:", pre.shape)

    macs, params = profile(model, inputs=(context_points, num_context_points, num_target_points, target_x)) # target_y))
    print('macs: ', macs, 'params: ', params)
    print('macs: %.2f G, params: %.2f M' % (macs / 1000000000.0, params / 1000000.0))

    num_params = count_parameters(model)
    print(f'num_params: {num_params}')


    
