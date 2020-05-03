import math

from torch import nn, Tensor

from yukarin_nsf.network.residual_block import ResidualBlock


class WaveNet(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            condition_size: int,
            stack_num: int,
            layer_num_per_stack: int,
    ):
        super().__init__()

        self.input_conv = nn.utils.weight_norm(nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=1,
        ))

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_size=hidden_size,
                condition_size=condition_size,
                dilation=2 ** i_layer,
                is_last=i_stack == stack_num - 1 and i_layer == layer_num_per_stack - 1,
            )
            for i_stack in range(stack_num)
            for i_layer in range(layer_num_per_stack)
        ])

    def forward(
            self,
            x: Tensor,
            c: Tensor,
    ):
        """
        :param x: float (batch_size, length, 1)
        :param c: float (batch_size, length, ?)
        :return:
            output: float (batch_size, length, ?)
        """
        x = x.transpose(1, 2)
        c = c.transpose(1, 2)

        x = self.input_conv(x)

        output = None
        for residual_block in self.residual_blocks:
            x, s = residual_block(x=x, c=c)
            if output is None:
                output = s
            else:
                output += s

        output *= math.sqrt(1.0 / len(self.residual_blocks))

        output = output.transpose(1, 2)
        return output
