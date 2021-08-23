import pytorch_lightning as pl
import torch.nn as nn


class CNN(pl.LightningModule):
    '''
    This class will create the structure of the CNNs, it can take a wide range of optional parameters.
    The only mandatory parameter is the input_size and output_size.
    Every parameter can be defined as a singular value, this will then be used at every step.
    It can also be defined as a list/tuple and then it can have separate values for each layer/step.
    Note that the length should match the amount of convolutional and linear layers if it is passed as list or tuple.
    '''

    def __init__(self, input_size, output_size, activation=nn.ReLU,
                 size_2d=[], kernel_2d=4, stride_2d=1, pad_2d=0, drop_2d=0, groups_2d=1, dilation_2d=1,
                 pool_2d=False, pool_2d_size=0, pool_2d_stride=2, pool_2d_pad=0, batchnorm_2d=False,
                 size_1d=[], drop_1d=0, batchnorm_1d=False, final_activation=None):
        super().__init__()

        self.output_size, self.activation, \
        self.size_2d, self.kernel_2d, self.stride_2d, self.pad_2d, self.drop_2d, self.groups_2d, self.dilation_2d, \
        self.pool_2d, self.pool_2d_size, self.pool_2d_stride, self.pool_2d_pad, self.batchnorm_2d, \
        self.size_1d, self.drop_1d, self.batchnorm_1d, self.final_activation \
            = output_size, activation, \
              size_2d, kernel_2d, stride_2d, pad_2d, drop_2d, groups_2d, dilation_2d, \
              pool_2d, pool_2d_size, pool_2d_stride, pool_2d_pad, batchnorm_2d, \
              size_1d, drop_1d, batchnorm_1d, final_activation

        if len(input_size) == 1:
            self.height, self.width, self.channels = 1, input_size[0], 1
        elif len(input_size) == 2:
            self.height, self.width, self.channels = input_size[0], input_size[1], 1
        elif len(input_size) == 3:
            self.height, self.width, self.channels = input_size[1], input_size[2], input_size[0]

        # This will check whether the length of the input parameters corresponds to the amount of convolutional and/or linear layers
        self.validate_input_parameters()
        self.build_2d_layers()  # Build the 2d convolutional layers
        self.build_1d_layers()  # Construct the linear layers

    def forward(self, x):
        """
        Defines a forward pass of the CNN
        """
        x = self.layers_2d(x)
        x = x.view(-1, self.conv_output_size)  # Flatten the output of the convolutional layers
        x = self.layers_1d(x)
        return x.view(-1, self.output_size).float()

    def validate_input_parameters(self):
        """
        Validates the length and type of the input parameters
        """

        if type(self.size_2d) == int:
            self.size_2d = [self.size_2d]
        for size in self.size_2d:
            assert size > 0, 'Invalid size_2d, must be strictly positive'
        length_2d = len(self.size_2d)

        if type(self.kernel_2d) == int:
            self.kernel_2d = [self.kernel_2d] * length_2d
        for kernel_2d in self.kernel_2d:
            assert kernel_2d > 0, 'Invalid kernel_2d, must be strictly positive'

        if type(self.stride_2d) == int:
            self.stride_2d = [self.stride_2d] * length_2d
        for stride_2d in self.stride_2d:
            assert stride_2d > 0, 'Invalid stride_2d, must be strictly positive'

        if type(self.pad_2d) == int:
            self.pad_2d = [self.pad_2d] * length_2d
        for pad_2d in self.pad_2d:
            assert pad_2d >= 0, 'Invalid pad_2d, may not be negative'

        if type(self.dilation_2d) == int:
            self.dilation_2d = [self.dilation_2d] * length_2d
        for dilation_2d in self.dilation_2d:
            assert dilation_2d > 0, 'Invalid dilation_2d, must be strictly positive'

        if type(self.groups_2d) == int:
            self.groups_2d = [self.groups_2d] * length_2d
        for groups_2d in self.groups_2d:
            assert groups_2d > 0, 'Invalid groups_2d, must be strictly positive'

        if callable(self.pool_2d):
            self.pool_2d = [self.pool_2d] * length_2d
        for pool_2d in self.pool_2d:
            assert callable(
                pool_2d) or pool_2d == None, 'Invalid pool_2d, should be a 2D non-adaptive torch.nn pool callable or None'

        if type(self.pool_2d_size) == int:
            self.pool_2d_size = [self.pool_2d_size] * length_2d
        for pool_2d_size in self.pool_2d_size:
            assert pool_2d_size > 0, 'Invalid pool_2d_size, must be strictly positive'

        if type(self.pool_2d_stride) == int:
            self.pool_2d_stride = [self.pool_2d_stride] * length_2d
        for pool_2d_stride in self.pool_2d_stride:
            assert pool_2d_stride > 0, 'Invalid pool_2d_stride, must be strictly positive'

        if type(self.pool_2d_pad) == int:
            self.pool_2d_pad = [self.pool_2d_pad] * length_2d
        for pool_2d_pad in self.pool_2d_pad:
            assert pool_2d_pad >= 0, 'Invalid pool_2d_pad, may not be negative'

        if type(self.batchnorm_2d) == bool:
            self.batchnorm_2d = [self.batchnorm_2d] * length_2d

        if type(self.drop_2d) in (float, int):
            self.drop_2d = [self.drop_2d] * length_2d
        for drop_2d in self.drop_2d:
            assert drop_2d >= 0 and drop_2d < 1, 'Invalid drop_2d, must be between 0 and 1'

        for param in (
                self.kernel_2d, self.stride_2d, self.pad_2d, self.pool_2d, self.pool_2d_size, self.pool_2d_stride,
                self.pool_2d_pad,
                self.batchnorm_2d, self.drop_2d):
            assert len(
                param) == length_2d, f'Wrong size of 2d convolutional parameter, expected {length_2d}, got {len(param)}'

        #######################################################################################

        if type(self.size_1d) == int:
            self.size_1d = [self.size_1d]
        for size in self.size_1d:
            assert size > 0, 'Invalid size_1d, must be strictly positive'
        length_1d = len(self.size_1d)

        if type(self.batchnorm_1d) == bool:
            self.batchnorm_1d = [self.batchnorm_1d] * length_1d

        if type(self.drop_1d) in (float, int):
            self.drop_1d = [self.drop_1d] * length_1d
        for drop_1d in self.drop_1d:
            assert drop_1d >= 0 and drop_1d < 1, 'Invalid drop_1d, must be between 0 and 1'
        assert len(self.drop_1d) == length_1d, 'Wrong length of drop_1d parameter'

        ######################################################################################

        assert callable(self.activation), 'activation function not callable'
        if self.final_activation:
            assert callable(self.final_activation), 'final_activation function not callable'

    def build_2d_layers(self):
        """
        Builds the 2D convolutional layers with the specified parameters
        """
        self.layers_2d = []

        for index, out_size in enumerate(self.size_2d):  # Here the 2D convolutional layers are generated
            self.layers_2d.append(
                nn.Conv2d(in_channels=self.channels, out_channels=out_size, kernel_size=self.kernel_2d[index],
                          stride=self.stride_2d[index], padding=self.pad_2d[index], groups=self.groups_2d[index],
                          dilation=self.dilation_2d[index]))
            assert self.kernel_2d[index] <= self.height, f'Kernel size too big in 2D convolutional layer {index + 1}'
            assert self.kernel_2d[index] <= self.width, f'Kernel size too big in 2D convolutional layer {index + 1}'
            self.height = (self.height + 2 * self.pad_2d[index] - (self.kernel_2d[index] - 1) - 1) // self.stride_2d[
                index] + 1  # Update self.height

            self.width = (self.width + 2 * self.pad_2d[index] - (self.kernel_2d[index] - 1) - 1) // self.stride_2d[
                index] + 1  # Update self.height

            if self.batchnorm_2d[index]:
                self.layers_2d.append(nn.BatchNorm2d(self.size_2d[index]))

            self.layers_2d.append(self.activation())  # Add activation function

            if self.pool_2d[index]:
                pool = self.pool_2d[index]
                self.layers_2d.append(
                    pool(kernel_size=self.pool_2d_size[index], stride=self.pool_2d_stride[index],
                         padding=self.pool_2d_pad[index]))
                self.height = (self.height + 2 * self.pool_2d_pad[index] - (self.pool_2d_size[index] - 1) - 1) // \
                              self.pool_2d_stride[index] + 1  # Update self.height

                self.width = (self.width + 2 * self.pool_2d_pad[index] - (self.pool_2d_size[index] - 1) - 1) // \
                             self.pool_2d_stride[index] + 1  # Update self.width

            if self.drop_2d[index]:
                self.layers_2d.append(nn.Dropout2d(self.drop_2d[index]))

            self.channels = out_size  # We need this in order to construct the next convolutional layer

        # This will be needed to construct the first linear layer and to flatten the data between the convolutional and linear layers
        self.conv_output_size = self.channels * self.height * self.width
        self.layers_2d = nn.Sequential(*self.layers_2d)

    def build_1d_layers(self):
        """
        Builds the linear layers with the specified parameters
        """
        previous_lin_size = self.conv_output_size
        self.layers_1d = []

        for index, out_size in enumerate(self.size_1d):
            self.layers_1d.append(nn.Linear(previous_lin_size, out_size))

            if self.batchnorm_1d[index]:
                self.layers_1d.append(nn.BatchNorm1d(out_size))

            self.layers_1d.append(self.activation())

            if self.drop_1d[index]:
                self.layers_1d.append(nn.Dropout(self.drop_1d[index]))

            previous_lin_size = out_size

        self.layers_1d.append(
            nn.Linear(previous_lin_size, self.output_size))  # We always have at least one linear layer
        if callable(self.final_activation):
            self.layers_1d.append(self.final_activation())
        self.layers_1d = nn.Sequential(*self.layers_1d)
