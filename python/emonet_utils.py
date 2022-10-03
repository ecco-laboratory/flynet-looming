# %%
# imports per usual
import torch
from torch import nn

# %%
# instantiate emonet (from onnx) pytorch class :3

# Can't use torchvision.models.AlexNet()
# because it comes out slightly diff from Matlab. Lol
class EmoNet(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        alexnet_lrn_alpha = 9.999999747378752e-05

        # TODO: still needs the onnx initializer, whatever that thing is
        # Initialized with an empty module?
        self.initializers = nn.Module()
        # initializer has one weight per RGB per pixel
        self.initializers.register_parameter(name='onnx_initializer_0',param=nn.Parameter(torch.ones(torch.Size([1, 3, 227, 227]))))
        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.Conv_0 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # Adjust each weight independently
        # ReLU contains no learnable parameters, so the positive values aren't scaled and are left as is
        self.Relu_0 = nn.ReLU()
        # Sort of kind of does lateral inhibition
        # Same input/output weight dimensions
        # Just with the values adjusted
        self.LRN_0 = nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1)
        # Pooling operates separately on each convolved feature map
        # Does it cut down the number of nodes or leave it intact?
        # when kernel_size == stride, each weight is included in only one pool
        # thus downsampling by a factor of kernel_size (or stride)
        # In the past, pooling was the "trendy" way of downsampling (to reduce training computations)
        # while retaining some kind of translation invariance for the incoming features
        # but pooling is neither necessary nor sufficient for those two
        self.MaxPool_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # Do it all again but with different convolutional dimensions
        # channels per "pixel" going up, n "pixels" actually staying the same
        self.Conv_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.Relu_1 = nn.ReLU()
        self.LRN_1 = nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # and again, a few rounds of channels going up, pixels staying the same
        self.Conv_2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.Relu_2 = nn.ReLU()
        self.Conv_3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2)
        self.Relu_3 = nn.ReLU()
        self.Conv_4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2)
        self.Relu_4 = nn.ReLU()
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.Conv_5 = nn.Conv2d(256, 4096, kernel_size=6, stride=1)
        self.Relu_5 = nn.ReLU()
        # a conv layer becomes fully connected when kernel_size and stride are both 1?
        # I would think it was fully connected once there's only one pixel left...
        # TODO: Confirm that this layer has only one pixel
        self.Conv_6 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1)
        self.Relu_6 = nn.ReLU()
        # Make sure this outputs 20 channels on one pixel
        # AlexNet that comes with torchvision has this as a Linear module, not Conv2d
        # I feel like they are equivalent when the nodes coming in are 1x1px
        # but this feels hackier somehow. Matlab! Dum dum!
        self.Conv_7 = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)
        self.Flatten_0 = nn.Flatten()
        self.Softmax_0 = nn.Softmax(dim=-1)
            
        # TODO: What happens when ReLU is inplace vs not (as ONNX import)?
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        # Don't pass through the initializers module because... it doesn't DO anything to the data
        x = x.to(torch.float)
        x = self.Conv_0(x)
        x = self.Relu_0(x)
        x = self.LRN_0(x)
        x = self.MaxPool_0(x)
        x = self.Conv_1(x)
        x = self.Relu_1(x)
        x = self.LRN_1(x)
        x = self.MaxPool_1(x)
        x = self.Conv_2(x)
        x = self.Relu_2(x)
        x = self.Conv_3(x)
        x = self.Relu_3(x)
        x = self.Conv_4(x)
        x = self.Relu_4(x)
        x = self.MaxPool_2(x)
        x = self.Conv_5(x)
        x = self.Relu_5(x)
        x = self.Conv_6(x)
        x = self.Relu_6(x)
        x = self.Conv_7(x)
        x = self.Flatten_0(x)
        x = self.Softmax_0(x)
        return x

# %%
# Pythonic EmoNet (same as above, minus technically unnecessary layers and grouped nicer)
class EmoNetPythonic(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        alexnet_lrn_alpha = 9.999999747378752e-05

        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=6, stride=1),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        x = x.to(torch.float)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.classifier(x)

        return x

# %%
# Loopy headless EmoNet (effectively Alexnet, minus the last conv aka MLP layer, looping over frames in a video)
class EmoNetHeadlessVideo(nn.Module):
    def __init__(self) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        alexnet_lrn_alpha = 9.999999747378752e-05

        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=6, stride=1),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        x = x.to(torch.float)
        # I don't know how to preallocate this and then assign by slicing
        # so we'll start with an empty one and then concatenate onto it
        output = torch.Tensor()

        # loop over FRAMES
        # Weirdly, I think this works by processing the nth frame from each batch simultaneously
        for t in range(x.size(1)):
            # input dims for a batch of videos:
            # batch, n frames, channels, height, width
            x_t = x[:, t, :, :, :].squeeze()
            # need this because the batch dimension is necessary but will be squeezed if it was 1
            if x_t.dim() < 4:
                x_t = x_t.unsqueeze(0)
            
            x_t = self.conv_0(x_t)
            x_t = self.conv_1(x_t)
            x_t = self.conv_2(x_t)
            x_t = self.conv_3(x_t)
            x_t = self.conv_4(x_t)
            x_t = self.conv_5(x_t)
            # After this layer, activation dims should be batch x 4096
            x_t = self.conv_6(x_t)
            print('iteration:', t)
            print('size of x_t before squeezing:', x_t.size())
            x_t = x_t.squeeze()
            print('size of x_t after squeezing:', x_t.size())
            # again, might need to unsqueeze the batch dimension
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            # Need to fencepost this because stack() won't stack onto an empty tensor
            if t == 0:
                output = x_t
            else:
                output = torch.stack((output, x_t))

        return output
