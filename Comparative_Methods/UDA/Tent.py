import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.training.loss_functions.deep_supervision import DeepSupervisionWrapper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from copy import deepcopy


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @staticmethod
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # Forward pass
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Take main output (nnUNet's deep supervision output)

        # Compute entropy
        entropy = softmax_entropy(outputs)

        # Calculate mean entropy as loss (minimize entropy)
        loss = entropy.mean()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        return outputs


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    x = F.softmax(x, dim=1)
    entropy = -torch.sum(x * torch.log(x + 1e-8), dim=1)
    return entropy


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def setup_tent_nnunet(model, steps=1, lr=1e-5, episodic=True):
    """Set up Tent adapter for nnUNet model"""
    # Ensure model is in eval mode (but enable gradients)
    model.eval()

    # Adapt only batch norm layers and last layers (optional)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.requires_grad_(True)
            # Set to training mode to update running stats
            m.track_running_stats = True
            m.train()
        else:
            # Optional: freeze other layers or only unfreeze partial layers
            # m.requires_grad_(False)
            pass

    # Select parameters to optimize
    params = [p for p in model.parameters() if p.requires_grad]

    # Create optimizer (usually use SGD or Adam)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.)
    # optimizer = torch.optim.Adam(params, lr=lr)

    # Create Tent adapter
    tent_model = Tent(model, optimizer, steps=steps, episodic=episodic)

    return tent_model


# Usage example
if __name__ == "__main__":
    # Assume we have a pre-trained nnUNet model
    # This is example initialization, load pre-trained model in practice
    num_classes = 4
    base_num_features = 32
    num_pool = 4
    net_num_pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
    net_conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)

    # Create nnUNet model
    nnunet_model = Generic_UNet(
        input_channels=1,
        base_num_features=base_num_features,
        num_classes=num_classes,
        num_pool=num_pool,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv3d,
        norm_op=nn.BatchNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={'p': 0, 'inplace': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
        deep_supervision=True,
        net_num_pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
        net_conv_kernel_sizes=net_conv_kernel_sizes,
        upscale_logits=False,
        convolutional_pool=True,
        convolutional_upsample=True
    )

    # Wrap deep supervision
    nnunet_model = DeepSupervisionWrapper(nnunet_model)

    # Set up Tent adapter
    tent_adapter = setup_tent_nnunet(nnunet_model, steps=1, lr=1e-5)

    # Inference during testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tent_adapter.to(device)

    # Simulate input (batch, channels, x, y, z)
    test_input = torch.randn(1, 1, 64, 64, 64).to(device)

    # Inference (automatic test-time adaptation)
    with torch.no_grad():  # Note: Tent enables gradients internally
        outputs = tent_adapter(test_input)

    # Get final prediction
    if isinstance(outputs, tuple):
        final_pred = outputs[0]
    else:
        final_pred = outputs

    pred_seg = torch.argmax(final_pred, dim=1)
    print(f"Predicted segmentation shape: {pred_seg.shape}")