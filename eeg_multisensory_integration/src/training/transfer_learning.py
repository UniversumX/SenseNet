"""
def transfer_learning_setup(encoder, trainable_layers):

    # Set up the encoder for transfer learning.

    # Freeze all layers first
    for param in encoder.parameters():
        param.requires_grad = False
    # Unfreeze specified layers
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
"""
def setup_transfer_learning(encoder, cfg):
    """
    Set up the encoder for transfer learning based on the configuration.
    """
    if cfg.transfer_learning.freeze_layers:
        layers_to_freeze = cfg.transfer_learning.layers_to_freeze
        for name, param in encoder.named_parameters():
            if any(layer_name in name for layer_name in layers_to_freeze):
                param.requires_grad = False
    else:
        # Optionally fine-tune all layers
        for param in encoder.parameters():
            param.requires_grad = True