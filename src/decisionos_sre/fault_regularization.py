"""Training-only fault regularization; serialized inference architecture is unchanged."""
import math
import torch.nn.functional as F


def configure_cached_heads(model, config):
    policy = config.get("head_training_policy", "all_heads")
    dropout = config.get("fault_hidden_dropout", 0.)
    smoothing = config.get("fault_label_smoothing", 0.)
    if policy not in ("all_heads", "fault_heads"):
        raise ValueError("Unknown head training policy")
    if not math.isfinite(dropout) or not 0. <= dropout < 1.:
        raise ValueError("Invalid fault dropout")
    if not math.isfinite(smoothing) or not 0. <= smoothing < 1.:
        raise ValueError("Invalid fault label smoothing")
    if policy == "fault_heads" and not config.get("initialization_artifact"):
        raise ValueError("Fault-only training requires an existing parent model")
    fault_prefixes = ("fault_head.", "numeric_fault.", "numeric_local_fault.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("backbone.") and (
            policy == "all_heads" or name.startswith(fault_prefixes))
    handles = []
    if dropout:
        for name in ("numeric_fault", "numeric_local_fault"):
            module = getattr(model, name, None)
            if module is not None:
                handles.append(module[1].register_forward_hook(
                    lambda module, inputs, output: F.dropout(output, p=dropout, training=module.training)))
    return handles
