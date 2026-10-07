"""Training-only fault regularization; serialized inference architecture is unchanged."""
import math
import torch.nn.functional as F


def configure_cached_heads(model, config):
    policy = config.get("head_training_policy", "all_heads")
    dropout = config.get("fault_hidden_dropout", 0.)
    smoothing = config.get("fault_label_smoothing", 0.)
    if policy not in ("all_heads", "fault_heads", "application_fault_heads"):
        raise ValueError("Unknown head training policy")
    if not math.isfinite(dropout) or not 0. <= dropout < 1.:
        raise ValueError("Invalid fault dropout")
    if not math.isfinite(smoothing) or not 0. <= smoothing < 1.:
        raise ValueError("Invalid fault label smoothing")
    if policy in ("fault_heads", "application_fault_heads") and not config.get("initialization_artifact"):
        raise ValueError("Fault-only training requires an existing parent model")
    if policy == "application_fault_heads" and not model.application_fault_names:
        raise ValueError("Application-only training requires application heads")
    if model.application_fault_names and policy != "application_fault_heads":
        raise ValueError("Application experts require application-only training")
    fault_prefixes = ("fault_head.", "numeric_fault.", "numeric_local_fault.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("backbone.") and (
            policy == "all_heads" or (name.startswith("application_fault_heads.") if policy == "application_fault_heads" else name.startswith(fault_prefixes)))
    handles = []
    if dropout:
        branches = list(model.application_fault_heads) if policy == "application_fault_heads" else [model]
        for branch in branches:
            for name in ("numeric_fault", "numeric_local_fault"):
                module = branch[name] if hasattr(branch,"keys") and name in branch else getattr(branch,name,None)
                if module is not None:
                    handles.append(module[1].register_forward_hook(
                        lambda module, inputs, output: F.dropout(output, p=dropout, training=module.training)))
    return handles
