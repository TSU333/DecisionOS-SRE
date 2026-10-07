"""Fault experts routed only by the public incident application field."""
import copy
from torch import nn


def application_names(names):
    if isinstance(names, str):
        raise ValueError("Application heads require an ordered list")
    names = tuple(names)
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("Application head names must be unique nonempty strings")
    return names


def copy_fault_heads(model):
    names = ("fault_head", "numeric_fault", "numeric_local_fault")
    return nn.ModuleDict({n: copy.deepcopy(getattr(model, n)) for n in names if hasattr(model, n)})


def migrate_application_heads(state, model, old_names, new_names, upgrade):
    old_names, new_names = application_names(old_names), application_names(new_names)
    if old_names == new_names:
        return []
    if old_names or not new_names or upgrade != "copy_parent_fault_v1":
        raise ValueError("Unsupported application head migration or reordered mapping")
    copied = []
    expected = model.state_dict()
    for name in expected:
        if name.startswith("application_fault_heads."):
            source = name.split(".", 2)[2]
            if source not in state or state[source].shape != expected[name].shape:
                raise ValueError("Application head parent shape mismatch")
            state[name] = state[source].clone()
            copied.append(name)
    return copied
