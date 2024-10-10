import torch


def pad_tensor(tensor: torch.Tensor, target_shape: torch.Size, value: float):
    padding = [0] * 2 * len(tensor.shape)
    for i, target_len in enumerate(target_shape):
        if i < len(tensor.shape):
            padding[-1 - 2 * i] = target_len - tensor.shape[i]

    return torch.nn.functional.pad(tensor, padding, mode="constant", value=value)


def pad_model_state_dict(state_dict: dict, model: torch.nn.Module, value: float):
    target_state_dict = model.state_dict()
    for param_name in state_dict.keys():
        params: torch.Tensor = state_dict[param_name]
        target_shape: torch.Size = target_state_dict[param_name].shape

        if params.shape == target_shape:
            continue

        print(f"Warning: Padded {param_name} from {params.shape} to {target_shape}")
        state_dict[param_name] = pad_tensor(params, target_shape, value)


def pad_optimizer_state_dict(
    state_dict: dict,
    model: torch.nn.Module,
    exp_avg_value: float,
    exp_avg_sq_value: float,
):
    model_state_dict = model.state_dict()
    for i, param_name in enumerate(model_state_dict.keys()):
        params: dict[str, torch.Tensor] = state_dict["state"][i]
        target_shape: torch.Size = model_state_dict[param_name].shape

        if not param_name.endswith((".weight", ".bias")) or (
            params["exp_avg"].shape == target_shape
        ):
            continue

        params["exp_avg"] = pad_tensor(params["exp_avg"], target_shape, exp_avg_value)
        params["exp_avg_sq"] = pad_tensor(
            params["exp_avg_sq"], target_shape, exp_avg_sq_value
        )
        print(f"Warning: Padded {param_name} in optimizer")
