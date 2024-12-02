import torch
import numpy as np


class DotDict:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def masked_mae_loss(predict, true, mask):
    mae = torch.sum(torch.absolute(predict - true) * (1 - mask)) / (
        torch.sum(1 - mask) + 1e-5
    )
    return mae


def masked_mse_loss(predict, true, mask):
    mse = torch.sum((predict - true) ** 2 * (1 - mask)) / (torch.sum(1 - mask) + 1e-5)
    return mse


def masked_rmse_loss(predict, true, mask):
    rmse = torch.sqrt(
        torch.sum((predict - true) ** 2 * (1 - mask)) / (torch.sum(1 - mask)) + 1e-5
    )
    return rmse


def masked_mape_loss(predict, true, mask):
    mape = torch.sum(torch.absolute((predict - true) * (1 - mask))) / (
        torch.sum(true * (1 - mask)) + 1e-5
    )
    return mape


def missed_eval_torch(predict, true, mask):
    mae = torch.sum(torch.absolute(predict - true) * (1 - mask)) / torch.sum(1 - mask)
    rmse = torch.sqrt(
        torch.sum((predict - true) ** 2 * (1 - mask)) / torch.sum(1 - mask)
    )
    mape = torch.sum(torch.absolute((predict - true) * (1 - mask))) / (
        torch.sum(torch.absolute(true * (1 - mask))) + 1e-5
    )
    return mae, rmse, mape


def missed_eval_np(predict, true, mask):
    """
    predict: [samples, seq_len, feature_dim]
    true: [samples, seq_len, feature_dim]
    mask: [samples, seq_len, feature_dim]
    """
    predict, true = np.asarray(predict), np.asarray(true)
    mae = np.sum(np.absolute(predict - true) * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    mse = np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    rmse = np.sqrt(
        np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    mape = np.sum(np.absolute((predict - true) * (1 - mask))) / (
        np.sum(np.absolute(true * (1 - mask))) + 1e-5
    )
    R2 = 1 - mse / (
        np.sum((true - np.mean(true)) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    return mae, rmse, mape, mse, R2


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler   
    forecast = forecast * scaler + mean_scaler  

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)
