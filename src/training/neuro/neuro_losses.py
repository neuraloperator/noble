"""
Module for computing electrophysiological feature losses for neuronal data.
"""
import inspect
import os
import torch
from typing import Tuple, Callable
from training.neuro.extract_features import extract_features
from training.neuro.differentiable_ap1width import compute_predicted_and_true_ap1_width
from training.neuro.differentiable_sagamplitude import compute_differentiable_sag_amplitude, compute_true_sag_amplitude
import datetime
import matplotlib.pyplot as plt

def _call_with_valid_kwargs(fn: Callable, **kwargs):
    sig = inspect.signature(fn).parameters
    usable = {k: v for k, v in kwargs.items() if k in sig}
    return fn(**usable)


def ap1_width_loss(stimulus: torch.Tensor, voltage_pred: torch.Tensor, voltage_true: torch.Tensor, loss_fn: Callable, time: torch.Tensor, dt: float, ds: int) -> torch.Tensor:
    """
    Compute loss between predicted and true AP1 widths.
    
    """
    predicted_widths, true_widths = compute_predicted_and_true_ap1_width(stimulus, voltage_pred * 1000, voltage_true * 1000, time, dt, ds)

    return loss_fn(predicted_widths, true_widths)


def sag_amplitude_loss(stimulus: torch.Tensor, voltage_pred: torch.Tensor, voltage_true: torch.Tensor, loss_fn: Callable, time: torch.Tensor, dt: float, ds: int) -> torch.Tensor:
    """
    Compute loss between predicted and true sag amplitudes.
    
    """
    predicted_sag_amplitudes = compute_differentiable_sag_amplitude(stimulus, voltage_pred * 1000, time, dt, ds)
    true_sag_amplitudes      = compute_differentiable_sag_amplitude(stimulus, voltage_true * 1000, time, dt, ds)

    return loss_fn(predicted_sag_amplitudes, true_sag_amplitudes)

def compute_feature_loss(stimulus: torch.Tensor, output_pred: torch.Tensor, output_true: torch.Tensor, feature_name: str, loss_fn: Callable, **kwargs) -> torch.Tensor:
    """
    Compute loss for a given electrophysiological feature.
    """
    valid_features = ['AP1_width', 'sag_amplitude']

    FEATURE_FUNCS = {
        "AP1_width": ap1_width_loss,
        "sag_amplitude": sag_amplitude_loss,
    }

    if feature_name not in valid_features:
        raise ValueError(f"Feature '{feature_name}' is not supported. Only {', '.join(valid_features)} are supported.")

    feature_fn = FEATURE_FUNCS[feature_name]

    return _call_with_valid_kwargs(
        feature_fn,
        stimulus=stimulus,
        voltage_pred=output_pred,
        voltage_true=output_true,
        loss_fn=loss_fn,
        **kwargs
    )