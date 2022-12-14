"""Collections of functions and classes used in the notebooks.
"""

from time import time
from collections import defaultdict
from shutil import copy
import torch as pt
import pandas as pd


def run_epoch(
    model: pt.nn.Module,
    optimizer: pt.optim.Optimizer,
    data_loader: pt.utils.data.DataLoader,
    loss_func: pt.nn.Module,
    device: str,
    results: dict,
    score_funcs: dict,
    prefix: str,
) -> float:
    """Perform one optimizing step on a model.

    This loop is a slightly modified version of 'run_epoch'
    provided in chapter 5 of 'Inside Deep Learning' by Edward Raff;
    refer to:
    https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py
    """

    # keeping track of loss, predictions, and time
    running_loss, labels_true, labels_pred = [], [], []
    start_time = time()

    # loop over all batches
    for features, labels in data_loader:
        features, labels = features.to(device), labels.to(device)
        pred = model(features)
        loss = loss_func(pred, labels)
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss.append(loss.item())

        # the dataset might get shuffled in the next loop
        if len(score_funcs) > 0:
            labels_true.extend(labels.detach().cpu().tolist())
            labels_pred.extend(pred.detach().cpu().tolist())

    # keep track of performance
    results[f"{prefix}_loss"].append(sum(running_loss) / len(running_loss))
    for name, func in score_funcs.items():
        results[f"{prefix}_{name}"].append(func(labels_true, labels_pred))

    return time() - start_time


def train_model(
    model: pt.nn.Module,
    loss_func: pt.nn.Module,
    train_loader: pt.utils.data.DataLoader,
    val_loader: pt.utils.data.DataLoader = None,
    test_loader: pt.utils.data.DataLoader = None,
    score_funcs: dict = {},
    epochs: int = 100,
    device: str = "cpu",
    checkpoint_file: str = None,
    log_all: bool = False,
    lr_schedule: pt.optim.lr_scheduler._LRScheduler = None,
    optimizer: pt.optim.Optimizer = None,
) -> pd.DataFrame:
    """Perform one optimizing step on a model.

    This function is a slightly modified version of 'train_network'
    provided in chapter 5 of 'Inside Deep Learning' by Edward Raff;
    refer to:
    https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py
    """

    # dictionary for keeping track of training performance
    results = defaultdict(list)
    best_loss = float("inf")
    ref_loss = "train_loss" if val_loader is None else "val_loss"

    # use AdamW as default optimizer if none specified
    delete_optimizer = False
    if optimizer is None:
        optimizer = pt.optim.AdamW(model.parameters())
        delete_optimizer = True

    total_train_time = 0.0
    model.to(device)
    for e in range(epochs):
        # model update
        model = model.train()
        total_train_time += run_epoch(
            model,
            optimizer,
            train_loader,
            loss_func,
            device,
            results,
            score_funcs,
            prefix="train",
        )
        results["epoch"].append(e)
        results["total_time"].append(total_train_time)
        message = f"Training loss: {results['train_loss'][-1]:2.6e}"

        # validation dataset
        if val_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch(
                    model,
                    optimizer,
                    val_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="val",
                )
            message += f"; Validation loss: {results['val_loss'][-1]:2.6e}"

        # update of learning rate
        if lr_schedule is not None:
            if isinstance(lr_schedule, pt.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(results["val_loss"][-1])
            else:
                lr_schedule.step()

        # test dataset
        if test_loader is not None:
            model = model.eval()
            with pt.no_grad():
                _ = run_epoch(
                    model,
                    optimizer,
                    test_loader,
                    loss_func,
                    device,
                    results,
                    score_funcs,
                    prefix="test",
                )
            message += f"; Test loss: {results['test_loss'][-1]:2.6e}"

        # save checkpoint
        if checkpoint_file is not None:
            suffix = f"_epoch_{e}" if log_all else ""
            pt.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "results": results,
                },
                checkpoint_file + suffix,
            )
            latest_loss = results[ref_loss][-1]
            if latest_loss < best_loss:
                best_loss = latest_loss
                copy(checkpoint_file + suffix, checkpoint_file + "_best")

        print("\r", f"Epoch {e:4d}/{epochs - 1} - " + message, end="")

    # if the optimizer was created in the training loop,
    # delete if to avoid unwanted side effects
    if delete_optimizer:
        del optimizer

    return pd.DataFrame.from_dict(results)


def create_simple_network(
    n_in: int, n_out: int, n_neurons: int, n_hidden: int, activation: pt.nn.Module
) -> pt.nn.Sequential:
    layers = [pt.nn.Linear(n_in, n_neurons), activation()]
    for _ in range(n_hidden):
        layers.append(pt.nn.Linear(n_neurons, n_neurons))
        layers.append(activation())
    layers.append(pt.nn.Linear(n_neurons, n_out))
    return pt.nn.Sequential(*layers)
