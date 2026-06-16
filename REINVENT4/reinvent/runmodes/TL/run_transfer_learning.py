"""Reinvent transfer learning

Reads in a SMILES file and performs transfer learning.
"""

import os
import logging

import torch
import torch.optim as topt

from reinvent.runmodes import TL, create_adapter
from reinvent.utils import setup_reporter, read_smiles_csv_file
from reinvent.chemistry import conversions
from reinvent.chemistry.standardization.rdkit_standardizer import (
    RDKitStandardizer,
)
from reinvent.utils import get_tokens_from_vocabulary
from .validation import TLConfig

logger = logging.getLogger(__name__)


def run_transfer_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str = None,
    write_config: str = None,
    *args,
    **kwargs,
):
    """Run transfer learning with Reinvent

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: log directory for TensorBoard
    :param write_config: callable to write config
    """

    logger.info("Starting Transfer Learning")

    config = TLConfig(**input_config)

    parameters = config.parameters
    scheduler_config = config.scheduler

    model_filename = parameters.input_model_file
    adapter, _, model_type = create_adapter(model_filename, "training", device)

    logger.info(f"Using generator {model_type}")

    # ── Optional layer freezing ────────────────────────────────────────────────
    # freeze_n_layers in [scheduler] TOML: 0=none, 1=embed+layer0, 2=+layer1, etc.
    # This reduces overfitting by only fine-tuning the top LSTM layer + linear head
    _freeze_n = scheduler_config.pop("freeze_n_layers", 0)
    if _freeze_n > 0 and model_type == "Reinvent":
        network = adapter.network
        frozen_count = 0
        # Freeze embedding
        if _freeze_n >= 1:
            for p in network._embedding.parameters():
                p.requires_grad = False
                frozen_count += p.numel()
        # Freeze LSTM layers 0..(_freeze_n-1)
        all_lstm_params = list(network._rnn.parameters())
        # LSTM stores params as {weight_ih_lX, weight_hh_lX, bias_ih_lX, bias_hh_lX}
        # Split by layer index
        for name, param in network._rnn.named_parameters():
            # name looks like weight_ih_l0, bias_hh_l2, etc.
            import re
            m = re.search(r'l(\d+)', name)
            if m and int(m.group(1)) < (_freeze_n - 1):
                param.requires_grad = False
                frozen_count += param.numel()
        trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
        total = sum(p.numel() for p in network.parameters())
        logger.info(f"Froze {frozen_count:,} params | Trainable: {trainable:,}/{total:,} "
                    f"({100*trainable/total:.1f}%) — freeze_n_layers={_freeze_n}")


    smiles_filename = os.path.abspath(parameters.smiles_file)
    do_standardize = parameters.standardize_smiles

    randomize_all_smiles = parameters.randomize_all_smiles
    do_randomize = parameters.randomize_smiles and not randomize_all_smiles

    actions = []
    cols = 0

    # FIXME: move to preprocessing
    if model_type == "Reinvent":
        if do_standardize:
            standardizer = RDKitStandardizer(None, isomeric=False)
            actions.append(standardizer.apply_filter)

        if do_randomize:
            actions.append(conversions.randomize_smiles)
    elif model_type == "Mol2Mol":
        if do_standardize:
            actions.append(conversions.convert_to_standardized_smiles)
    else:
        cols = slice(0, 2, None)

    # NOTE: we expect here that all data will fit into memory
    allowed_tokens = get_tokens_from_vocabulary(adapter.vocabulary)
    smilies = read_smiles_csv_file(
        smiles_filename, cols, allowed_tokens, actions=actions, remove_duplicates=True
    )
    logger.info(f"Read {len(smilies)} input SMILES from {smiles_filename}")

    if not smilies:
        msg = f"Unable to read valid SMILES from {smiles_filename}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    validation_smiles_filename = parameters.validation_smiles_file
    validation_smilies = None

    if validation_smiles_filename:
        validation_smiles_filename = os.path.abspath(validation_smiles_filename)
        validation_smilies = read_smiles_csv_file(
            validation_smiles_filename,
            cols,
            allowed_tokens,
            actions=actions,
            remove_duplicates=True,
        )
        logger.info(
            f"Read {len(validation_smilies)} validation SMILES from {validation_smiles_filename}"
        )

    if model_type == "Mol2Mol":
        model_size = adapter.network.encoder.layers[0].self_attn.linears[0].in_features
        lr_config = TL.LambdaLRConfiguration(**scheduler_config)

        optimizer = topt.Adam(
            adapter.get_network_parameters(),
            lr=lr_config.lr,
            betas=(lr_config.beta1, lr_config.beta2),
            eps=lr_config.eps,
            capturable=str(device) != "cpu",  # workaround for pytorch 1.11
        )

        lr_step = (
            lambda step: lr_config.factor
            / 1e-4
            * (
                model_size ** (-0.5)
                * min(
                    max(step, 1) ** (-0.5),
                    max(step, 1) * lr_config.warmup ** (-1.5),
                )
            )
        )

        lr_scheduler = topt.lr_scheduler.LambdaLR(optimizer, lr_step)
    else:
        _weight_decay = scheduler_config.pop("weight_decay", 0.0)
        lr_config = TL.StepLRConfiguration(**scheduler_config)
        optimizer = topt.Adam(
            adapter.get_network_parameters(),
            lr=lr_config.lr,
            weight_decay=_weight_decay,
        )

        lr_scheduler = topt.lr_scheduler.StepLR(
            optimizer, step_size=lr_config.step, gamma=lr_config.gamma
        )

    runner_class = getattr(TL, f"{model_type}")

    optimize = runner_class(
        adapter,
        smilies,
        validation_smilies,
        tb_logdir,
        parameters,
        optimizer,
        lr_scheduler,
        lr_config,
        tb_isim=parameters.tb_isim,
        zero_epoch=parameters.training_zero_epoch_start,
    )

    if "responder" in config:
        url = config.responder.endpoint
        success = setup_reporter(url)

        if success:
            logger.info(f"Remote reporting to {url}")

    if callable(write_config):
        write_config(config.model_dump())

    optimize()
