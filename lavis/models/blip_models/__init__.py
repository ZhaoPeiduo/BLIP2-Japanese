"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import List

from torch import nn
import datetime
import logging
import os
import time

import lavis.common.dist_utils as dist_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.logger import MetricLogger
from lavis.models.vit import interpolate_pos_embed


def tie_encoder_decoder_weights(
    encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str
):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logging.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + " is tied")
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set(
                [module_name + "/" + sub_name for sub_name in encoder_modules.keys()]
            )
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                        decoder_modules[decoder_name],
                        type(encoder_modules[encoder_name]),
                    ) and len(encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(
        decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key
    )

def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_output = model.text_encoder.forward_text(text_input)
        text_embed = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :])
        )
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    if hasattr(model.tokenizer, "enc_token_id"):
        text_ids[:, 0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat = model.visual_encoder.forward_features(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        encoder_output = image_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            text_ids[topk_idx],
            attention_mask=text_atts[topk_idx],
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):

        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(model.device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            model.device
        )
        output = model.text_encoder(
            text_ids[start + i].repeat(k_test, 1),
            attention_mask=text_atts[start + i].repeat(k_test, 1),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
