#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Precompute the visually aligned initialization of visual-token embeddings.

Board images from expert trajectories are encoded twice: by the visual
tokenizer (one code per 16x16 patch) and by the model's own vision encoder
(language-space features per image soft token, i.e. per merged or pooled patch;
see ``model.families``). Each code's embedding is the mean feature of the patches
it was assigned to; unseen codes get the global mean. Uses every visible GPU::

    uv run python -m mental_imagery.prepare.embed_init model=qwen3_5_9b visual_tokens=llamagen
"""

from __future__ import annotations

import os
from collections import defaultdict

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForImageTextToText

from mental_imagery.model.families import resolve_family
from mental_imagery.model.loading import DTYPES, check_pinned_revision, load_processor
from mental_imagery.puzzles import build_engine, build_game_configs, excluded_seeds
from mental_imagery.visual_tokenizers.codec import PATCH_SIZE, build_visual_tokenizer


def accumulate(codes, features, embedding_sum, embedding_count, code_grid, vision_grid) -> None:
    """Add each patch's feature to its code's sum, mapping the vision grid onto the code grid."""
    (code_h, code_w), (vision_h, vision_w) = code_grid, vision_grid
    features = features.reshape(codes.shape[0], vision_h, vision_w, -1)
    if code_h % vision_h == 0 and code_w % vision_w == 0:
        features = features.repeat_interleave(code_h // vision_h, dim=1).repeat_interleave(
            code_w // vision_w, dim=2
        )
    else:
        features = torch.nn.functional.interpolate(
            features.permute(0, 3, 1, 2), size=code_grid
        ).permute(0, 2, 3, 1)
    codes = codes.reshape(-1)
    embedding_sum.index_add_(0, codes, features.reshape(len(codes), -1))
    embedding_count.index_add_(0, codes, torch.ones(len(codes), dtype=torch.float64))


def worker(rank, world_size, config, all_sums, all_counts, all_image_counts):
    config = OmegaConf.create(config)
    episodes = config.num_episodes // world_size + (rank < config.num_episodes % world_size)
    if episodes == 0:
        return
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    family = resolve_family(config.model.family)
    processor = load_processor(config.model.name, config.model.revision, config.visual_tokens)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model.name, revision=config.model.revision, dtype=DTYPES[config.model.dtype]
    )
    family.check_model_type(model.config)
    vision_encoder = family.vision_encoder(model, device)
    del model
    tokenizer = build_visual_tokenizer(config.visual_tokens, device, grid_size=None)
    engine = build_engine(
        processor.tokenizer,
        family=family,
        engine_config=config.engine,
        visual_tokens_config=config.visual_tokens,
        seed=config.engine.seed,
        rank=rank,
    )
    loader = engine.loader(
        games=build_game_configs(config.games),
        n=config.batch_size,
        prefetch=config.engine.prefetch,
        excluded_seeds=excluded_seeds(config.games),
    )

    num_batches = -(-episodes // config.batch_size)
    for batch_index in range(num_batches):
        rollout = loader.next()
        images = rollout.initial_images
        rollout.teacher_force_labels(
            config.max_tokens
        )  # renders the boards along the expert trajectory
        images = images + rollout.take_frames()[0]
        all_image_counts[rank] += len(images)
        by_size = defaultdict(list)
        for image in images:
            by_size[image.size].append(image)
        for (width, height), group in by_size.items():
            features, vision_grid = vision_encoder.features(
                processor, group, config.vision_batch_size
            )
            codes = torch.stack(tokenizer.encode(group)).cpu()
            accumulate(
                codes,
                features,
                all_sums[rank],
                all_counts[rank],
                (height // PATCH_SIZE, width // PATCH_SIZE),
                vision_grid,
            )
        print(
            f"[embed_init][gpu {rank}] batch {batch_index + 1}/{num_batches}: "
            f"{int((all_counts[rank] > 0).sum())} codes seen",
            flush=True,
        )


@hydra.main(config_path="../configs", config_name="embed_init", version_base=None)
def main(config) -> None:
    check_pinned_revision(config.model.name, config.model.revision)
    world_size = torch.cuda.device_count()
    codebook_size = config.visual_tokens.codebook_size
    hidden_size = AutoConfig.from_pretrained(
        config.model.name, revision=config.model.revision
    ).text_config.hidden_size
    load_processor(
        config.model.name, config.model.revision, config.visual_tokens
    )  # download once before spawning
    build_visual_tokenizer(config.visual_tokens, "cpu", grid_size=None)

    all_sums = torch.zeros(
        world_size, codebook_size, hidden_size, dtype=torch.float64
    ).share_memory_()
    all_counts = torch.zeros(world_size, codebook_size, dtype=torch.float64).share_memory_()
    all_image_counts = torch.zeros(world_size, dtype=torch.int64).share_memory_()
    resolved = OmegaConf.to_container(config, resolve=True)
    mp.spawn(
        worker,
        nprocs=world_size,
        args=(world_size, resolved, all_sums, all_counts, all_image_counts),
    )

    embedding_sum, counts = all_sums.sum(dim=0), all_counts.sum(dim=0)
    observed = counts > 0
    embeddings = torch.zeros(codebook_size, hidden_size, dtype=torch.float64)
    embeddings[observed] = embedding_sum[observed] / counts[observed].unsqueeze(1)
    embeddings[~observed] = embeddings[observed].mean(dim=0)
    metadata = {
        "model_name": config.model.name,
        "model_revision": config.model.revision,
        "visual_tokenizer": config.visual_tokens.tokenizer,
        "fsq_checkpoint": config.visual_tokens.fsq_checkpoint,
        "codebook_size": codebook_size,
        "render_resolution": config.engine.render_resolution,
        "num_episodes": config.num_episodes,
        "num_images": int(all_image_counts.sum()),
        "codes_seen": int(observed.sum()),
    }
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings.to(DTYPES[config.model.dtype]),
            "counts": counts,
            "metadata": metadata,
        },
        config.output_path,
    )
    print(
        f"[embed_init] wrote {config.output_path}: {metadata['codes_seen']}/{codebook_size} codes seen",
        flush=True,
    )


if __name__ == "__main__":
    main()
