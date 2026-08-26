"""Load processor metadata from the runtime artifact bundle."""

from __future__ import annotations

import os
from pathlib import Path


def load_o5_processor(assets_dir: str):
    """Load the vendored O5 processor with metadata from runtime assets.

    The Token2Wav directory and Hugging Face processor metadata are deployed
    as one assets bundle, but the latter may also be supplied separately for
    older layouts.  Model code always comes from this checkout.
    """
    from modeling.o5.processing_minicpmo import (
        MiniCPMAAudioProcessor,
        MiniCPMOProcessor,
        MiniCPMVImageProcessor,
    )
    from modeling.o5.tokenization_minicpmo_fast import MiniCPMOTokenizerFast

    if not assets_dir:
        raise FileNotFoundError("O5 processor assets are required; set O5_ASSETS_DIR")

    processor_candidates = [
        Path(assets_dir),
        Path(os.environ["O5_PROCESSOR_DIR"])
        if os.environ.get("O5_PROCESSOR_DIR")
        else None,
        Path(os.environ["MODEL_PATH"]) if os.environ.get("MODEL_PATH") else None,
        Path("/user/weihongliang/MiniCPM-o-4_6"),
    ]
    processor_dir = next(
        (
            candidate
            for candidate in processor_candidates
            if candidate is not None
            and (candidate / "preprocessor_config.json").is_file()
            and (
                (candidate / "tokenizer.json").is_file()
                or (candidate / "tokenizer_config.json").is_file()
            )
        ),
        None,
    )
    if processor_dir is None:
        raise FileNotFoundError(
            "O5 processor metadata is required; checked assets_dir, "
            "O5_PROCESSOR_DIR, MODEL_PATH, and the shared model metadata path"
        )

    image_processor = MiniCPMVImageProcessor.from_pretrained(str(processor_dir))
    audio_processor = MiniCPMAAudioProcessor.from_pretrained(str(processor_dir))
    tokenizer = MiniCPMOTokenizerFast.from_pretrained(str(processor_dir))
    return MiniCPMOProcessor(
        image_processor=image_processor,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
    )
