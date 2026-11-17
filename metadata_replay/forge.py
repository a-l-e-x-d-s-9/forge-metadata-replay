"""The Forge-facing metadata and generation adapter."""
from __future__ import annotations

import base64
import copy
import io
import math
import re
import secrets
from pathlib import Path

from PIL import Image

from .queue import Entry, identity

CHOICES = ["Prompt", "Negative prompt", "Size", "Model", "Original seed", "Sampler / scheduler",
           "Steps", "CFG scale", "Distilled CFG scale", "Clip skip", "VAE / text encoders", "Hires fix"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".tif", ".tiff", ".jxl"}


def thumbnail(image):
    image = image.copy()
    image.thumbnail((144, 144), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def parse_infotext(infotext):
    from modules import infotext_utils

    params = infotext_utils.parse_generation_parameters(infotext, skip_fields=[])
    # Forge's paste parser can strip prompt styles, omit the currently loaded model,
    # and discard Clip skip. Keep source literals without changing global options.
    *lines, last = infotext.strip().split("\n")
    if len(infotext_utils.re_param.findall(last)) < 3:
        lines.append(last)
        last = ""
    positive, negative, is_negative = [], [], False
    for line in lines:
        if line.lstrip().startswith("Negative prompt:"):
            line = line.lstrip()[len("Negative prompt:"):].lstrip()
            is_negative = True
        (negative if is_negative else positive).append(line)
    params["Prompt"] = "\n".join(positive)
    params["Negative prompt"] = "\n".join(negative)
    for key, value in infotext_utils.re_param.findall(last):
        if key in ("Model", "Model hash", "Clip skip", "VAE") or key.startswith("Module "):
            params[key] = infotext_utils.unquote(value)
    return params


def load_image(path):
    from modules import images

    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Image exceeds 128 MiB.")
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        infotext, _ = images.read_info_from_image(image)
        if not infotext or not infotext.strip():
            raise ValueError("No supported generation metadata found.")
        params = parse_infotext(infotext)
        return Entry(identity(data), path.name, infotext, params, source_preview=thumbnail(image))


def number(params, key, cast, minimum=None, maximum=None):
    value = cast(params[key])
    if not math.isfinite(float(value)) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {key}: {params[key]}")
    return value


def build_processing(base, entry, selected, seed, batch):
    from modules import sd_models, sd_samplers

    p = copy.copy(base)
    for name in ("override_settings", "extra_generation_params", "comments"):
        setattr(p, name, copy.deepcopy(getattr(base, name, {})))
    for name in ("styles", "extra_result_images", "latents_after_sampling", "pixels_after_sampling"):
        setattr(p, name, list(getattr(base, name, []) or []))
    p.extra_result_images = []
    p.n_iter = 1
    p.do_not_save_grid = True
    p.override_settings_restore_afterwards = True
    p.seed = (seed + batch * base.batch_size) % (2**32)
    params = entry.parameters
    if "Original seed" not in selected:
        p.subseed, p.subseed_strength = -1, 0
        p.seed_resize_from_h = p.seed_resize_from_w = 0
    if "Prompt" in selected:
        p.prompt = params.get("Prompt", base.prompt)
        p.styles = []
    if "Negative prompt" in selected:
        p.negative_prompt = params.get("Negative prompt", base.negative_prompt)
    for label, key, attribute, cast, low, high in (
        ("Steps", "Steps", "steps", int, 1, 10000),
        ("CFG scale", "CFG scale", "cfg_scale", float, 0, 1000),
        ("Distilled CFG scale", "Distilled CFG Scale", "distilled_cfg_scale", float, 0, 1000),
        ("Size", "Size-1", "width", int, 64, 32768),
        ("Size", "Size-2", "height", int, 64, 32768),
    ):
        if label in selected and key in params:
            setattr(p, attribute, number(params, key, cast, low, high))
    if "Sampler / scheduler" in selected:
        if "Sampler" in params:
            sampler = sd_samplers.samplers_map.get(str(params["Sampler"]).lower())
            if sampler is None:
                raise ValueError(f"Sampler is not installed: {params['Sampler']}")
            p.sampler_name = sampler
        if "Schedule type" in params:
            from modules import sd_schedulers
            scheduler = str(params["Schedule type"])
            if scheduler.lower() not in {k.lower() for k in sd_schedulers.schedulers_map}:
                raise ValueError(f"Scheduler is not installed: {scheduler}")
            p.scheduler = scheduler
    if "Clip skip" in selected and "Clip skip" in params:
        p.override_settings["CLIP_stop_at_last_layers"] = number(params, "Clip skip", int, 1, 12)
    if "Model" in selected:
        refs = [params.get("Model hash"), params.get("Model")]
        refs = [str(ref) for ref in refs if ref]
        model = next((match for ref in refs if (match := sd_models.get_closet_checkpoint_match(ref))), None)
        if refs and model is None:
            raise ValueError(f"Checkpoint is not installed: {' / '.join(refs)}")
        if model:
            p.override_settings["sd_model_checkpoint"] = model.title
    if "VAE / text encoders" in selected:
        from modules_forge import main_entry
        modules = [str(v) for k, v in params.items() if k.startswith("Module ")]
        if not modules and "VAE/TE" in params:
            modules = params["VAE/TE"]
        if modules:
            resolved = []
            for module in modules:
                matches = [v for k, v in main_entry.module_list.items()
                           if module in (k, str(v), Path(k).stem)]
                if not matches:
                    raise ValueError(f"VAE / text encoder is not installed: {module}")
                resolved.append(matches[0])
            p.override_settings["forge_additional_modules"] = resolved
        elif "VAE" in params:
            p.override_settings["sd_vae"] = str(params["VAE"])
    if "Hires fix" in selected:
        p.enable_hr = any(k in params for k in ("Hires upscale", "Hires upscaler", "Denoising strength"))
        for key, attribute, cast, low, high in (
            ("Hires upscale", "hr_scale", float, 1, 16),
            ("Hires steps", "hr_second_pass_steps", int, 0, 10000),
            ("Hires resize-1", "hr_resize_x", int, 0, 32768),
            ("Hires resize-2", "hr_resize_y", int, 0, 32768),
            ("Denoising strength", "denoising_strength", float, 0, 1),
            ("Hires CFG Scale", "hr_cfg", float, 0, 1000),
            ("Hires Distilled CFG Scale", "hr_distilled_cfg", float, 0, 1000),
        ):
            if key in params:
                setattr(p, attribute, number(params, key, cast, low, high))
        for key, attribute in (("Hires upscaler", "hr_upscaler"), ("Hires prompt", "hr_prompt"),
                               ("Hires negative prompt", "hr_negative_prompt"),
                               ("Hires sampler", "hr_sampler_name"), ("Hires schedule type", "hr_scheduler")):
            if key in params:
                value = str(params[key])
                setattr(p, attribute, None if value.startswith("Use same ") else value)
    p.extra_generation_params["Metadata Replay source"] = entry.name
    return p


def run_queue(base, queue, selected):
    from modules import errors, processing, shared

    queue.begin(int(base.n_iter))
    state = shared.state
    results = None
    seeds = {}
    aggregate = {key: [] for key in ("images", "all_prompts", "all_negative_prompts", "all_seeds", "all_subseeds", "infotexts", "extra_images")}
    failures = []

    def interrupted():
        return state.interrupted or state.stopping_generation

    try:
        while task := queue.take(interrupted):
            entry, batch = task
            p = None
            try:
                if batch == 0:
                    seeds[entry.id] = number(entry.parameters, "Seed", int, 0, 2**32 - 1) if (
                        "Original seed" in selected and "Seed" in entry.parameters
                    ) else secrets.randbelow(2**32)
                p = build_processing(base, entry, selected, seeds[entry.id], batch)
                # Recompute the remaining batch count after queue edits or navigation.
                with queue.condition:
                    position = next(i for i, item in enumerate(queue.entries) if item.id == entry.id)
                    remaining = (len(queue.entries) - position) * base.n_iter - batch
                    count = len(queue.entries)
                state.job_count = state.job_no + remaining * (2 if p.enable_hr else 1)
                state.job = f"Metadata {position + 1}/{count}: {entry.name} · batch {batch + 1}/{base.n_iter}"
                state.textinfo = state.job
                state.processing_has_refined_job_count = True
                results = processing.process_images(p)
                for key in aggregate:
                    aggregate[key].extend(getattr(results, key, []) or [])
                if interrupted():
                    break
                if state.skipped:
                    state.skipped = False
                    queue.command("skip")
                queue.complete_batch(entry.id, thumbnail(results.images[0]) if results.images else "")
            except Exception as exc:
                queue.fail(entry.id, exc)
                failures.append(f"{entry.name}: {exc}")
                errors.report(f"Metadata Replay: {entry.name}", exc_info=True)
            finally:
                if p is not None:
                    p.close()
    finally:
        queue.end(interrupted())
    if results is None:
        if failures:
            raise RuntimeError("No entries could be generated. " + "; ".join(failures))
        return processing.Processed(base, [], seed=base.seed, info="Metadata Replay stopped before generation.")
    for key, values in aggregate.items():
        setattr(results, key, values)
    if aggregate["all_seeds"]:
        results.seed = aggregate["all_seeds"][0]
    for field, plural in (("prompt", "all_prompts"), ("negative_prompt", "all_negative_prompts"), ("subseed", "all_subseeds")):
        if aggregate[plural]:
            setattr(results, field, aggregate[plural][0])
    results.info = aggregate["infotexts"][0] if aggregate["infotexts"] else ""
    results.index_of_first_image = 0
    if failures:
        results.comments += "\n" + "\n".join(failures)
    return results
