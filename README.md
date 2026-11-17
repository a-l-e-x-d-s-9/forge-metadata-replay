# Metadata Replay for Forge Neo

A selectable **txt2img → Script → Metadata Replay** script. Drop previously generated
images into a queue and generate new variations from their prompts and optional
settings. No img2img image conditioning is applied.

## Features

- Upload multiple images from your browser PC. Forge reads their metadata once
  using its PNG Info reader. The queue retains the extracted settings and a tiny
  source thumbnail; it does not require the uploaded originals for generation.
- Prompt reuse and fresh random seeds by default. Optionally reuse negative
  prompts, size, checkpoint, original seed, sampler/scheduler, steps, CFG,
  distilled CFG, clip skip, VAE/text encoders, and hires fix.
- **Remove LoRAs from prompts** is enabled by default. It strips `<lora:…>` tags
  from positive, negative, and hires generation prompts. Disable it to reuse
  those tags. Source metadata, editable prompts, and JSON exports retain them.
- The normal txt2img **Batch count × Batch size applies to every queue entry**.
  There are no separate batch fields in this script.
- Shuffle, sort filenames A–Z or Z–A (naturally, with 2 before 10), or reverse the
  current list. Drag rows or use ↑/↓ afterward to adjust the resulting order.
  Select **Start here** before Generate, or
  **Jump here** while running. Expand a row to inspect metadata and edit prompts
  while stopped.
- **Pause / Resume**, **Previous**, and **Skip** work at batch boundaries. Pause
  finishes the current batch and waits before the next; it holds Forge's active
  generation slot. Use the main **Interrupt** button to end the run and release
  that slot. Generate again starts the selected/interrupted entry at batch one.
- Portable JSON export/import, with merge or replacement. Deduplication uses the
  SHA-256 of the original file bytes, including across JSON imports and later
  image uploads. Different files with identical prompts remain separate entries.
- The first generated image supplies each row's small preview, with no extra
  generation. Generated thumbnails are included in JSON; full source images and
  source thumbnails are omitted. A JSON entry without a thumbnail receives one
  after it generates. Re-adding its source image restores the source thumbnail.
- Optional server folder import, including subfolders. This is a folder on the
  computer running Forge. It respects `--hide-ui-dir-config`.
- Results retain per-image prompts, seeds, and infotext in Forge's normal gallery.

## Install

Clone this repository into Forge Neo's `extensions` directory:

```sh
git clone https://github.com/a-l-e-x-d-s-9/forge-metadata-replay.git extensions/forge-metadata-replay
```

Restart Forge when idle and refresh your browser. No extra Python packages are
required. Use Forge Neo with Gradio 4.40; compatibility with other WebUI forks
has not been established.

## Use

1. Open txt2img and select **Metadata Replay** in the Script dropdown.
2. Drop source PNG/JPEG/WebP images in the upload area, or import an exported JSON.
   Images need metadata supported by Forge's PNG Info reader. Images without it
   are skipped with an explanation.
3. Select which settings to reuse. Missing source settings retain the current
   txt2img values. Prompt reuse clears selected txt2img styles because the source
   prompt already contains its style text. Uncheck **Remove LoRAs from prompts**
   if you want the source prompt's LoRA tags to remain active.
4. Arrange entries and set Batch count / Batch size in the normal txt2img UI.
5. Click **Generate**. Pause before navigating if you want to inspect the list;
   Previous, Skip, and Jump keep the paused state until Resume is clicked.
6. Export JSON before restarting Forge, refreshing the page, or closing the tab.

Unchecked settings use the txt2img controls, except seeds: unchecked **Original
seed** always chooses a fresh random starting seed for every visit to an entry,
then increments across that entry's batches. Checked Original seed uses the
source seed and increments in the same way. The normal seed field does not
override this policy.

Unavailable checkpoints, samplers, or modules mark the entry as an error and the
queue continues. Files are never downloaded automatically. Only the listed
parameter groups are applied; arbitrary extension settings in infotext are
preserved for inspection/export but are not restored into other extensions.
Installed always-on extensions still use their current txt2img settings.

Hires fix reuses its size/scale, upscaler, steps, denoise, prompts, sampler,
scheduler and guidance values. Hires-specific checkpoint and VAE/TE selections
remain those of the main interface. Forge model families may ignore unsupported
settings, such as clip skip on models without CLIP.

The queue lives in Forge's memory. Each page load gets its own session; use JSON
to retain queues between page loads. Inactive sessions expire after 24
hours without access. JSON preserves entry order and metadata, not an active run,
pause state, batch progress or the reuse checkboxes. Replacing, removing, or
editing entries requires stopping the run; merging and reordering remain
available during generation. A maximum of 2,000 entries is supported per queue.
Ordering controls also work during a run: the current batch keeps its entry,
and subsequent entries follow the new order after that entry. Pause first if
you want to arrange the queue and explicitly choose where to resume.

Uploaded originals use Gradio's normal temporary upload handling. JSON exports
contain prompts and source filenames, so share them accordingly.

## Development

```sh
python3 -m unittest discover -s tests -v
node --check javascript/metadata-replay.js
```

Tests cover queue round-trips, deduplication, reordering, live merge, pause,
interrupt, navigation, per-entry batch counts, seed behavior, failed models, and
per-image gallery metadata. `tests/browser_smoke.py` exercises a running isolated
Forge server; its default port is 17866 rather than the live UI.
