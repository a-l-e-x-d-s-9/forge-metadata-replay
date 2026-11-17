"""A selectable txt2img script; all queue controls bypass Gradio's generation queue."""
import json
import sys
import tempfile
import uuid
from pathlib import Path

import gradio as gr

from modules import scripts, shared

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metadata_replay.forge import CHOICES, IMAGE_SUFFIXES, load_image, run_queue
from metadata_replay.queue import decode_document, get_queue


def snapshot(queue):
    data = queue.snapshot()
    return json.dumps(data, ensure_ascii=False), data["revision"]


def message(queue, text):
    with queue.condition:
        queue.message = text
        queue.changed()


def poll(token, revision):
    if not token:
        return gr.skip(), gr.skip()
    queue = get_queue(token)
    with queue.condition:
        if queue.revision == revision:
            return gr.skip(), gr.skip()
        return snapshot(queue)


def upload_images(token, files):
    queue = get_queue(token)
    loaded, failures = [], []
    for path in files or []:
        try:
            loaded.append(load_image(path))
        except Exception as exc:
            failures.append(f"{Path(path).name}: {exc}")
    try:
        queue.merge(loaded)
        if failures:
            message(queue, queue.message + " Skipped: " + " | ".join(failures))
    except ValueError as exc:
        message(queue, str(exc))
    return None, *snapshot(queue)


def upload_json(token, file, mode):
    queue = get_queue(token)
    try:
        if file:
            queue.merge(decode_document(Path(file).read_bytes()), replace=mode == "Replace queue")
    except (ValueError, OSError) as exc:
        message(queue, f"Import failed: {exc}")
    return None, *snapshot(queue)


def folder_images(token, folder, recursive):
    if shared.cmd_opts.hide_ui_dir_config:
        raise gr.Error("Server-folder input is disabled by Forge's --hide-ui-dir-config.")
    path = Path(folder).expanduser()
    if not folder or not path.is_dir():
        raise gr.Error("Enter an existing folder on the Forge server.")
    iterator = path.rglob("*") if recursive else path.iterdir()
    paths = sorted(str(p) for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) > 2000:
        raise gr.Error("Folder contains more than 2000 images. Choose a smaller folder.")
    _, data, revision = upload_images(token, paths)
    return data, revision


def export_json(token):
    queue = get_queue(token)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="metadata-replay-", encoding="utf-8", delete=False) as file:
        json.dump(queue.document(), file, ensure_ascii=False, indent=2, allow_nan=False)
    return file.name


def command(token, value):
    queue = get_queue(token)
    try:
        args = json.loads(value)
        queue.command(**args)
    except (ValueError, TypeError, KeyError) as exc:
        message(queue, str(exc))
    return snapshot(queue)


class MetadataReplay(scripts.Script):
    def title(self):
        return "Metadata Replay"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        gr.Markdown("Drop source images, arrange their metadata, then use **txt2img Generate**. "
                    "Each entry uses the main **Batch count × Batch size**. Unchecked settings use the txt2img controls. "
                    "Seeds are fresh unless **Original seed** is checked.")
        token = gr.Textbox(value=lambda: uuid.uuid4().hex, elem_id="mr_token", elem_classes=["mr-internal"], label="Queue session")
        selected = gr.CheckboxGroup(choices=CHOICES, value=["Prompt"], label="Reuse from source metadata", elem_id="mr_reuse")
        files = gr.File(file_count="multiple", file_types=["image"], type="filepath", label="Drop source images here (metadata is retained after upload)", elem_id="mr_images")
        with gr.Row():
            json_file = gr.File(file_count="single", file_types=[".json"], type="filepath", label="Import queue JSON", elem_id="mr_json_import")
            with gr.Column():
                mode = gr.Radio(["Merge and deduplicate", "Replace queue"], value="Merge and deduplicate", label="JSON import mode")
                export = gr.Button("Export queue as JSON", elem_id="mr_export")
                download = gr.File(label="Download queue JSON", interactive=False, elem_id="mr_download")
        if not shared.cmd_opts.hide_ui_dir_config:
            with gr.Accordion("Import a folder on the Forge server", open=False):
                folder = gr.Textbox(label="Server folder (on the computer running Forge)")
                recursive = gr.Checkbox(label="Include subfolders", value=False)
                load_folder = gr.Button("Add folder images")
        gr.HTML('<div id="mr_app"><div class="mr-toolbar"></div><div class="mr-status" role="status" aria-live="polite"></div><div class="mr-list"></div></div>')
        data = gr.Textbox(value="", label="Queue snapshot", elem_id="mr_snapshot", elem_classes=["mr-internal"])
        revision = gr.Number(value=-1, precision=0, elem_classes=["mr-internal"])
        payload = gr.Textbox(value="", label="Queue command", elem_id="mr_command", elem_classes=["mr-internal"])
        send = gr.Button("Apply queue command", elem_id="mr_send", elem_classes=["mr-internal"])
        tick = gr.Button("Refresh queue", elem_id="mr_poll", elem_classes=["mr-internal"])
        events = dict(queue=False, show_progress="hidden")
        outputs = [data, revision]
        files.upload(upload_images, [token, files], [files, *outputs], **events)
        json_file.upload(upload_json, [token, json_file, mode], [json_file, *outputs], **events)
        export.click(export_json, [token], [download], **events)
        if not shared.cmd_opts.hide_ui_dir_config:
            load_folder.click(folder_images, [token, folder, recursive], outputs, **events)
        send.click(command, [token, payload], outputs, **events)
        tick.click(poll, [token, revision], outputs, **events)
        return [token, selected]

    def run(self, p, token, selected):
        return run_queue(p, get_queue(token), set(selected or []))
