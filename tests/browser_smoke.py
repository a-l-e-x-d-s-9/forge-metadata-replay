"""Browser integration check; use a dedicated Forge data directory and port."""
import argparse
import json
import tempfile
import time
from pathlib import Path

from PIL import Image, PngImagePlugin
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:17866")
    parser.add_argument("--browser", default="/usr/bin/google-chrome")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output", default="/tmp/metadata-replay-browser-report.json")
    args = parser.parse_args()
    report = {"browser_errors": []}
    with tempfile.TemporaryDirectory(prefix="metadata-replay-fixtures-") as directory, sync_playwright() as pw:
        fixtures = Path(directory)
        paths = []
        for index, subject in enumerate(("red ceramic teapot", "blue ceramic teapot")):
            image = Image.new("RGB", (64, 64), (180, 60 + index * 90, 70))
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", f"a {subject} on a wooden table\nNegative prompt: blurry, noisy\nSteps: 4, Sampler: Euler, Schedule type: Simple, CFG scale: 5, Seed: {100 + index}, Size: 256x256, Clip skip: 2")
            path = fixtures / f"teapot-{index}.png"
            image.save(path, pnginfo=info)
            paths.append(str(path))
        browser = pw.chromium.launch(executable_path=args.browser, headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
        page.on("pageerror", lambda error: report["browser_errors"].append(str(error)))
        page.goto(args.base_url, wait_until="domcontentloaded")
        page.wait_for_selector("#script_list", timeout=90000)
        page.wait_for_timeout(4000)
        page.locator("#script_list input").first.fill("Metadata Replay")
        page.locator("#script_list input").first.press("Enter")
        page.wait_for_selector("#mr_images")
        print("Script selected", flush=True)

        def state():
            raw = page.locator("#mr_snapshot textarea").input_value()
            return json.loads(raw) if raw else {}

        def wait_for(predicate, seconds=30):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                result = predicate()
                if result:
                    return result
                page.wait_for_timeout(150)
            page.screenshot(path="/tmp/metadata-replay-browser-failure.png", full_page=True)
            raise AssertionError("Timed out waiting for browser check")

        def click(text):
            page.locator("#mr_app").get_by_role("button", name=text, exact=True).click()
            page.wait_for_timeout(450)

        wait_for(lambda: state().get("entries") == [])
        page.locator("#mr_images input[type=file]").set_input_files(paths)
        wait_for(lambda: len(state().get("entries", [])) == 2)
        assert state()["entries"][0]["parameters"]["Clip skip"] == "2"
        assert state()["entries"][0]["parameters"]["Prompt"] == "a red ceramic teapot on a wooden table"
        report["image_metadata"] = "passed"
        print("Image metadata passed", flush=True)
        page.locator("#mr_images input[type=file]").set_input_files(paths[0])
        wait_for(lambda: "deduplicated 1" in state().get("message", ""))
        assert len(state()["entries"]) == 2
        report["deduplication"] = "passed"
        print("Deduplication passed", flush=True)
        page.locator(".mr-entry").first.get_by_role("button", name="↓", exact=True).click()
        wait_for(lambda: state()["entries"][0]["name"] == "teapot-1.png")
        report["reorder"] = "passed"
        print("Reorder passed", flush=True)
        click("Filename A–Z")
        wait_for(lambda: state()["entries"][0]["name"] == "teapot-0.png")
        click("Filename Z–A")
        wait_for(lambda: state()["entries"][0]["name"] == "teapot-1.png")
        click("Reverse order")
        wait_for(lambda: state()["entries"][0]["name"] == "teapot-0.png")
        revision_before = state()["revision"]
        click("Shuffle")
        wait_for(lambda: state()["revision"] > revision_before)
        assert sorted(e["name"] for e in state()["entries"]) == ["teapot-0.png", "teapot-1.png"]
        first_id = state()["entries"][0]["id"]
        wait_for(lambda: page.locator(".mr-entry").first.get_attribute("data-id") == first_id)
        page.locator(".mr-entry").first.get_by_role("button", name="↓", exact=True).click()
        wait_for(lambda: state()["entries"][1]["id"] == first_id)
        click("Filename Z–A")
        wait_for(lambda: state()["entries"][0]["name"] == "teapot-1.png")
        report["sort_shuffle_and_subsequent_manual_reorder"] = "passed"
        print("Sort, reverse, shuffle, and manual reorder passed", flush=True)
        page.locator("#mr_export").click()
        page.wait_for_selector("#mr_download a[href]")
        href = page.locator("#mr_download a[href]").first.get_attribute("href")
        exported = page.request.get(href).json()
        assert "source_preview" not in exported["entries"][0]
        json_path = fixtures / "queue.json"
        json_path.write_text(json.dumps(exported))
        page.once("dialog", lambda dialog: dialog.accept())
        click("Clear queue")
        wait_for(lambda: len(state()["entries"]) == 0)
        page.locator("#mr_json_import input[type=file]").set_input_files(str(json_path))
        wait_for(lambda: len(state()["entries"]) == 2)
        assert not state()["entries"][0]["source_preview"]
        report["json_round_trip"] = "passed"
        print("JSON round-trip passed", flush=True)
        page.locator("#mr_images input[type=file]").set_input_files(paths)
        wait_for(lambda: all(e["source_preview"] for e in state()["entries"]))
        assert len(state()["entries"]) == 2
        report["json_image_merge"] = "passed"
        print("JSON/image merge passed", flush=True)
        click("Pause")
        wait_for(lambda: state()["paused"])
        click("Resume")
        wait_for(lambda: not state()["paused"])
        if args.generate:
            for label in ("Negative prompt", "Size", "Steps", "CFG scale", "Sampler / scheduler", "Clip skip"):
                page.locator("#mr_reuse").get_by_label(label, exact=True).check()
            def number(field, value):
                locator = page.locator(f"#txt2img_{field} input[type=number]")
                locator.fill(str(value))
                locator.press("Tab")
            for field, value in (("width", 256), ("height", 256), ("steps", 4), ("batch_count", 2), ("batch_size", 2)):
                number(field, value)
            click("Pause")
            page.locator("#txt2img_generate").click()
            wait_for(lambda: state()["active"] and state()["waiting"], 90)
            report["paused_generation"] = "passed"
            click("Resume")
            wait_for(lambda: not state()["active"] and all(e["status"] == "done" for e in state()["entries"]), 300)
            assert all(e["batches"] == 2 and e["generated_preview"] for e in state()["entries"])
            wait_for(lambda: bool(page.locator("#generation_info_txt2img textarea").input_value()), 30)
            info = json.loads(page.locator("#generation_info_txt2img textarea").input_value())
            assert len(info["all_seeds"]) == 8
            assert len(set(info["all_seeds"])) == 8
            assert info["all_prompts"] == ["a blue ceramic teapot on a wooden table"] * 4 + ["a red ceramic teapot on a wooden table"] * 4
            assert info["all_negative_prompts"] == ["blurry, noisy"] * 8
            assert all("Steps: 4" in text and "CFG scale: 5" in text and "Size: 256x256" in text for text in info["infotexts"])
            report.update(generated_images=8, generated_previews="passed", gallery_metadata="passed", optional_parameters="passed")
        page.screenshot(path="/tmp/metadata-replay-browser-success.png", full_page=True)
        browser.close()
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert not report["browser_errors"], report["browser_errors"]


if __name__ == "__main__":
    main()
