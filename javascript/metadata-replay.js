/* Native Gradio uploads/callbacks keep Forge's existing authentication and URL prefix. */
(() => {
    "use strict";
    let current = null;
    let lastSnapshot = "";
    let dragging = null;
    let busyUntil = 0;
    const commands = [];
    let commandTimer = null;
    const root = () => typeof gradioApp === "function" ? gradioApp() : document;
    const field = id => root().querySelector(`#${id} textarea, #${id} input`);
    function setField(id, value) {
        const target = field(id);
        if (!target) return false;
        // Respect the component's own value setter, as Forge's UI helpers do.
        target.value = value;
        // Forge's updateInput pins the event target across Gradio's shadow root.
        if (typeof updateInput === "function") updateInput(target);
        else {
            const event = new Event("input", {bubbles: true});
            Object.defineProperty(event, "target", {value: target});
            target.dispatchEvent(event);
        }
        return true;
    }
    function send(action, args = {}) {
        commands.push({action, ...args});
        drainCommands();
    }
    function drainCommands() {
        if (commandTimer || !commands.length) return;
        if (Date.now() < busyUntil) {
            commandTimer = setTimeout(() => { commandTimer = null; drainCommands(); }, busyUntil - Date.now() + 10);
            return;
        }
        // Capture each payload separately, retaining rapid consecutive clicks.
        busyUntil = Date.now() + 350;
        setField("mr_command", JSON.stringify(commands.shift()));
        root().querySelector("#mr_send")?.click();
        if (commands.length) drainCommands();
    }
    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }
    function button(text, action, title = "") {
        const node = element("button", "mr-button", text);
        node.type = "button";
        node.title = title;
        node.addEventListener("click", action);
        return node;
    }
    function render(data) {
        const app = root().querySelector("#mr_app");
        if (!app) return;
        current = data;
        const toolbar = app.querySelector(".mr-toolbar");
        const clear = button("Clear queue", () => {
            if (window.confirm("Remove all metadata entries from this queue?")) send("clear");
        });
        clear.disabled = data.active;
        toolbar.replaceChildren(
            button(data.paused ? "Resume" : "Pause", () => send(data.paused ? "resume" : "pause")),
            button("← Previous", () => send("previous"), "Repeat the previous entry after this batch"),
            button("Skip →", () => send("skip"), "Skip the remaining batches of this entry"),
            clear,
            button("Shuffle", () => send("order", {kind: "shuffle"}), "Randomize the generation order"),
            button("Filename A–Z", () => send("order", {kind: "filename"}), "Sort naturally by filename, 2 before 10"),
            button("Filename Z–A", () => send("order", {kind: "filename_reverse"}), "Sort by filename in reverse"),
            button("Reverse order", () => send("order", {kind: "reverse"}), "Reverse the current list")
        );
        const state = data.waiting ? "Paused" : data.paused && data.active ? "Pausing after this batch" : data.active ? "Generating" : data.paused ? "Paused before start" : "Ready";
        app.querySelector(".mr-status").textContent = `${state} · ${data.entries.length} entries. ${data.message}`;
        // Preserve open details and an unsaved prompt edit across progress refreshes.
        const list = app.querySelector(".mr-list");
        const expanded = new Set([...list.querySelectorAll("details[open]")].map(n => n.dataset.id));
        const focused = document.activeElement;
        const editing = focused && list.contains(focused) && focused.tagName === "TEXTAREA";
        if (editing) return;
        const fragment = document.createDocumentFragment();
        for (const [index, entry] of data.entries.entries()) {
            const row = element("div", "mr-entry");
            row.dataset.id = entry.id;
            row.dataset.status = entry.status;
            row.draggable = true;
            if (entry.id === data.current && data.active) row.classList.add("mr-current");
            if (entry.id === data.next) row.classList.add("mr-next");
            row.addEventListener("dragstart", event => {
                if (["TEXTAREA", "INPUT", "BUTTON"].includes(event.target.tagName)) {
                    event.preventDefault();
                    return;
                }
                dragging = entry.id;
                event.dataTransfer.setData("text/plain", entry.id);
                event.dataTransfer.effectAllowed = "move";
            });
            row.addEventListener("dragover", event => {
                if (dragging) { event.preventDefault(); row.classList.add("mr-drop-target"); }
            });
            row.addEventListener("dragleave", () => row.classList.remove("mr-drop-target"));
            row.addEventListener("drop", event => {
                if (!dragging) return; // Image files are handled by the upload area.
                event.preventDefault();
                row.classList.remove("mr-drop-target");
                send("move", {key: dragging, index});
                dragging = null;
            });
            row.addEventListener("dragend", () => { dragging = null; });
            const thumb = element("div", "mr-thumb");
            if (entry.generated_preview || entry.source_preview) {
                const img = element("img");
                img.src = entry.generated_preview || entry.source_preview;
                img.alt = entry.generated_preview ? "First generated image" : "Source image";
                img.loading = "lazy";
                img.draggable = false;
                thumb.append(img);
            } else thumb.textContent = "Preview after generation";
            row.append(thumb);
            const content = element("div", "mr-content");
            content.append(element("strong", "mr-name", `${index + 1}. ${entry.name}`));
            const label = entry.id === data.current && data.active ? `${data.waiting ? "paused" : "generating"} · ${entry.batches}/${data.batch_count} batches` : `${entry.status}${entry.batches ? ` · ${entry.batches} batches` : ""}`;
            content.append(element("span", "mr-entry-status", label));
            content.append(element("p", "mr-prompt", entry.parameters.Prompt || "(empty prompt)"));
            if (entry.error) content.append(element("p", "mr-error", entry.error));
            const details = element("details", "mr-details");
            details.dataset.id = entry.id;
            details.open = expanded.has(entry.id);
            details.append(element("summary", "", "Metadata / edit prompts"));
            details.append(element("pre", "mr-parameters", JSON.stringify(entry.parameters, null, 2)));
            if (!data.active) {
                const positive = element("textarea", "mr-edit");
                positive.value = entry.parameters.Prompt || "";
                positive.setAttribute("aria-label", "Prompt");
                const negative = element("textarea", "mr-edit");
                negative.value = entry.parameters["Negative prompt"] || "";
                negative.setAttribute("aria-label", "Negative prompt");
                const save = button("Save prompts", () => send("edit", {key: entry.id, "Prompt": positive.value, "Negative prompt": negative.value}));
                details.append(positive, negative, save);
            }
            content.append(details);
            row.append(content);
            const controls = element("div", "mr-entry-controls");
            controls.append(
                button(data.active ? "Jump here" : "Start here", () => send("jump", {key: entry.id}), "Select this entry; resume separately if paused"),
                button("↑", () => send("move", {key: entry.id, index: index - 1}), "Move up"),
                button("↓", () => send("move", {key: entry.id, index: index + 1}), "Move down")
            );
            const remove = button("Remove", () => send("remove", {key: entry.id}));
            remove.disabled = data.active;
            controls.append(remove);
            row.append(controls);
            fragment.append(row);
        }
        if (!data.entries.length) fragment.append(element("p", "mr-empty", "Your generation queue is empty. Drop images above or import a saved JSON."));
        list.replaceChildren(fragment);
    }
    function tick() {
        const token = field("mr_token");
        if (!token) return;
        if (!token.value) return;
        const snapshot = field("mr_snapshot")?.value;
        if (snapshot && snapshot !== lastSnapshot) {
            try { render(JSON.parse(snapshot)); lastSnapshot = snapshot; }
            catch (error) { console.error("Metadata Replay:", error); }
        }
        // Skip polling when the script is hidden. Pause/stop still work through Forge.
        const app = root().querySelector("#mr_app");
        if (app?.getClientRects().length && !dragging && Date.now() >= busyUntil) {
            root().querySelector("#mr_poll")?.click();
        }
    }
    setInterval(tick, 1000);
    if (typeof onUiLoaded === "function") onUiLoaded(tick);
})();
