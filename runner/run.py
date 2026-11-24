"""Play variants against a model, turn by turn, and save the replies."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from contracts.io import append_jsonl, read_jsonl
from contracts.models import Response, ServerUnreachable


def play_variant(variant, chat, saved_replies=(), saved_cut_off=(), on_turn=None):
    """Send the turns one at a time and return (replies, cut_off).

    Every reply is added to the history before the next turn is sent.
    Saved replies from an earlier stopped run are put back into the history without asking the model again.
    """
    replies, cut_off = list(saved_replies), list(saved_cut_off)
    messages = []
    for i, turn in enumerate(variant.turns):
        messages.append({"role": "user", "content": turn.text})
        if i < len(replies):
            messages.append({"role": "assistant", "content": replies[i]})
            continue
        text, cut = chat(messages)    # every backend returns (reply text, cut off by the token limit)
        messages.append({"role": "assistant", "content": text})
        replies.append(text)
        cut_off.append(cut)
        if on_turn:
            on_turn(replies, cut_off)
    return replies, cut_off


def _partial_key(variant, model, settings):
    """The key for a half-finished conversation in partial.json.

    It includes the settings, so replies made with old settings are never mixed with new ones.
    """
    return f"{variant.conv_id}|{variant.condition}|{model}|{json.dumps(settings, sort_keys=True)}"


def _load_partial(partial_path, log=print):
    """Read partial.json. If it was cut short (the job was killed while writing it), keep it aside and start empty.

    Only half-finished conversations are lost that way, and they are simply asked again.
    """
    if not os.path.exists(partial_path):
        return {}
    try:
        with open(partial_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        broken = f"{partial_path}.broken-{time.strftime('%Y%m%d-%H%M%S')}"
        os.replace(partial_path, broken)
        log(f"partial.json could not be read (job killed while writing?). Moved to {broken}, starting it empty.")
        return {}


def _save_partial(partial_path, partial):
    """Write partial.json in one step: write a temporary file, then rename it, so a kill never leaves half a file."""
    tmp = partial_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False)
    os.replace(tmp, partial_path)


def _repair_responses(responses_path, log=print):
    """Drop a half-written last line of responses.jsonl, left when the job was killed while appending.

    That conversation is then not counted as finished, so it is run again.
    """
    if not os.path.exists(responses_path):
        return
    with open(responses_path, "rb") as f:
        data = f.read()
    if not data or data.endswith(b"\n"):
        return
    keep = data.rfind(b"\n") + 1
    with open(responses_path, "r+b") as f:
        f.truncate(keep)
    log(f"responses.jsonl ended with a half-written line ({len(data) - keep} bytes). Removed it; that conversation is redone.")


def _saved_responses(responses_path, model, settings):
    """The rows of responses.jsonl made by this model with these exact settings.

    A half-written last line (job killed while appending) is skipped here; run_all removes it.
    """
    if not os.path.exists(responses_path):
        return []
    rows = []
    with open(responses_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["model"] == model and row["settings"] == settings:
                rows.append(row)
    return rows


def _finished(responses_path, model, settings):
    """The (conv_id, condition) pairs already saved by this model with these exact settings."""
    return {(r["conv_id"], r["condition"]) for r in _saved_responses(responses_path, model, settings)}


def run_all(variants, chat, model, settings, out_folder, workers=4, log=print):
    """Run every variant that is not finished yet. Returns a summary dict.

    Files in out_folder:
      responses.jsonl  one finished Response per line
      partial.json     replies of conversations that stopped half way, so a rerun continues there
    """
    os.makedirs(out_folder, exist_ok=True)
    responses_path = os.path.join(out_folder, "responses.jsonl")
    partial_path = os.path.join(out_folder, "partial.json")
    _repair_responses(responses_path, log)
    partial = _load_partial(partial_path, log)
    lock = threading.Lock()

    done = _finished(responses_path, model, settings)
    todo = [v for v in variants if (v.conv_id, v.condition) not in done]
    summary = {"asked": len(variants), "already_done": len(variants) - len(todo),
               "finished": 0, "failed": [], "cut_off_replies": 0}
    stop = threading.Event()

    def work(variant):
        if stop.is_set():
            return variant, None
        key = _partial_key(variant, model, settings)
        saved = partial.get(key, {"replies": [], "cut_off": []})
        # A reply that was cut off is asked again, so keep only the turns before the first cut-off.
        keep = next((i for i, cut in enumerate(saved["cut_off"]) if cut), len(saved["replies"]))

        def save_progress(replies, cut_off):
            with lock:
                partial[key] = {"replies": list(replies), "cut_off": list(cut_off)}
                _save_partial(partial_path, partial)

        result = play_variant(variant, chat, saved["replies"][:keep], saved["cut_off"][:keep], save_progress)
        with lock:
            partial.pop(key, None)
            _save_partial(partial_path, partial)
        return variant, result

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, v): v for v in todo}
        for future in as_completed(futures):
            variant = futures[future]
            try:
                _, result = future.result()
            except ServerUnreachable as error:
                stop.set()    # no point trying the other conversations
                summary["failed"].append((variant.conv_id, variant.condition, f"cannot reach {error}"))
                continue
            except Exception as error:    # one bad conversation must not stop a long run
                summary["failed"].append((variant.conv_id, variant.condition, str(error)))
                log(f"failed: {variant.conv_id} {variant.condition}: {error}")
                continue
            if result is None:
                continue
            replies, cut_off = result
            # Only the main thread writes the responses file, so it needs no lock.
            append_jsonl(responses_path, Response(
                conv_id=variant.conv_id, condition=variant.condition, model=model, replies=replies,
                cut_off=cut_off, settings=settings, finished_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            summary["finished"] += 1
            summary["cut_off_replies"] += sum(cut_off)
            log(f"[{summary['finished']}/{len(todo)}] {variant.conv_id} {variant.condition}")
    return summary


def status(variants, model, settings, out_folder):
    """Count what is finished, half way and still to do, and check the finished replies.

    The reply checks are the ones PLAN.md asks for on the first finished conversations:
      empty_replies    replies with no text
      cut_off_replies  replies still cut off at the token cap (graded UNGRADED later)
      think_text       replies that contain <think> (hidden reasoning leaking into the answer)
    """
    wanted = {(v.conv_id, v.condition) for v in variants}
    rows = [r for r in _saved_responses(os.path.join(out_folder, "responses.jsonl"), model, settings)
            if (r["conv_id"], r["condition"]) in wanted]
    done = {(r["conv_id"], r["condition"]) for r in rows}
    partial = _load_partial(os.path.join(out_folder, "partial.json"), log=lambda _: None) \
        if os.path.exists(os.path.join(out_folder, "partial.json")) else {}
    keys = {_partial_key(v, model, settings) for v in variants}
    half_way = [k for k in partial if k in keys]
    replies = [text for r in rows for text in r["replies"]]
    return {"total": len(variants), "finished": len(done), "half_way": len(half_way),
            "to_do": len(wanted - done),
            "replies": len(replies),
            "empty_replies": sum(1 for text in replies if not text.strip()),
            "cut_off_replies": sum(sum(r["cut_off"]) for r in rows),
            "think_text": sum(1 for text in replies if "<think>" in text)}
