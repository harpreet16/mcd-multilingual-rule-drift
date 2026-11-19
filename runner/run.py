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


def _finished(responses_path, model, settings):
    """The (conv_id, condition) pairs already saved by this model with these exact settings."""
    if not os.path.exists(responses_path):
        return set()
    return {(r["conv_id"], r["condition"]) for r in read_jsonl(responses_path)
            if r["model"] == model and r["settings"] == settings}


def run_all(variants, chat, model, settings, out_folder, workers=4, log=print):
    """Run every variant that is not finished yet. Returns a summary dict.

    Files in out_folder:
      responses.jsonl  one finished Response per line
      partial.json     replies of conversations that stopped half way, so a rerun continues there
    """
    os.makedirs(out_folder, exist_ok=True)
    responses_path = os.path.join(out_folder, "responses.jsonl")
    partial_path = os.path.join(out_folder, "partial.json")
    partial = json.load(open(partial_path, encoding="utf-8")) if os.path.exists(partial_path) else {}
    lock = threading.Lock()

    done = _finished(responses_path, model, settings)
    todo = [v for v in variants if (v.conv_id, v.condition) not in done]
    summary = {"asked": len(variants), "already_done": len(variants) - len(todo),
               "finished": 0, "failed": [], "cut_off_replies": 0}
    stop = threading.Event()

    def work(variant):
        if stop.is_set():
            return variant, None
        key = f"{variant.conv_id}|{variant.condition}|{model}"
        saved = partial.get(key, {"replies": [], "cut_off": []})
        # A reply that was cut off is asked again, so keep only the turns before the first cut-off.
        keep = next((i for i, cut in enumerate(saved["cut_off"]) if cut), len(saved["replies"]))

        def save_progress(replies, cut_off):
            with lock:
                partial[key] = {"replies": list(replies), "cut_off": list(cut_off)}
                with open(partial_path, "w", encoding="utf-8") as f:
                    json.dump(partial, f, ensure_ascii=False)

        result = play_variant(variant, chat, saved["replies"][:keep], saved["cut_off"][:keep], save_progress)
        with lock:
            partial.pop(key, None)
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, ensure_ascii=False)
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
    """Count what is finished, half way, and still to do."""
    done = _finished(os.path.join(out_folder, "responses.jsonl"), model, settings)
    partial_path = os.path.join(out_folder, "partial.json")
    partial = json.load(open(partial_path, encoding="utf-8")) if os.path.exists(partial_path) else {}
    half_way = [k for k in partial if k.endswith(f"|{model}")]
    return {"total": len(variants), "finished": len(done), "half_way": len(half_way),
            "to_do": len(variants) - len(done)}
