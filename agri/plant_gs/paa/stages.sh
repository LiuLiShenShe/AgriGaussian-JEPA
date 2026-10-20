#!/usr/bin/env bash
# Stage helpers for run_paa_pilot.sh, kept in their own file so they can be
# sourced by the tests without running a training pipeline.
#
# The rule every helper here serves: **a stage's state is read from the
# artefacts the trainer wrote, never guessed from a file timestamp.** A
# directory is "complete" only when the trainer said so in status.json *and*
# wrote its terminal checkpoint. Everything else is either empty or partial,
# and partial is never silently overwritten.

# stage_state DIR -> prints "empty", "complete" or "partial"
stage_state() {
  local dir="$1"
  local status="$dir/status.json"
  if [ ! -d "$dir" ]; then
    echo empty
    return 0
  fi
  if [ -f "$dir/splats.pt" ] && [ -f "$status" ]; then
    if "$PY_GS" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="complete" else 1)' "$status" 2>/dev/null; then
      echo complete
      return 0
    fi
  fi
  if [ -e "$dir/config.json" ] || [ -e "$dir/history.jsonl" ] || \
     [ -e "$dir/splats.pt" ] || [ -e "$dir/splats_best.pt" ] || \
     [ -n "$(ls -A "$dir" 2>/dev/null | grep -E '\.pt$|\.jsonl$|\.json$' || true)" ]; then
    echo partial
    return 0
  fi
  echo empty
}

# gate_check VALIDATION_JSON MIN_PSNR MIN_IOU MAX_BG_ALPHA
# -> exits 0 when the gate passes, 1 when it fails, 2 when it cannot be read.
gate_check() {
  local validation="$1" min_psnr="$2" min_iou="$3" max_bg="$4"
  [ -f "$validation" ] || { echo "GATE_ERROR no $validation" >&2; return 2; }
  "$PY_GS" - "$validation" "$min_psnr" "$min_iou" "$max_bg" <<'PY'
import json, sys
path, min_psnr, min_iou, max_bg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
try:
    m = json.load(open(path))
except Exception as exc:
    print(f"GATE_ERROR unreadable {path}: {exc}")
    sys.exit(2)
try:
    psnr, iou, bg = m["foreground_psnr"], m["silhouette_iou"], m["background_alpha"]
except KeyError as exc:
    print(f"GATE_ERROR {path} is missing {exc}")
    sys.exit(2)
problems = []
if psnr < min_psnr: problems.append(f"foreground_psnr {psnr:.3f} < {min_psnr}")
if iou  < min_iou:  problems.append(f"silhouette_iou {iou:.3f} < {min_iou}")
if bg   > max_bg:   problems.append(f"background_alpha {bg:.4f} > {max_bg}")
print(f"GATE psnr={psnr:.3f} iou={iou:.4f} bg_alpha={bg:.4f}")
if problems:
    print("GATE_FAILED " + "; ".join(problems))
    sys.exit(1)
print("GATE_PASSED")
PY
}

# stage_manifest DIR STAGE STEPS "COMMAND"
# Records which code and which exact command produced a stage.
stage_manifest() {
  local dir="$1" name="$2" steps="$3" command="$4"
  "$PY_GS" - "$dir" "$name" "$steps" "$command" "$REPO" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
dir_, name, steps, command, repo = sys.argv[1:6]
repo = Path(repo)
def git(*args):
    try:
        out = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None
payload = {
    "stage": name,
    "steps": int(steps),
    "command": command,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
    "pid": os.getpid(),
}
target = Path(dir_) / "stage_manifest.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

# --------------------------------------------------------------------------- #
# Progress and failure reporting
# --------------------------------------------------------------------------- #
step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$1"; }
die()  { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# run_stage NAME STEPS EVAL_EVERY SAVE_EVERY
#
# One training stage, with its own directory, log, status and checkpoints.
# A completed stage is skipped, never re-run. A partial stage is refused unless
# RESUME=1, and even then the trainer only continues it when the stage's own
# resumable checkpoint carries the state that makes a continuation honest.
# --------------------------------------------------------------------------- #
run_stage() {
  local name="$1" steps="$2" eval_every="$3" save_every="$4"
  local dir="$WS/results/$name"
  local log="$WS/logs/03_${name}.log"
  local state
  state="$(stage_state "$dir")"

  case "$state" in
    complete)
      echo "$name already complete; skipping (no overwrite)"
      return 0 ;;
    partial)
      [ "$RESUME" = "1" ] || die "$dir holds an INCOMPLETE run. Refusing to start a second run over it. Re-run with RESUME=1 to continue it from its own splats_resume.pt, or use a new workspace."
      echo "$name is incomplete; RESUME=1 -> continuing from its resumable checkpoint"
      ;;
    empty)
      ;;
  esac

  local extra=()
  [ "$state" = "partial" ] && extra+=(--resume)

  step "train $name ($steps steps)"
  local cmd="$PY_GS $FG/train.py --data $WS/dataset --result $dir --steps $steps --eval-every $eval_every --save-every $save_every --downscale 1 --sh-degree 3 --foreground-alpha-weight 0.1 --background-alpha-weight 0.75 --max-gaussians 300000 --seed $SEED ${extra[*]:-}"
  "$PY_GS" -u "$FG/train.py" \
    --data "$WS/dataset" --result "$dir" \
    --steps "$steps" --eval-every "$eval_every" --save-every "$save_every" \
    --downscale 1 --sh-degree 3 \
    --foreground-alpha-weight 0.1 --background-alpha-weight 0.75 \
    --max-gaussians 300000 --seed "$SEED" \
    "${extra[@]}" 2>&1 | tee "$log"

  [ "$(stage_state "$dir")" = "complete" ] || \
    die "$name did not reach 'complete' (see $log)"
  stage_manifest "$dir" "$name" "$steps" "$cmd"
}
