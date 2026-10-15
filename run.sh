#!/usr/bin/env bash
# Crosswalk Traffic Analytics Pipeline — exported from PipeGen 0.5.1 on 2026-10-15T14:45:44Z.
#
#   ./run.sh                          run it
#   ./run.sh --help                   every flag
#
# Requirements on this machine: an NVIDIA GPU, Docker, and the NVIDIA container
# toolkit. Nothing else — no PipeGen, no Python, no CUDA/TensorRT/DeepStream.
#
# HOW IT WORKS
#   This script runs twice. The first time on your machine: it checks the
#   prerequisites, picks the container image for this CPU architecture, pulls it
#   if needed, and starts it with THIS FOLDER bind-mounted at its own path. The
#   second time inside that container (PG_IN_CONTAINER=1): it builds the parser
#   libraries from source, resolves the inference engine against the machine it
#   is actually on, and launches the pipeline.
#
#   Nothing here is tied to where the folder lives. Move it, rename it, hand it
#   to someone else — it locates itself from $BASH_SOURCE every time.

# `sh run.sh` is a reflex for a lot of people, and on most Linuxes /bin/sh is
# dash. Every line below is bash — `set -o pipefail`, arrays, ${BASH_SOURCE} —
# so without this the very next line dies with dash's "set: Illegal option -o
# pipefail", which names neither bash nor the fix. This test is plain POSIX and
# survives any shell long enough to say what to do.
if [ -z "${BASH_VERSION:-}" ]; then
  echo "run.sh needs bash: use ./run.sh or 'bash run.sh' (not 'sh run.sh')." >&2
  exit 1
fi
set -euo pipefail

# ── Where we are ────────────────────────────────────────────────────────────
# `pwd -P` resolves symlinks in the DIRECTORY it lands in — but the directory it
# lands in is the one holding ${BASH_SOURCE[0]}, and that may itself be a
# symlink: putting a launcher on $PATH (`ln -s …/run.sh ~/bin/pose-run`) is an
# ordinary thing to do with a folder whose README says "./run.sh is the whole
# interface". Following the link chain first is what makes PG_HERE the EXPORT's
# directory rather than the launcher's; without it run.sh looks for
# pipeline/pipeline.sh in the wrong place and reports the export incomplete.
# The result has to be a real path either way: the container half re-derives
# PG_ROOT from the same expression, and the folder is bind-mounted at exactly
# this string, so host and container agree on every path in the pipeline.
_pg_src="${BASH_SOURCE[0]}"
_pg_hops=0
while [ -L "$_pg_src" ] && [ "$_pg_hops" -lt 40 ]; do
  _pg_dir="$(cd -- "$(dirname -- "$_pg_src")" && pwd -P)"
  _pg_src="$(readlink -- "$_pg_src")"
  case "$_pg_src" in /*) ;; *) _pg_src="$_pg_dir/$_pg_src" ;; esac
  _pg_hops=$((_pg_hops + 1))
done
PG_HERE="$(cd -- "$(dirname -- "$_pg_src")" && pwd -P)"
unset _pg_src _pg_dir _pg_hops

PG_PROJECT_NAME="Crosswalk Traffic Analytics Pipeline"
PG_MANIFEST="$PG_HERE/manifest.json"
PG_PIPELINE_SH="$PG_HERE/pipeline/pipeline.sh"
PG_FIRST_RUN_SH="$PG_HERE/pipeline/first-run.sh"
PG_STAMP="$PG_HERE/.pipegen-state/first-run.stamp"
PG_XAUTH="$PG_HERE/.pipegen-state/xauth"
PG_DEFAULT_OUT="$PG_HERE/out"
PG_MIN_DRIVER="560.35.03"
PG_IMAGE_REPO="craftifaidocker/deepstream"

# ── Output helpers ──────────────────────────────────────────────────────────
_c_green=''; _c_red=''; _c_yellow=''; _c_bold=''; _c_reset=''
if [ -t 1 ]; then
  _c_green=$'\033[32m'; _c_red=$'\033[31m'; _c_yellow=$'\033[33m'
  _c_bold=$'\033[1m';   _c_reset=$'\033[0m'
fi
PG_FAILED=0
ok()   { printf '  %s\xe2\x9c\x93%s %s\n' "$_c_green"  "$_c_reset" "$1"; }
note() { printf '  %s-%s %s\n'            "$_c_yellow" "$_c_reset" "$1"; }
fail() { printf '  %s\xe2\x9c\x98%s %s\n' "$_c_red"    "$_c_reset" "$1"; PG_FAILED=1; }
say()  { printf '%s[run.sh]%s %s\n' "$_c_bold" "$_c_reset" "$1"; }
die()  { printf '\n%s%serror:%s %s\n' "$_c_bold" "$_c_red" "$_c_reset" "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

  --input <src>     What to read from. One of:
                      a video file        ./run.sh --input clip.mp4
                      an RTSP stream      ./run.sh --input rtsp://cam/live
                      a camera            ./run.sh --input /dev/video0
                    Default: whatever the pipeline was built against.
  --output <dir>    Where results land. Default: ./out
  --headless        Run without a window (results still go to --output).
  --display         Force a window. Fails if $DISPLAY is unset or unreachable.
                    Default: a window when a display answers, headless otherwise.
  --rebuild         Redo the first-run work (parser build + engine resolution)
                    even if it was already done for this machine.
  --help            This text.

Environment (all optional, all passed through to the pipeline):
  CAM CAM_W CAM_H CAM_FPS CAM_FMT CAM_PROBE
                    Camera device and capture mode, when the automatic v4l2
                    probe picks one your camera cannot deliver. CAM_PROBE=0
                    turns the probe off and uses the values as given.
  BITRATE FORCE_SW_ENC
                    Recording bitrate in kbit/s, and FORCE_SW_ENC=1 to use the
                    software H.264 encoder.
  JSON_OUT JSON_MIN_CONF JSON_MASK_OUTLINE
                    Where the metadata JSON goes, and what goes in it.
  MQTT_HOST MQTT_PORT MQTT_TOPIC
                    Where to publish metadata, if this pipeline publishes.
  PG_GPU            Which GPU to use on a multi-GPU machine: an index (PG_GPU=1)
                    or a UUID (PG_GPU=GPU-xxxxxxxx). Default: all of them.
  PG_IMAGE          Override the container image tag.

Everything runs inside a container. The first run builds the parser libraries
and resolves the inference engine for THIS machine, which takes about a minute;
later runs skip straight to the pipeline. One run of a folder at a time — a
second one would overwrite the first one's results, so it is refused.
EOF
}

# ── Reading manifest.json without depending on anything ─────────────────────
# python3 is guaranteed inside the container and merely likely on the host, so
# on the host an unreadable manifest degrades to "skip the early check" and the
# container half — which can always read it — is the authority.
# shellcheck disable=SC2016
_PG_PY_GET='
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
cur = doc
for key in sys.argv[2].split("."):
    if isinstance(cur, dict) and key in cur:
        cur = cur[key]
    else:
        sys.exit(0)
if isinstance(cur, bool):
    print("1" if cur else "0")
elif isinstance(cur, (list, tuple)):
    print(" ".join(str(x) for x in cur))
elif cur is not None:
    print(cur)
'

# Root-relative path of every engine the manifest declares, one per line. Used
# by the container half to check that the products of a previous first-run are
# still on disk. Container-only, so python3 is guaranteed here.
# shellcheck disable=SC2016
_PG_PY_PLANS='
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for m in doc.get("models") or []:
    p = m.get("plan")
    if p:
        print(p)
'

# Third choice, and the one that removes the dependency: awk. Without it, a
# host with docker and the NVIDIA toolkit but no python3 and no jq silently
# loses EVERY manifest-driven guardrail — no "you must say what to read", no
# "that camera does not exist here", no "this pipeline cannot take a file" —
# and the recipient gets a GStreamer failure, or a silent capture from the
# wrong camera, instead of one sentence naming the flag that fixes it.
# Measured on a PATH with python3/jq removed: the pipeline started capturing
# from this machine's real webcam and ran until it was killed.
#
# This is not a JSON parser and does not pretend to be one. manifest.json is
# written by `json.dumps(indent=2)`, so nesting is exactly two spaces per level
# and one key per line, which is all the scalar and string-array fields these
# scripts read actually need. Anything it cannot make sense of comes back
# empty, exactly like a missing interpreter — and the caller then says so.
# shellcheck disable=SC2016
_PG_AWK_GET='
/^[ ]*$/ { next }
{
  ind = match($0, /[^ ]/) - 1
  s = substr($0, ind + 1)
  sub(/[ \t\r]+$/, "", s)
  sub(/,$/, "", s)
  d = int(ind / 2)
  if (collecting) {
    if (s == "]") { print out; exit }
    # A list of objects (models[]) is not a value any caller can use, and a
    # half-flattened one would be worse than none: give up and say nothing.
    if (s !~ /^(".*"|-?[0-9][0-9.eE+-]*|true|false|null)$/) exit
    v = s
    if (v ~ /^".*"$/) v = substr(v, 2, length(v) - 2)
    out = (out == "" ? v : out " " v)
    next
  }
  if (s !~ /^"[^"]*"[ ]*:/) next
  key = s; sub(/^"/, "", key); sub(/"[ ]*:.*$/, "", key)
  val = s; sub(/^"[^"]*"[ ]*:[ ]*/, "", val)
  stack[d] = key
  p = ""
  for (i = 1; i <= d; i++) p = (p == "" ? stack[i] : p "." stack[i])
  if (p != q) next
  if (val == "[")     { collecting = 1; out = ""; next }
  if (val == "[]" || val == "{}" || val == "{" || val == "null") exit
  if (val == "true")  { print "1"; exit }
  if (val == "false") { print "0"; exit }
  if (val ~ /^".*"$/) val = substr(val, 2, length(val) - 2)
  print val
  exit
}
'

# Which of the three we have. Decided once, at load, because manifest_get runs
# in `$(...)` — a subshell, where it could not record the answer for anyone.
if   [ ! -f "$PG_MANIFEST" ];              then PG_MANIFEST_READER="none"
elif command -v python3 >/dev/null 2>&1;   then PG_MANIFEST_READER="python3"
elif command -v jq      >/dev/null 2>&1;   then PG_MANIFEST_READER="jq"
elif command -v awk     >/dev/null 2>&1;   then PG_MANIFEST_READER="awk"
else                                            PG_MANIFEST_READER="none"
fi

manifest_get() {   # manifest_get <dotted.path> -> value, or "" when unknown
  case "$PG_MANIFEST_READER" in
    python3) python3 -c "$_PG_PY_GET" "$PG_MANIFEST" "$1" 2>/dev/null || true ;;
    jq)      jq -r --arg q "$1" '
               (getpath($q | split("."))? // empty)
               | if type == "array" then join(" ")
                 elif type == "boolean" then (if . then "1" else "0" end)
                 else tostring end' "$PG_MANIFEST" 2>/dev/null || true ;;
    awk)     awk -v q="$1" "$_PG_AWK_GET" "$PG_MANIFEST" 2>/dev/null || true ;;
    *)       : ;;
  esac
}

# ── Small utilities ─────────────────────────────────────────────────────────
# `realpath` is not on every host, and `readlink -f` differs across platforms.
# This needs only the PARENT to exist, which is what both --input (file must
# exist) and --output (directory we are about to create) actually require.
abspath() {
  local p="$1" d b
  d="$(dirname -- "$p")"
  b="$(basename -- "$p")"
  d="$(cd -- "$d" 2>/dev/null && pwd -P)" || return 1
  case "$d" in
    /) printf '/%s\n' "$b" ;;
    *) printf '%s/%s\n' "$d" "$b" ;;
  esac
}

# abspath, plus the FINAL component's symlink chain. abspath deliberately stops
# short of that — its --output caller names a path that need not exist yet — but
# --input must go the whole way. run.sh bind-mounts the directory holding the
# input file, and for a symlink that is the LINK's directory, not the target's:
# the link is then dangling inside the container while the host-side `[ -f ]`,
# which follows it, saw nothing wrong. Measured: an ordinary
# `~/Videos/latest.mp4 -> …` style link produced "GStreamer error: state change
# failed" at preroll, exit 255, and a zero-byte output.mp4 that looks like a
# result. `readlink -f` would do this in one call but is not portable; this is.
pg_realpath() {
  local p d n=0
  p="$(abspath "$1")" || return 1
  while [ -L "$p" ] && [ "$n" -lt 40 ]; do
    d="$(dirname -- "$p")"
    p="$(readlink -- "$p")" || return 1
    case "$p" in /*) ;; *) p="$d/$p" ;; esac
    p="$(abspath "$p")" || return 1
    n=$((n + 1))
  done
  printf '%s\n' "$p"
}

# Lowercase, non-alphanumerics collapsed to "-" — the same slug PipeGen's own
# engine builder uses for its timing cache (engine_builder.py:813-819), so the
# names an export leaves behind read the same as the ones it came with.
pg_slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' \
                   | sed -e 's/[^a-z0-9]\{1,\}/-/g' -e 's/^-//' -e 's/-$//'
}

# A glob rather than `ls /dev/video*`: with no camera at all the glob is
# literal, so the loop has to test each candidate for existence.
pg_cameras() {
  local dev found=""
  for dev in /dev/video*; do
    if [ -e "$dev" ]; then found="$found $dev"; fi
  done
  if [ -n "$found" ]; then printf '%s' "${found# }"; else printf 'none'; fi
}

# Mount bookkeeping. `docker run` refuses the same target twice, and a
# directory mounted read-only shadows anything inside it that a later mount
# would have made writable — so every candidate is checked against what is
# already there instead of being appended blindly.
PG_MOUNTS=()
PG_MOUNT_DIRS=()
pg_add_dir_mount() {   # pg_add_dir_mount <host-dir> [ro]
  local d="$1" m
  for m in ${PG_MOUNT_DIRS[@]+"${PG_MOUNT_DIRS[@]}"}; do
    case "$d" in "$m"|"$m"/*) return 0 ;; esac
  done
  PG_MOUNT_DIRS+=("$d")
  if [ "${2:-}" = "ro" ]; then
    PG_MOUNTS+=(-v "$d:$d:ro")
  else
    PG_MOUNTS+=(-v "$d:$d")
  fi
}

pg_gpu_name() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true
}

pg_trt_version() {
  command -v python3 >/dev/null 2>&1 || return 0
  python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true
}

# $DISPLAY being SET is not the same as a display being REACHABLE, and the
# difference is the whole default path: a leftover `export DISPLAY` in .bashrc,
# a tmux or screen session, `sudo -E`, or an ssh session on a box whose $DISPLAY
# points at a console X server the container cannot reach — all of them set the
# variable and none of them can show a window. Trusting it turns `./run.sh
# --input clip.mp4` into "Failed to set pipeline to PAUSED", exit 255, and a log
# in which the word "display" appears exactly twice, both times as a green tick.
# So probe, and let the caller fall back to headless — which is what the design
# means by "no display -> headless".
#   0 = reachable    1 = definitely not reachable    2 = cannot tell from here
pg_display_probe() {
  [ -n "${DISPLAY:-}" ] || return 1
  # A dead TCP $DISPLAY can hang a client for the full connect timeout, and this
  # runs before anything the recipient asked for, so cap it when we can.
  local t=()
  command -v timeout >/dev/null 2>&1 && t=(timeout 5)
  if command -v xdpyinfo >/dev/null 2>&1; then
    "${t[@]+"${t[@]}"}" xdpyinfo >/dev/null 2>&1 && return 0
    return 1
  fi
  if command -v xset >/dev/null 2>&1; then
    "${t[@]+"${t[@]}"}" xset q >/dev/null 2>&1 && return 0
    return 1
  fi
  # No X client tools here. A ":N" display is a unix socket we can look for
  # ourselves; a "host:N" one we genuinely cannot judge, and refusing a display
  # that may well work would be worse than trying it.
  case "$DISPLAY" in
    :*|unix:*)
      local n="${DISPLAY#unix}"; n="${n#:}"; n="${n%%.*}"
      [ -S "/tmp/.X11-unix/X$n" ] && return 0
      return 1 ;;
    *) return 2 ;;
  esac
}

# 0 = reachable, 1 = not, 2 = unknown. Filled once by host_main before the
# checklist prints, and read again when the display decision is made, so the
# tick the recipient sees and the mode they get are the same judgement.
PG_DISPLAY_STATE=1

# What display mode did to this machine that has to be undone. A downloaded
# artifact must not leave the host's X access control weaker than it found it,
# and must not leave a cookie for the recipient's screen lying in a folder.
PG_XAUTH_MADE=0
PG_XHOST_MADE=0
pg_display_cleanup() {
  if [ "$PG_XAUTH_MADE" = "1" ]; then rm -f "$PG_XAUTH"; PG_XAUTH_MADE=0; fi
  if [ "$PG_XHOST_MADE" = "1" ]; then
    xhost -local:root >/dev/null 2>&1 || true
    PG_XHOST_MADE=0
  fi
  return 0
}


# ═══════════════════════════════════════════════════════════════════════════
# HOST PHASE
# ═══════════════════════════════════════════════════════════════════════════

# The checklist. Shape, wording and the Tegra branch follow PipeGen's own
# launcher (packaging/run.src.sh:30-116); the two checks that only make sense
# for the PipeGen application itself are dropped, and DISPLAY is a note rather
# than a failure because this export runs headless by design without one.
host_checklist() {
  printf '%sPrerequisite checklist%s\n' "$_c_bold" "$_c_reset"

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
    ok "NVIDIA GPU detected"
    local driver
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    # Jetson/Tegra numbers its driver on a SEPARATE scale from the dGPU one, so
    # the two are not comparable: L4T R36.4.7 reports 540.4.0 while the dGPU
    # minimum is 560.35.03, and refusing to launch on that reading is simply
    # wrong — that exact release runs TensorRT 10.3.0 and DeepStream fine. So
    # the numeric gate is asked only of a dGPU.
    if [ -e /etc/nv_tegra_release ]; then
      ok "NVIDIA Tegra driver ${driver:-unknown} -- dGPU minimum not applicable"
    elif [ -n "$driver" ] && \
         [ "$(printf '%s\n%s\n' "$PG_MIN_DRIVER" "$driver" | sort -V | head -1)" = "$PG_MIN_DRIVER" ]; then
      # sort -V orders dotted versions correctly; the minimum sorting first
      # means the installed driver is >= it.
      ok "NVIDIA driver $driver (>= $PG_MIN_DRIVER)"
    else
      fail "NVIDIA driver ${driver:-unknown} is older than $PG_MIN_DRIVER"
    fi
  else
    fail "No NVIDIA GPU detected (nvidia-smi missing or reports no device)"
  fi

  if command -v docker >/dev/null 2>&1; then
    ok "Docker installed"
  else
    fail "Docker is not installed -- see https://docs.docker.com/engine/install/"
  fi

  local user_name
  user_name="$(id -un)"
  if [ "$(id -u)" = "0" ]; then
    ok "Running as root -- 'docker' group not required"
  elif id -nG "$user_name" 2>/dev/null | grep -qw docker; then
    ok "User '$user_name' in 'docker' group"
  else
    fail "User '$user_name' is not in the 'docker' group -- fix: sudo usermod -aG docker $user_name (then log out and back in)"
  fi

  if docker info >/dev/null 2>&1; then
    ok "Docker daemon reachable"
  else
    fail "Docker daemon is not reachable -- fix: sudo systemctl start docker"
  fi

  if command -v nvidia-ctk >/dev/null 2>&1 \
     || dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit \
     || [ -e /etc/nv_tegra_release ]; then
    ok "NVIDIA Container Toolkit installed"
  else
    fail "NVIDIA Container Toolkit is not installed -- see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
  fi

  # The container renders through the host's X socket, so a bare Wayland
  # session with no XWayland bridge cannot show a window at all. Not fatal:
  # without a usable display this export runs headless and writes files
  # instead. PG_DISPLAY_STATE is the PROBE's answer, not the variable's — see
  # pg_display_probe for why the two are not the same question.
  if [ -z "${DISPLAY:-}" ]; then
    note "No \$DISPLAY -- running headless (results go to the output directory)"
  elif [ "$PG_DISPLAY_STATE" = "0" ]; then
    ok "Display session: ${XDG_SESSION_TYPE:-X11} (\$DISPLAY=$DISPLAY)"
  elif [ "$PG_DISPLAY_STATE" = "2" ]; then
    note "\$DISPLAY=$DISPLAY -- cannot check it from here (no xdpyinfo or xset); a window will be attempted"
  else
    note "\$DISPLAY=$DISPLAY is set but no X server answers there -- running headless (--display insists, and will say why it cannot)"
  fi

  # Every input check run.sh makes before launching is driven by manifest.json,
  # and a host that cannot read it loses all of them at once: no "you must say
  # what to read", no "that camera does not exist here", no "this pipeline
  # cannot take a file". What that hides is not a crash but a pipeline reading
  # the WRONG source — measured on a PATH without python3 or jq, a
  # fall-through to pipeline.sh's built-in /dev/video0 and a recording of that
  # machine's webcam, with nothing printed either way. Three readers make this
  # very hard to reach; reaching it silently is what must not happen.
  if [ "$PG_MANIFEST_READER" = "none" ]; then
    note "manifest.json cannot be read here (no python3, jq or awk) -- the input checks are skipped; the container will report any problem instead"
    note "if this pipeline's own source is not on this machine, say what to read: --input <file.mp4 | rtsp://... | /dev/videoN>"
  fi

  echo
  if [ "$PG_FAILED" -ne 0 ]; then
    printf '%s%sPrerequisite checks failed -- fix the items marked above.%s\n' \
           "$_c_bold" "$_c_red" "$_c_reset" >&2
    exit 1
  fi
}

host_main() {
  local input="" output="" want_display="auto" force_first_run=0

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --input)     [ "$#" -ge 2 ] || die "--input needs a value"; input="$2";  shift 2 ;;
      --input=*)   input="${1#*=}";                                            shift ;;
      --output)    [ "$#" -ge 2 ] || die "--output needs a value"; output="$2"; shift 2 ;;
      --output=*)  output="${1#*=}";                                           shift ;;
      --headless)  want_display=0;                                             shift ;;
      --display)   want_display=1;                                             shift ;;
      --rebuild)   force_first_run=1;                                          shift ;;
      -h|--help)   usage; exit 0 ;;
      *)           die "unknown option: $1  (try ./run.sh --help)" ;;
    esac
  done

  # A bind mount is written "host:container", so a colon in the path makes the
  # spec ambiguous and docker misparses it. Catch it here with a sentence the
  # recipient can act on, rather than at `docker run` with one they cannot.
  case "$PG_HERE" in
    *:*) die "this folder's path contains a ':' which Docker cannot bind-mount:
    $PG_HERE
  Move the folder somewhere without a colon in the path and try again." ;;
  esac

  [ -f "$PG_PIPELINE_SH" ] || die "pipeline/pipeline.sh is missing -- this export is incomplete.
  (run.sh looked in $PG_HERE. If you started it through a symlink or a copied
  launcher, run it by its real path instead.)"

  printf '%s%s%s\n\n' "$_c_bold" "$PG_PROJECT_NAME" "$_c_reset"
  if [ -n "${DISPLAY:-}" ]; then
    PG_DISPLAY_STATE=0
    pg_display_probe || PG_DISPLAY_STATE=$?
  fi
  host_checklist

  # ── Architecture -> image tag ────────────────────────────────────────────
  # An ARRAY, not a string: "--gpus all" is two words and a UUID
  # ("--gpus device=GPU-…") must survive as one, which no amount of unquoted
  # word-splitting gets right on its own.
  local arch image
  local gpu_flag=()
  case "$(uname -m)" in
    x86_64)
      arch=amd64; gpu_flag=(--gpus all)
      # "--gpus all" is right for one GPU and useless on a shared multi-GPU
      # box, where the recipient has no other way to say which one to use:
      # CUDA_VISIBLE_DEVICES cannot reach the container (docker run inherits
      # nothing) and the shipped nvinfer config pins gpu-id=0. `device=` makes
      # the chosen GPU the container's device 0, which is the one that config
      # asks for.
      [ -n "${PG_GPU:-}" ] && gpu_flag=(--gpus "device=$PG_GPU") ;;
    aarch64)
      arch=arm64; gpu_flag=(--runtime nvidia)
      # Tegra is single-GPU and shares memory with the CPU; there is nothing
      # to choose between, so say so rather than ignoring the request.
      [ -n "${PG_GPU:-}" ] && note "PG_GPU is ignored on Tegra -- it has one integrated GPU" ;;
    *) die "unsupported CPU architecture: $(uname -m)
  This pipeline ships images for x86_64 and aarch64 only." ;;
  esac
  image="${PG_IMAGE:-${PG_IMAGE_REPO}:7.1-${arch}-v3}"

  if docker image inspect "$image" >/dev/null 2>&1; then
    ok "Image present: $image"
  else
    say "Pulling $image -- this is a large download (~35 GB) and happens once."
    docker pull "$image" || die "docker pull $image failed."
  fi
  echo

  # ── Resolve --input ──────────────────────────────────────────────────────
  local kind="" uri="" dev="" live="" raw="$input"
  local extra_dir=""
  if [ -n "$raw" ]; then
    case "$raw" in
      rtsp://*|rtsps://*)
        kind="rtsp";   uri="$raw"; live=1 ;;
      /dev/video*)
        kind="camera"; dev="$raw"; live=1
        [ -e "$dev" ] || die "no such camera: $dev
  Cameras present on this machine: $(pg_cameras)" ;;
      *)
        kind="file"; live=0
        case "$raw" in file://*) raw="${raw#file://}" ;; esac
        local abs real
        abs="$(abspath "$raw")" || die "cannot resolve --input path: $raw"
        # Three different problems, three different sentences. `[ -f ]` alone
        # answers all of them with "no such input file", which for a directory
        # that plainly exists sends the recipient looking for a typo that is
        # not there — and pointing --input at a folder of clips is an obvious
        # first guess.
        [ -e "$abs" ] || die "no such input file: $abs"
        if [ -d "$abs" ]; then
          die "--input takes a single video file, not a directory: $abs
  Pass one of the files inside it."
        fi
        [ -f "$abs" ] || die "--input is not a regular file: $abs"
        # Follow the link before deriving the mount — see pg_realpath.
        real="$(pg_realpath "$abs")" || die "cannot resolve --input path: $raw"
        [ -f "$real" ] || die "--input is a symlink whose target is missing:
    $abs -> $real"
        uri="file://$real"
        extra_dir="$(dirname -- "$real")" ;;
    esac

    # Refuse a kind pipeline.sh cannot switch to, naming what it can — better
    # than a GStreamer negotiation error 200 lines into the log.
    local supported
    supported="$(manifest_get io.supported_inputs)"
    if [ -n "$supported" ]; then
      case " $supported " in
        *" $kind "*) : ;;
        *) die "this pipeline cannot take a '$kind' input.
  It supports: $supported" ;;
      esac
    fi
  else
    # No --input: make sure the source the pipeline was BUILT against is
    # actually here, and if not say exactly which flag fixes it.
    #
    local dflt_kind dflt required
    dflt_kind="$(manifest_get io.default_input_kind)"
    dflt="$(manifest_get io.default_input)"
    required="$(manifest_get io.input_required)"
    if [ "$required" = "1" ]; then
      die "this pipeline reads from ${dflt:-a file that was not part of the export}.
  That file is not included, so you must say what to read:
      ./run.sh --input <file.mp4 | rtsp://... | /dev/videoN>"
    fi
    if [ "$dflt_kind" = "camera" ] && [ -n "$dflt" ] && [ ! -e "$dflt" ]; then
      die "this pipeline reads from the camera $dflt, which does not exist on this machine.
  Pass a source that does:
      ./run.sh --input <file.mp4 | rtsp://... | /dev/videoN>
  Cameras present here: $(pg_cameras)"
    fi
  fi

  # ── Resolve --output ─────────────────────────────────────────────────────
  local out_abs
  if [ -n "$output" ]; then
    # Create it BEFORE resolving: abspath only needs the parent to exist, so
    # `--output ~/new/nested/dir` would otherwise be refused as unresolvable
    # when the honest answer is that it simply is not there yet.
    mkdir -p "$output" || die "cannot create output directory: $output"
    out_abs="$(abspath "$output")" || die "cannot resolve --output path: $output"
  else
    out_abs="$PG_DEFAULT_OUT"
  fi
  mkdir -p "$out_abs" || die "cannot create output directory: $out_abs"
  case "$out_abs" in
    *:*) die "the output path contains a ':' which Docker cannot bind-mount: $out_abs" ;;
  esac

  # ── Resolve display ──────────────────────────────────────────────────────
  # The auto rule is the design's: no display -> headless. "No display" means
  # none answered the probe, not "the variable happened to be empty" — see
  # pg_display_probe. State 2 (nothing here can tell) counts as available,
  # because the container may well manage what this host cannot check.
  if [ "$want_display" = "auto" ]; then
    if [ -n "${DISPLAY:-}" ] && [ "$PG_DISPLAY_STATE" != "1" ]; then
      want_display=1
    else
      want_display=0
    fi
  fi
  if [ "$want_display" = "1" ] && [ -z "${DISPLAY:-}" ]; then
    die "--display was asked for but \$DISPLAY is not set in this shell.
  Run from a graphical session, or use --headless."
  fi
  if [ "$want_display" = "1" ] && [ "$PG_DISPLAY_STATE" = "1" ]; then
    die "--display was asked for but nothing answers on \$DISPLAY=$DISPLAY.
  Nothing can open a window from here. Check that an X server is running and
  that this shell's \$DISPLAY (and \$XAUTHORITY) point at it -- or use
  --headless, which writes the results to files instead."
  fi
  local headless=1
  [ "$want_display" = "1" ] && headless=0

  # ── JSON_OUT, if the caller set it ───────────────────────────────────────
  # pipeline.sh's header advertises `JSON_OUT=/tmp/d.json ./run.sh`, and it is
  # a PATH: forwarding it without mounting the directory it names would write
  # the metadata into the container's throwaway filesystem, where it is
  # discarded on exit and nothing says so.
  local json_dir=""
  if [ -n "${JSON_OUT:-}" ]; then
    local json_base
    json_base="$(basename -- "$JSON_OUT")"
    json_dir="$(dirname -- "$JSON_OUT")"
    # Created before it is resolved, like --output: naming a directory that is
    # not there yet is a perfectly ordinary thing to do, and refusing it as
    # "unresolvable" would be answering a different question.
    mkdir -p "$json_dir" || die "cannot create the directory JSON_OUT names: $json_dir"
    json_dir="$(cd -- "$json_dir" 2>/dev/null && pwd -P)" \
      || die "cannot resolve JSON_OUT: $JSON_OUT"
    case "$json_dir" in
      *:*) die "JSON_OUT is in a folder whose path contains a ':', which Docker cannot bind-mount:
    $json_dir
  Point JSON_OUT somewhere without a colon in the path." ;;
    esac
    JSON_OUT="$json_dir/$json_base"
  fi

  # ── Mounts ───────────────────────────────────────────────────────────────
  # The export folder is mounted AT ITS OWN PATH so every path the container
  # computes is a path the host also understands — which is what lets the
  # recipient find the outputs afterwards without translating anything.
  # Order matters: the writable destinations go in before the read-only input,
  # so an input that happens to sit inside one of them stays writable.
  pg_add_dir_mount "$PG_HERE"
  pg_add_dir_mount "$out_abs"
  [ -n "$json_dir" ] && pg_add_dir_mount "$json_dir"
  if [ -n "$extra_dir" ]; then
    # The same hazard the export root and the output directory are already
    # guarded against, on the one path the recipient supplies most often.
    # Unguarded it reaches `docker run` as "invalid spec: …: too many colons"
    # and exit 125, which names neither --input nor the fix. Directories called
    # 2026-08-13T10:15 or clips:raw are ordinary.
    case "$extra_dir" in
      *:*) die "the folder holding your --input file contains a ':' which Docker cannot bind-mount:
    $extra_dir
  Move or copy the file somewhere without a colon in the path." ;;
      /) die "an --input file directly in / cannot be mounted; move it into a directory." ;;
    esac
    pg_add_dir_mount "$extra_dir" ro
  fi
  # /dev wholesale, exactly as PipeGen's own launcher does: v4l2 cameras
  # (char major 81) and the Jetson encoder node /dev/v4l2-nvenc both live there,
  # and the encoder branch of pipeline.sh probes for that node at run time.
  # The cgroup rules are inert on a machine with no such devices.
  local mounts=("${PG_MOUNTS[@]}" -v /dev:/dev)

  local envs=(
    -e PG_IN_CONTAINER=1
    -e "PG_ROOT=$PG_HERE"
    -e "PG_OUT=$out_abs"
    -e "PG_INPUT=$input"
    -e "PG_INPUT_KIND=$kind"
    -e "PG_INPUT_URI=$uri"
    -e "PG_INPUT_DEV=$dev"
    -e "PG_LIVE=$live"
    -e "PG_HEADLESS=$headless"
    -e "PG_DISPLAY=$want_display"
    -e "PG_PROJECT=$PG_PROJECT_NAME"
    -e "PG_IMAGE=$image"
    -e "PG_FORCE_FIRST_RUN=$force_first_run"
    -e "PG_HOST_UID=$(id -u)"
    -e "PG_HOST_GID=$(id -g)"
  )

  # Every override the export documents INSIDE ITSELF. pipeline.sh's header
  # tells the recipient to set CAM_W/CAM_H/CAM_FPS/CAM_FMT to match their
  # camera, CAM_PROBE=0 to stop the v4l2 probe choosing for them, and
  # MQTT_HOST=… / JSON_OUT=… to redirect the metadata — and `docker run`
  # inherits NOTHING from this shell, so a fixed -e list drops all of it
  # silently. Measured before this loop existed: CAM_PROBE=0 BITRATE=2000
  # JSON_OUT=… changed nothing whatsoever, and the run still reported success.
  # An allow-list rather than --env-file or a blanket copy: only what
  # pipeline.sh actually reads, so nothing else about this shell leaks into
  # the container. Unset stays unset, which is what keeps pipeline.sh's own
  # defaults in charge.
  local v
  for v in CAM CAM_W CAM_H CAM_FPS CAM_FMT CAM_PROBE BITRATE FORCE_SW_ENC JSON_OUT JSON_MIN_CONF JSON_MASK_OUTLINE MQTT_HOST MQTT_PORT MQTT_TOPIC; do
    if [ -n "${!v:-}" ]; then envs+=(-e "$v=${!v}"); fi
  done

  local tty_args=()
  if [ -t 0 ] && [ -t 1 ]; then
    # A tty puts gst-launch in the terminal's foreground process group, so
    # Ctrl-C reaches IT and `gst-launch -e` can emit EOS and finalise the mp4.
    # Without one, docker proxies SIGINT to PID 1 only and the container half
    # has to forward it by hand (see container_main).
    tty_args=(-i -t); envs+=(-e PG_TTY=1)
  else
    envs+=(-e PG_TTY=0)
  fi

  if [ "$want_display" = "1" ]; then
    envs+=(-e "DISPLAY=$DISPLAY")
    if [ -d /tmp/.X11-unix ]; then
      mounts+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
    fi
    # X authorisation, in the order that actually works.
    #
    # THE COOKIE, first choice. As far as X is concerned the container is a
    # different host, so it needs the MIT-MAGIC-COOKIE for this display.
    # `-e XAUTHORITY=$XAUTHORITY` on its own is useless — the file it names is
    # on the HOST filesystem and nothing mounts it (measured: the container
    # sees "No such file or directory" for /run/user/1000/gdm/Xauthority) — and
    # `xhost +local:root` grants the LOCAL family only, which a TCP $DISPLAY
    # can never use. Between them that made `ssh -X`, the single most common
    # way to drive a remote GPU box, fail at preroll with exit 255 and an empty
    # output directory, on the DEFAULT path. Extracting one entry with the
    # address family rewritten to FamilyWild (the leading `ffff`) is what makes
    # the cookie match whatever host name the container thinks it has.
    local used_cookie=0
    if command -v xauth >/dev/null 2>&1; then
      mkdir -p "$PG_HERE/.pipegen-state"
      rm -f "$PG_XAUTH"
      # 0600 from the moment it exists: this file is a credential for the
      # recipient's screen, not part of the export.
      ( umask 077
        xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' \
          | xauth -f "$PG_XAUTH" nmerge - >/dev/null 2>&1 ) || true
      if [ -s "$PG_XAUTH" ]; then
        used_cookie=1
        PG_XAUTH_MADE=1
        mounts+=(-v "$PG_XAUTH:/tmp/.pgxauth:ro")
        envs+=(-e "XAUTHORITY=/tmp/.pgxauth")
      else
        rm -f "$PG_XAUTH"
      fi
    fi
    if [ "$used_cookie" = "0" ]; then
      # No xauth, or no entry for this display. Mount the caller's own
      # authority file if it is mountable — that at least carries a cookie.
      if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY:-}" ]; then
        case "$XAUTHORITY" in
          *:*) : ;;
          *) mounts+=(-v "$XAUTHORITY:$XAUTHORITY:ro")
             envs+=(-e "XAUTHORITY=$XAUTHORITY")
             used_cookie=1 ;;
        esac
      fi
    fi
    if [ "$used_cookie" = "0" ] && command -v xhost >/dev/null 2>&1; then
      # Last resort, and only useful for a local (unix-socket) display.
      # `xhost +local:root` is not scoped to this container and does not expire
      # with the run: it opens the display to every local process of every
      # account on the machine, for the rest of the login session — measured
      # still present in `xhost` output long after run.sh exited. So it is
      # granted only if it was not already there, and taken back on the way
      # out (see pg_display_cleanup).
      if xhost 2>/dev/null | grep -qx 'LOCAL:'; then
        : # someone else's grant -- leave it exactly as found
      elif xhost +local:root >/dev/null 2>&1; then
        PG_XHOST_MADE=1
      fi
    fi
  fi

  say "Starting $image"
  # --entrypoint bash is not cosmetic. The image's own entrypoint ends with
  #     /opt/nvidia/nvidia_entrypoint.sh $@
  # — an UNQUOTED "$@" — so it re-splits its arguments on whitespace before
  # anything execs them. Handed a folder whose path contains a space it fails
  # with "bash: /tmp/pgexport-test/somewhere: No such file or directory", which
  # is a recipient with a perfectly ordinary "My Downloads" directory and no
  # idea why. Going straight to bash also drops the licence banner the
  # entrypoint prints over every run. Verified equivalent: the container
  # environment is identical either way apart from SHLVL (66 variables, one
  # differing), because the entrypoint only ever cat's banners.
  local docker_args=(
    run --rm "${gpu_flag[@]}"
    --entrypoint bash
    --device-cgroup-rule "c 81:* rmw"
    --device-cgroup-rule "c 189:* rmw"
    --network host
    "${tty_args[@]+"${tty_args[@]}"}"
    "${envs[@]}"
    "${mounts[@]}"
    -w "$PG_HERE"
    "$image" "$PG_HERE/run.sh"
  )

  if [ "$PG_XAUTH_MADE" = "1" ] || [ "$PG_XHOST_MADE" = "1" ]; then
    # NOT `exec`: display mode left something behind on this machine — an X
    # cookie in .pipegen-state/, or a LOCAL: entry in the X server's access
    # list — and exec would replace this shell before it could take either
    # back. So wait for the container, clean up, and pass its status on.
    trap 'pg_display_cleanup' EXIT
    trap 'pg_display_cleanup; exit 130' INT TERM
    local rc=0
    docker "${docker_args[@]}" || rc=$?
    exit "$rc"
  fi
  exec docker "${docker_args[@]}"
}


# ═══════════════════════════════════════════════════════════════════════════
# CONTAINER PHASE
# ═══════════════════════════════════════════════════════════════════════════

# Everything the container writes into the bind mount is written as root.
# Left alone, the recipient cannot delete their own output directory or
# re-export over it, so hand it back on the way out.
pg_reown() {
  [ -n "${PG_HOST_UID:-}" ] || return 0
  [ "$PG_HOST_UID" = "0" ] && return 0
  chown -R "$PG_HOST_UID:${PG_HOST_GID:-$PG_HOST_UID}" \
        "$PG_ROOT/.pipegen-state" \
        "$PG_ROOT/pipeline/build" \
        "$PG_ROOT/models" \
        "${PG_OUT:-$PG_ROOT/out}" 2>/dev/null || true
  # JSON_OUT can point outside all of those, and a root-owned file dropped in
  # the recipient's own directory is the same nuisance as a root-owned output
  # directory. Just the file: the directory around it is theirs already.
  if [ -n "${JSON_OUT:-}" ] && [ -e "${JSON_OUT:-}" ]; then
    chown "$PG_HOST_UID:${PG_HOST_GID:-$PG_HOST_UID}" "$JSON_OUT" 2>/dev/null || true
  fi
  return 0
}

# The identity of the machine this engine has to run on. All three fields,
# because none of them is sufficient alone: a Jetson Orin reports TensorRT
# 10.3.0 — identical to the x86 box that built these plans — and still refuses
# them ("Platform specific tag mismatch detected"), while the same GPU under a
# newer TensorRT refuses a plan built by an older one.
pg_fingerprint() {
  printf '%s|%s|%s|%s|%s\n' \
    "1" "$(uname -m)" "$(pg_gpu_name)" "$(pg_trt_version)" \
    "${PG_IMAGE:-unknown}"
}

container_main() {
  PG_ROOT="${PG_ROOT:-$PG_HERE}"
  PG_OUT="${PG_OUT:-$PG_ROOT/out}"
  export PG_ROOT PG_OUT
  export PG_INPUT="${PG_INPUT:-}"        PG_INPUT_KIND="${PG_INPUT_KIND:-}"
  export PG_INPUT_URI="${PG_INPUT_URI:-}" PG_INPUT_DEV="${PG_INPUT_DEV:-}"
  export PG_LIVE="${PG_LIVE:-}"          PG_HEADLESS="${PG_HEADLESS:-1}"
  export PG_DISPLAY="${PG_DISPLAY:-0}"   PG_PROJECT="${PG_PROJECT:-$PG_PROJECT_NAME}"

  mkdir -p "$PG_OUT" "$PG_ROOT/.pipegen-state"
  trap pg_reown EXIT

  # ── One run of a folder at a time ────────────────────────────────────────
  # Two runs of the same export are not independent. They share
  # pipeline/build/, models/*.plan and — unless --output says otherwise — the
  # same fixed output FILENAMES, so the second silently truncates the first's
  # results while both report success. Measured with two clips and one output
  # directory: a pose_results.json holding a complete document, then 186 049
  # NUL bytes where the second open-with-truncate left a hole, then the other
  # run's records resuming mid-file; and an output.mp4 half again the size of
  # either run's, decoding to neither. Nothing warned, and both exited 0.
  #
  # The lock lives in the export folder, so it is shared by every container
  # started from it, and the fd is released when this shell exits however it
  # exits. --output is deliberately NOT part of the identity: two runs writing
  # to different directories still share the engine and the parser build.
  exec 9>"$PG_ROOT/.pipegen-state/run.lock" || die "cannot write to $PG_ROOT/.pipegen-state/run.lock -- is the export folder read-only?"
  if command -v flock >/dev/null 2>&1; then
    flock -n 9 || die "another run of this export is already in progress.
  Its results would be overwritten by this one. Wait for it to finish, or use
  a separate copy of this folder to run two things at once."
  fi

  # ── Steps 4-5: first run only ────────────────────────────────────────────
  # The stamp records the machine, not just "done". Carrying the folder to a
  # different GPU or a different TensorRT invalidates it automatically, which is
  # the whole point: the parser .so built here is the wrong architecture there,
  # and the engine resolved here is unloadable there.
  local want have
  want="$(pg_fingerprint)"
  have=""
  [ -f "$PG_STAMP" ] && have="$(head -1 "$PG_STAMP" 2>/dev/null || true)"

  # A matching stamp says "this machine already did the setup". It does NOT say
  # the products of that setup are still here, and one of them is both the
  # largest file in the export and the only one that cannot heal itself:
  # pipeline.sh's inherited `make` transparently rebuilds a missing parser .so,
  # but nothing rebuilds a missing engine once the stamp is trusted. A recipient
  # who deletes the ~100 MB .plan, or a copy that dropped it, then gets
  # TensorRT's "failed to build network since there is no model file matched"
  # and NVDSINFER_CONFIG_FAILED at preroll — measured, exit 255 and an empty
  # output directory — instead of the 15-second rebuild from the ONNX this
  # export ships for exactly that purpose. first-run.sh needs no help here: its
  # shipped-engine shortcut is already gated on the plan actually loading.
  if [ -n "$have" ]; then
    local _plan
    while IFS= read -r _plan; do
      [ -n "$_plan" ] || continue
      if [ ! -s "$PG_ROOT/$_plan" ]; then
        say "a model engine is missing from this export ($_plan) -- redoing setup."
        have=""
        break
      fi
    done <<EOF
$(python3 -c "$_PG_PY_PLANS" "$PG_MANIFEST" 2>/dev/null || true)
EOF
  fi

  if [ "${PG_FORCE_FIRST_RUN:-0}" = "1" ] || [ "$want" != "$have" ]; then
    if [ -n "$have" ] && [ "$want" != "$have" ]; then
      say "This machine differs from the last one that ran this export -- redoing setup."
    fi
    [ -f "$PG_FIRST_RUN_SH" ] || die "pipeline/first-run.sh is missing -- this export is incomplete."
    bash "$PG_FIRST_RUN_SH"
    # Written only after first-run.sh exits 0, so an interrupted setup is
    # retried rather than skipped.
    printf '%s\n' "$want" > "$PG_STAMP"
  else
    say "Setup already done for this machine (delete .pipegen-state/first-run.stamp or pass --rebuild to redo it)."
  fi

  # ── Steps 6-7: I/O resolution and launch ─────────────────────────────────
  if [ -n "$PG_INPUT_KIND" ]; then
    say "input: $PG_INPUT_KIND -> ${PG_INPUT_URI:-$PG_INPUT_DEV}"
  fi
  if [ "$PG_HEADLESS" = "1" ]; then
    say "headless: no window; results are written to $PG_OUT"
  else
    say "display: on (\$DISPLAY=${DISPLAY:-})"
  fi
  say "output: $PG_OUT"

  local rc=0
  if [ "${PG_TTY:-0}" = "1" ]; then
    # The terminal signals the whole foreground group, so gst-launch gets the
    # Ctrl-C directly and shuts down cleanly. Nothing to forward.
    bash "$PG_PIPELINE_SH" || rc=$?
  else
    # No tty: docker delivers SIGINT to PID 1 only and bash does not pass it on,
    # which would orphan gst-launch and leave a half-written, unplayable mp4.
    # Ignore the signal here (so this shell survives to reap and to reown) and
    # re-send it to the process group, which is where gst-launch is.
    bash "$PG_PIPELINE_SH" &
    local child=$!
    trap 'trap "" INT TERM; kill -INT -- "-$$" 2>/dev/null || true' INT TERM
    wait "$child" || rc=$?
    # `wait` returns the moment a trapped signal arrives, NOT when the child
    # finally exits -- so the first wait comes back while gst-launch is still
    # doing the very shutdown the signal asked for. Measured: run.sh printed
    # "Pipeline exited with status 130" three lines BEFORE "Forcing EOS on the
    # pipeline" / "Waiting for EOS...". Returning there tears the container
    # down mid-flush, which for a recording is a truncated mp4 with no moov
    # atom -- the exact unplayable file `gst-launch -e` exists to prevent.
    # The handler disarms itself, so this second wait blocks properly.
    while kill -0 "$child" 2>/dev/null; do
      wait "$child" 2>/dev/null || rc=$?
    done
    trap - INT TERM
  fi

  echo
  if [ "$rc" -eq 0 ]; then
    say "Done. Results are in $PG_OUT"
  else
    say "Pipeline exited with status $rc. Anything it produced is in $PG_OUT"
  fi
  return "$rc"
}


# ═══════════════════════════════════════════════════════════════════════════
if [ "${PG_IN_CONTAINER:-0}" = "1" ]; then
  container_main
else
  host_main "$@"
fi
