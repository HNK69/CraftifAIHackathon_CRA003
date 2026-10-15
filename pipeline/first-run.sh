#!/usr/bin/env bash
# Steps 4 and 5 of the export's run flow, done ONCE per machine, INSIDE the
# container. run.sh calls this and then writes the stamp that makes later runs
# skip it; you can also run it by hand for debugging.
#
#   4. Build the parser libraries FROM SOURCE.
#      The export ships parsers/*.cpp + *.h + Makefile and no .so at all,
#      because a .so built on an x86 laptop is not loadable on an Orin. This
#      step is what makes one export folder run on both.
#
#   5. Resolve the inference engine.
#      A TensorRT plan is tied to the GPU AND the TensorRT build that produced
#      it. Where the fingerprint in manifest.json matches this machine, the
#      shipped plan is used as-is. Where it does not, the plan is rebuilt from
#      the shipped ONNX at the recorded precision and cached beside it under a
#      name naming this machine, so coming back to it later is free.
#
# Bias is hard toward rebuilding: a needless rebuild costs one trtexec run —
# measured at the optimization level the manifest now ships (3, TensorRT's own
# default): 15 s on an RTX 3050, 37-46 s on a Jetson Orin — while wrongly
# trusting a foreign plan costs a pipeline that fails at preroll with
# NVDSINFER_CONFIG_FAILED and an error the recipient cannot act on.
set -euo pipefail

PG_ROOT="${PG_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
PG_MANIFEST="$PG_ROOT/manifest.json"
PG_PARSERS="$PG_ROOT/pipeline/parsers"

_c_bold=''; _c_red=''; _c_reset=''
if [ -t 1 ]; then _c_bold=$'\033[1m'; _c_red=$'\033[31m'; _c_reset=$'\033[0m'; fi
say() { printf '%s[first-run]%s %s\n' "$_c_bold" "$_c_reset" "$1"; }
die() { printf '\n%s%serror:%s %s\n' "$_c_bold" "$_c_red" "$_c_reset" "$*" >&2; exit 1; }

pg_slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' \
                   | sed -e 's/[^a-z0-9]\{1,\}/-/g' -e 's/^-//' -e 's/-$//'
}

# ── Step 4: parsers ─────────────────────────────────────────────────────────
# -B rebuilds every target from the CURRENT sources regardless of timestamps.
# Plain `make` is wrong here for the reason this whole step exists: after the
# folder moves to another architecture, pipeline/build/*.so is NEWER than the
# .cpp files that would replace it, so `make` declares everything up to date
# and leaves a library this machine cannot load.
build_parsers() {
  if [ ! -f "$PG_PARSERS/Makefile" ]; then
    say "no parsers to build"
    return 0
  fi
  command -v make >/dev/null 2>&1 || die "make is not available in this container image."
  say "building parser libraries for $(uname -m) ..."
  make -B -C "$PG_PARSERS" || die "parser build failed (output above).
  The sources are in pipeline/parsers/ and the build is a plain 'make -B -C pipeline/parsers'."
  say "parser libraries -> pipeline/build/"
}

# ── Step 5: engines ─────────────────────────────────────────────────────────
# shellcheck disable=SC2016
_PG_PY_MODELS='
import json, sys
doc = json.load(open(sys.argv[1]))
for m in doc.get("models") or []:
    shapes = m.get("shapes") or {}
    row = [
        str(m.get("name") or ""),
        str(m.get("onnx") or ""),
        str(m.get("plan") or ""),
        str(m.get("plan_sha256") or ""),
        str(m.get("precision") or "fp32"),
        str(m.get("workspace_mb") or 1024),
        str(0 if m.get("builder_optimization_level") is None
              else m.get("builder_optimization_level")),
        str(shapes.get("min") or ""),
        str(shapes.get("opt") or ""),
        str(shapes.get("max") or ""),
    ]
    print("\t".join(c.replace("\t", " ").replace("\n", " ") for c in row))
'

# shellcheck disable=SC2016
_PG_PY_ORIGIN='
import json, sys
doc = json.load(open(sys.argv[1]))
o = doc.get("origin") or {}
print("%s\t%s\t%s" % (o.get("gpu") or "", o.get("trt") or "", o.get("arch") or ""))
'

# The ONLY reliable check that a plan is loadable here.
# `trtexec --loadEngine=<foreign> --skipInference` prints the deserialization
# error and then "&&&& PASSED" and exits 0 — measured on a real Orin against a
# real x86 plan — so it cannot be used for this. deserialize_cuda_engine
# returns None rather than raising, so the probe must test the return value.
# shellcheck disable=SC2016
_PG_PY_PROBE='
import sys
try:
    import tensorrt as trt
except Exception:
    sys.exit(2)
with open(sys.argv[1], "rb") as fh:
    blob = fh.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
sys.exit(0 if runtime.deserialize_cuda_engine(blob) else 1)
'

plan_loads_here() {   # plan_loads_here <plan> -> 0 when this machine can use it
  [ -s "$1" ] || return 1
  python3 -c "$_PG_PY_PROBE" "$1" >/dev/null 2>&1
}

pg_sha256() {         # pg_sha256 <file> -> hex digest, or "" if nothing can hash
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" 2>/dev/null | cut -d' ' -f1
  else
    python3 -c 'import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],"rb") as fh:
    for chunk in iter(lambda: fh.read(1<<20), b""):
        h.update(chunk)
print(h.hexdigest())' "$1" 2>/dev/null || true
  fi
}

# TensorRT's own default optimization level, used as the fallback below.
PG_TRT_DEFAULT_LEVEL=3

# Fills the global PG_TRTEXEC_ARGS. A global rather than a return value because
# bash cannot return an array, and the argument list must stay an array —
# flattening it to a string would word-split any path containing a space.
_trtexec_args() {     # _trtexec_args <onnx> <tmp> <prec> <ws_mb> <lvl> <min> <opt> <max>
  PG_TRTEXEC_ARGS=(
    "--onnx=$1"
    "--saveEngine=$2"
    "--memPoolSize=workspace:$4"
    "--builderOptimizationLevel=$5"
    "--profilingVerbosity=detailed"
    # No inference: this step wants the engine, not a benchmark.
    "--skipInference"
  )
  # fp32 adds no flag at all — it is the builder's default. Nothing sets
  # --sparsity (PipeGen's released builder does not) and nothing sets --noTF32
  # (TF32 is default-on for both the builder API and trtexec, so the defaults
  # already agree).
  [ "$3" = "fp16" ] && PG_TRTEXEC_ARGS+=("--fp16")
  if [ -n "$6" ] && [ -n "$7" ] && [ -n "$8" ]; then
    PG_TRTEXEC_ARGS+=("--minShapes=$6" "--optShapes=$7" "--maxShapes=$8")
  fi
}

rebuild_engine() {    # rebuild_engine <onnx> <dst> <prec> <ws_mb> <lvl> <min> <opt> <max>
  local onnx="$1" dst="$2" prec="$3" ws="$4" lvl="$5" smin="$6" sopt="$7" smax="$8"
  command -v trtexec >/dev/null 2>&1 || die "trtexec is not on PATH in this container image."
  [ -f "$onnx" ] || die "cannot rebuild the engine: $onnx is missing from the export."

  # Build to a temporary name and move it into place only on success, so an
  # interrupted or failed build never leaves a truncated .plan that the next
  # run would try to load.
  local tmp="$dst.building.$$"

  # The level comes from the manifest, and the exporter records TensorRT's own
  # default 3 there, so `lvl` is normally 3 and this single build is the whole
  # story. It is passed explicitly rather than left to the default because the
  # manifest is the record of how the shipped engine was built; the level the
  # ORIGIN project happened to use is carried separately, as provenance, and is
  # not what gets built here.
  say "building $(basename "$dst") from $(basename "$onnx") at $prec ..."
  _trtexec_args "$onnx" "$tmp" "$prec" "$ws" "$lvl" "$smin" "$sopt" "$smax"
  if trtexec "${PG_TRTEXEC_ARGS[@]}"; then
    mv -f "$tmp" "$dst"
    return 0
  fi
  rm -f "$tmp"

  # A manifest MAY ask for a lower level than the default — an older export, or
  # one produced by something that copied the origin project's own setting — and
  # a low level is exactly where a build has no second chance: level 0 keeps a
  # MINIMAL tactic pool, so one layer with no viable tactic on this GPU fails
  # the whole build. Measured on a Jetson Orin at level 0: three consecutive
  # first-runs died with
  #   Error Code 10: Internal Error (Could not find any implementation for node
  #   PWN(PWN(node_of_260), PWN(node_of_263)).)
  # while the SAME command in an otherwise idle container succeeded in 1.8 s —
  # the failures all landed immediately after the parser build, when memory is
  # most contended. Level 3 has many more tactics to choose from and built the
  # same ONNX every time (37-46 s).
  #
  # So fidelity to the recorded level is the preference and a working engine is
  # the requirement. This retry is inert for the level the exporter ships (the
  # guard cannot fire when the manifest already asks for the default); it is
  # here for the manifests where it can, including one that omits the field
  # altogether, since the reader then falls back to 0.
  if [ "$lvl" != "$PG_TRT_DEFAULT_LEVEL" ]; then
    say "build failed at optimization level $lvl -- retrying at TensorRT's default ($PG_TRT_DEFAULT_LEVEL), which is slower but searches far more tactics"
    _trtexec_args "$onnx" "$tmp" "$prec" "$ws" "$PG_TRT_DEFAULT_LEVEL" "$smin" "$sopt" "$smax"
    if trtexec "${PG_TRTEXEC_ARGS[@]}"; then
      mv -f "$tmp" "$dst"
      return 0
    fi
    rm -f "$tmp"
  fi

  die "TensorRT could not build an engine from $(basename "$onnx") (output above).
  Tried optimization level $lvl and the default $PG_TRT_DEFAULT_LEVEL."
}

resolve_engines() {
  [ -f "$PG_MANIFEST" ] || die "manifest.json is missing -- this export is incomplete."
  command -v python3 >/dev/null 2>&1 || die "python3 is not available in this container image."

  local here_gpu here_trt here_arch origin fp_slug
  here_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  here_trt="$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)"
  here_arch="$(uname -m)"
  [ -n "$here_gpu" ] || die "nvidia-smi reported no GPU inside the container.
  The container was started without working GPU access -- check the NVIDIA container toolkit."
  [ -n "$here_trt" ] || die "TensorRT is not importable inside the container image."

  origin="$(python3 -c "$_PG_PY_ORIGIN" "$PG_MANIFEST")"
  local origin_gpu origin_trt origin_arch
  IFS=$'\t' read -r origin_gpu origin_trt origin_arch <<EOF
$origin
EOF

  local match=0
  if [ "$here_gpu" = "$origin_gpu" ] && [ "$here_trt" = "$origin_trt" ] \
     && [ "$here_arch" = "$origin_arch" ]; then
    match=1
  fi

  say "this machine : $here_arch / $here_gpu / TensorRT $here_trt"
  say "built on     : ${origin_arch:-?} / ${origin_gpu:-?} / TensorRT ${origin_trt:-?}"
  if [ "$match" = "1" ]; then
    say "fingerprint MATCHES -- the shipped engines should load here as they are."
  else
    say "fingerprint DIFFERS -- engines will be rebuilt from the shipped ONNX."
  fi

  fp_slug="$(pg_slug "$here_arch-$here_gpu-trt$here_trt")"
  local origin_slug
  origin_slug="$(pg_slug "${origin_arch:-unknown}-${origin_gpu:-unknown}-trt${origin_trt:-unknown}")"

  local n=0
  local name onnx_rel plan_rel plan_sha prec ws lvl smin sopt smax
  while IFS=$'\t' read -r name onnx_rel plan_rel plan_sha prec ws lvl smin sopt smax; do
    [ -n "$plan_rel" ] || continue
    n=$((n + 1))
    local onnx="$PG_ROOT/$onnx_rel" plan="$PG_ROOT/$plan_rel"
    local plan_dir plan_base cached origin_cache
    plan_dir="$(dirname "$plan")"
    plan_base="$(basename "$plan")"
    mkdir -p "$plan_dir"
    # Cached engines are hidden dotfiles beside the active one: same directory
    # so they travel with the export, out of the way so `ls models/` still
    # shows exactly the two files the layout promises.
    cached="$plan_dir/.$plan_base.$fp_slug"
    origin_cache="$plan_dir/.$plan_base.$origin_slug"

    # The active slot is the path the nvinfer config points at, so whatever
    # ends up there is what actually runs.
    if [ "$match" = "1" ]; then
      if plan_loads_here "$plan"; then
        say "$name: using the shipped engine"
        continue
      fi
      # The fingerprint said this engine would load and it did not. Recovery is
      # the same rebuild as always, but saying nothing here leaves the recipient
      # watching a 15-to-50-second trtexec run three lines under the assertion
      # that no rebuild was needed. Measured with the shipped plan truncated to
      # 4 MB: "fingerprint MATCHES", then straight into "building ...", with no
      # line anywhere saying the shipped engine had been rejected or why.
      local shipped_sha=""
      [ -s "$plan" ] && shipped_sha="$(pg_sha256 "$plan")"
      if [ ! -s "$plan" ]; then
        say "$name: the shipped engine is not in this export -- rebuilding it from the ONNX"
      elif [ -n "$plan_sha" ] && [ -n "$shipped_sha" ] && [ "$shipped_sha" != "$plan_sha" ]; then
        say "$name: the shipped engine does not match its checksum in manifest.json -- it arrived damaged; rebuilding it from the ONNX"
      else
        say "$name: the shipped engine will not load here even though the fingerprint matches -- rebuilding it from the ONNX"
      fi
    fi

    # Preserve the shipped plan once, under the name of the machine that built
    # it, before the active slot is overwritten. Coming back to that machine
    # then costs a copy instead of a rebuild.
    #
    # Gated on the manifest's checksum, and that gate is load-bearing: after the
    # first rebuild the active slot no longer holds the shipped plan, so a run
    # that finds the caches missing (someone tidied the dotfiles, or copied the
    # folder with a tool that skipped them) would file THIS machine's engine
    # under the ORIGIN machine's name. Measured on the Orin before this check
    # existed: .plan.x86-64-...-rtx-3050-... contained the aarch64 engine,
    # sha256 227f5aa2… against the manifest's 0285b83e…. Nothing wrong ever
    # ran -- the deserialize probe rejects a mislabelled cache -- but it cost
    # 97 MB and made the directory lie about itself.
    if [ "$match" != "1" ] && [ -f "$plan" ] && [ ! -f "$origin_cache" ] \
       && [ -n "$plan_sha" ] && [ "$(pg_sha256 "$plan")" = "$plan_sha" ]; then
      cp -f "$plan" "$origin_cache"
    fi

    if [ -f "$cached" ] && plan_loads_here "$cached"; then
      say "$name: reusing the engine built here earlier"
      cp -f "$cached" "$plan"
      continue
    fi

    rebuild_engine "$onnx" "$plan" "$prec" "$ws" "$lvl" "$smin" "$sopt" "$smax"
    plan_loads_here "$plan" || die "the engine just built for $name still will not load here.
  This is a TensorRT/driver problem on this machine, not an export problem."
    cp -f "$plan" "$cached"
    say "$name: engine ready ($prec)"
  done <<EOF
$(python3 -c "$_PG_PY_MODELS" "$PG_MANIFEST")
EOF

  if [ "$n" = "0" ]; then
    say "no models declared in manifest.json -- nothing to resolve"
  fi
}

say "one-time setup for this machine"
build_parsers
resolve_engines
say "setup complete"
