#!/usr/bin/env bash
# Shared helpers for the demo scripts.
#
# Sourced (not executed) by the run_*/train_* scripts.

# Load KEY=VALUE pairs from a dotenv file WITHOUT overriding variables that are
# already set in the environment. This matches standard dotenv behavior: an
# explicit `export FOO=bar` before running a script always wins over .env.
#
# Supported format: simple KEY=VALUE lines, optional inline "# comment"
# (preceded by a space), blank lines and full-line comments are ignored.
load_env_file() {
  local file="$1"
  [[ -f "${file}" ]] || return 0

  local line key val
  while IFS= read -r line || [[ -n "${line}" ]]; do
    # Trim leading whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    # Skip blanks and full-line comments.
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    # Require a KEY=VALUE shape.
    [[ "${line}" != *=* ]] && continue

    key="${line%%=*}"
    val="${line#*=}"
    # Strip any whitespace inside the key (keys never contain spaces).
    key="${key//[[:space:]]/}"
    # Strip an inline comment that is preceded by a space (" # ...").
    val="${val%% #*}"
    # Trim surrounding whitespace from the value.
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"

    [[ -z "${key}" ]] && continue
    # Only set if not already present in the environment.
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${val}"
    fi
  done < "${file}"
}
