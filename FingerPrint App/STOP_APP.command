#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")" && pwd)"
pid_file="$project_root/data/streamlit-process.pid"
stopped=false

stop_if_project_server() {
    local process_id="$1"
    local command_line
    command_line="$(ps -p "$process_id" -o command= 2>/dev/null || true)"
    if [[ "$command_line" == *"streamlit"* && "$command_line" == *"$project_root/app.py"* ]]; then
        kill "$process_id" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "$process_id" 2>/dev/null; then
                stopped=true
                return
            fi
            sleep 0.15
        done
        kill -9 "$process_id" 2>/dev/null || true
        stopped=true
    fi
}

if [[ -f "$pid_file" ]]; then
    saved_pid="$(tr -cd '0-9' < "$pid_file")"
    if [[ -n "$saved_pid" ]]; then
        stop_if_project_server "$saved_pid"
    fi
fi

while IFS= read -r listener_pid; do
    [[ -n "$listener_pid" ]] && stop_if_project_server "$listener_pid"
done < <(lsof -tiTCP:8501 -sTCP:LISTEN 2>/dev/null || true)

rm -f "$pid_file"
if [[ "$stopped" == true ]]; then
    echo "Fingerprint Attendance stopped."
else
    echo "Fingerprint Attendance is already stopped."
fi
