#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")" && pwd)"
data_directory="$project_root/data"
pid_file="$data_directory/streamlit-process.pid"
stdout_log="$data_directory/streamlit-output.log"
stderr_log="$data_directory/streamlit-error.log"
app_url="http://127.0.0.1:8501"

mkdir -p "$data_directory"

if [[ -f "$pid_file" ]]; then
    existing_pid="$(tr -cd '0-9' < "$pid_file")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Fingerprint Attendance is already running at $app_url"
        open "$app_url"
        exit 0
    fi
    rm -f "$pid_file"
fi

if lsof -nP -iTCP:8501 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port 8501 is already in use. No duplicate server was started."
    open "$app_url"
    exit 0
fi

if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_command="$project_root/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_command="$(command -v python3)"
else
    echo "Python 3 was not found. Install Python, then follow the macOS setup in README.md."
    read -r -p "Press Return to close..."
    exit 1
fi

if ! "$python_command" -c "import streamlit" >/dev/null 2>&1; then
    echo "Streamlit is not installed for $python_command."
    echo "Run: python3 -m pip install -r requirements.txt"
    read -r -p "Press Return to close..."
    exit 1
fi

nohup "$python_command" -m streamlit run "$project_root/app.py" \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false \
    >"$stdout_log" 2>"$stderr_log" &
server_pid=$!
printf '%s\n' "$server_pid" > "$pid_file"

ready=false
for _ in {1..40}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        break
    fi
    if curl --silent --fail --max-time 1 "$app_url" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 0.25
done

if [[ "$ready" != true ]]; then
    rm -f "$pid_file"
    echo "The app did not start. Check data/streamlit-error.log for details."
    read -r -p "Press Return to close..."
    exit 1
fi

echo "Fingerprint Attendance started at $app_url"
open "$app_url"
