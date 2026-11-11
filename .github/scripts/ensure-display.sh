set -euo pipefail
if ! command -v xvfb-run || ! command -v xauth || ! python -c 'import tkinter'; then
    timeout 60s sudo apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=15 update || true
    timeout 90s sudo apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=15 install -y xvfb xauth python3-tk
fi
command -v xvfb-run
command -v xauth
xvfb-run -a python -c 'import tkinter as tk; root = tk.Tk(); root.destroy()'
