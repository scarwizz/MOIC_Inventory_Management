"""
main.py — MOIC Supply Chain Agent Unified Entry Point

Usage:
    python main.py           # Launch Streamlit dashboard
    python main.py --ui      # Launch Streamlit dashboard
    python main.py --cli     # Run interactive CLI agent
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def run_ui():
    """Launch Streamlit dashboard."""
    print("🚀 Starting MOIC Streamlit Intelligence Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")])


def run_cli():
    """Run interactive terminal agent."""
    print("🤖 Launching MOIC Terminal Agent...")
    subprocess.run([sys.executable, str(ROOT / "agent_core.py")])


def main():
    args = sys.argv[1:]
    if "--cli" in args:
        run_cli()
    elif "--help" in args or "-h" in args:
        print(__doc__)
    else:
        run_ui()


if __name__ == "__main__":
    main()
