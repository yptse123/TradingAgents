#!/usr/bin/env python3
"""
Run the Portfolio Selection Dashboard

Usage:
    python run_dashboard.py
    # or
    streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)

    print("Starting Portfolio Selection Dashboard...")
    print("Open http://localhost:8501 in your browser")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false",
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except FileNotFoundError:
        print("Error: streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
