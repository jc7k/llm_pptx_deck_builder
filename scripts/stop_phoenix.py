#!/usr/bin/env python3
"""
Stop the standalone Phoenix server that was started with start_phoenix.py.
Provides a clean way to shut down the persistent Phoenix server.
"""

import os
import signal
import subprocess
import sys
import time
import requests
from pathlib import Path


def find_phoenix_processes():
    """Find all running Phoenix processes."""
    try:
        # Find processes running start_phoenix.py
        result = subprocess.run(
            ["pgrep", "-f", "start_phoenix.py"],
            capture_output=True,
            text=True
        )
        
        pids = []
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
        
        return pids
    except Exception as e:
        print(f"⚠️  Error finding Phoenix processes: {e}")
        return []


def check_phoenix_server(host="127.0.0.1", port="6006"):
    """Check if Phoenix server is responding."""
    try:
        response = requests.get(f"http://{host}:{port}", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    print("🛑 Stopping Phoenix server...")
    
    # Get Phoenix configuration
    host = os.getenv("PHOENIX_HOST", "127.0.0.1")
    port = os.getenv("PHOENIX_PORT", "6006")
    
    # Check if server is running
    if not check_phoenix_server(host, port):
        print(f"ℹ️  No Phoenix server found at http://{host}:{port}")
        return
    
    # Find Phoenix processes
    pids = find_phoenix_processes()
    
    if not pids:
        print("ℹ️  No Phoenix processes found")
        print("💡 You can also try: pkill -f 'start_phoenix.py'")
        return
    
    print(f"🔍 Found {len(pids)} Phoenix process(es): {', '.join(map(str, pids))}")
    
    # Try graceful shutdown first (SIGTERM)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"📤 Sent SIGTERM to process {pid}")
        except ProcessLookupError:
            print(f"⚠️  Process {pid} not found")
        except PermissionError:
            print(f"❌ Permission denied to kill process {pid}")
    
    # Wait for graceful shutdown
    print("⏳ Waiting for graceful shutdown...")
    time.sleep(3)
    
    # Check if server is still running
    if check_phoenix_server(host, port):
        print("⚠️  Server still running, trying force kill (SIGKILL)...")
        
        # Force kill if graceful shutdown failed
        remaining_pids = find_phoenix_processes()
        for pid in remaining_pids:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"💥 Force killed process {pid}")
            except ProcessLookupError:
                print(f"⚠️  Process {pid} already terminated")
            except PermissionError:
                print(f"❌ Permission denied to force kill process {pid}")
        
        time.sleep(1)
    
    # Final check
    if check_phoenix_server(host, port):
        print("❌ Phoenix server is still running")
        print("💡 Try manual cleanup:")
        print("   - pkill -f 'start_phoenix.py'")
        print("   - pkill -f 'phoenix'")
        print(f"   - Check processes: ps aux | grep phoenix")
        sys.exit(1)
    else:
        print("✅ Phoenix server stopped successfully")
        
        # Show data directory info
        phoenix_dir = Path.home() / ".phoenix_llm_pptx"
        if phoenix_dir.exists():
            print(f"💾 Traces are preserved in: {phoenix_dir}")
            print("🔄 Restart server with: uv run python scripts/start_phoenix.py")


if __name__ == "__main__":
    main()