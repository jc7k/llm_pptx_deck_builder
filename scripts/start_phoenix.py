#!/usr/bin/env python3
"""
Standalone Phoenix server launcher that runs independently of the main application.
This allows Phoenix to persist after CLI commands finish.
"""

import os
import sys
import signal
import atexit
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

def cleanup_handler(signum=None, frame=None):
    """Handle cleanup on exit."""
    print("\n🛑 Phoenix server shutting down...")
    sys.exit(0)

def main():
    print("🚀 Starting persistent Phoenix server...")
    
    # Set environment variables
    os.environ["PHOENIX_HOST"] = os.getenv("PHOENIX_HOST", "127.0.0.1")
    os.environ["PHOENIX_PORT"] = os.getenv("PHOENIX_PORT", "6006")
    os.environ["PHOENIX_GRPC_PORT"] = os.getenv("PHOENIX_GRPC_PORT", "4317")
    
    # Set Phoenix working directory for persistent storage
    phoenix_dir = Path.home() / ".phoenix_llm_pptx"
    phoenix_dir.mkdir(exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(phoenix_dir)
    
    print(f"📁 Phoenix data directory: {phoenix_dir}")
    
    try:
        import phoenix as px
        
        # Check if server is already running
        import requests
        try:
            response = requests.get(f"http://{os.environ['PHOENIX_HOST']}:{os.environ['PHOENIX_PORT']}", timeout=2)
            if response.status_code == 200:
                print(f"✓ Phoenix server already running at http://{os.environ['PHOENIX_HOST']}:{os.environ['PHOENIX_PORT']}")
                print("💡 Use Ctrl+C to stop the server")
                print("💡 Or run: pkill -f 'start_phoenix.py'")
                return
        except requests.RequestException:
            pass
        
        # Launch Phoenix server with persistent storage
        print(f"🌍 Starting Phoenix at http://{os.environ['PHOENIX_HOST']}:{os.environ['PHOENIX_PORT']}")
        
        # Phoenix automatically uses SQLite database when PHOENIX_WORKING_DIR is set
        session = px.launch_app()
        
        print(f"✅ Phoenix server started successfully!")
        print(f"🌐 Phoenix UI: {session.url}")
        print(f"💾 Traces persist in: {phoenix_dir}")
        print("📊 Dashboard accessible at all times")
        print("💡 Press Ctrl+C to stop the server")
        print("💡 Or run: pkill -f 'start_phoenix.py'")
        
        # Register cleanup handlers
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        atexit.register(lambda: print("👋 Phoenix server stopped"))
        
        # Keep server running
        try:
            while True:
                import time
                time.sleep(10)
        except KeyboardInterrupt:
            cleanup_handler()
            
    except ImportError:
        print("❌ Phoenix not available. Install with: uv sync")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start Phoenix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()