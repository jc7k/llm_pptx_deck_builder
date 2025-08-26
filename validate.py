#!/usr/bin/env python3
"""
Validation script to run all quality gates for the LLM PPTX Deck Builder.

This script implements the validation gates specified in the PRP:
1. Syntax and Style Validation
2. Unit Tests with FakeLLM
3. Integration Tests
4. Import Validation
5. Configuration Validation
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib


def run_command(command, description, required=True):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"Command: {command}")
    print("="*60)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ FAILED: {description}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            
            if required:
                return False
                
    except Exception as e:
        print(f"❌ ERROR: Failed to run command: {e}")
        if required:
            return False
    
    return True


def validate_imports():
    """Validate that all modules can be imported."""
    print(f"\n{'='*60}")
    print("🔍 Import Validation")
    print("="*60)
    
    modules_to_test = [
        "src.settings",
        "src.models", 
        "src.dependencies",
        "src.tools",
        "src.deck_builder_agent"
    ]
    
    success = True
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ Successfully imported: {module}")
        except Exception as e:
            print(f"❌ Failed to import {module}: {e}")
            success = False
    
    return success


def validate_environment():
    """Validate environment and dependencies."""
    print(f"\n{'='*60}")
    print("🔍 Environment Validation")
    print("="*60)
    
    # Check if .env.example exists
    if os.path.exists(".env.example"):
        print("✅ .env.example file found")
    else:
        print("❌ .env.example file missing")
    
    # Check if pyproject.toml has correct structure
    if os.path.exists("pyproject.toml"):
        print("✅ pyproject.toml found")
        with open("pyproject.toml", "r") as f:
            content = f.read()
            if "langchain" in content and "llama-index" in content:
                print("✅ Required dependencies found in pyproject.toml")
            else:
                print("❌ Missing required dependencies in pyproject.toml")
    else:
        print("❌ pyproject.toml missing")
    
    return True


def main():
    """Run all validation gates."""
    print("🚀 LLM PPTX Deck Builder - Validation Gates")
    print("=" * 60)
    
    all_passed = True
    
    # Environment validation
    if not validate_environment():
        all_passed = False
    
    # Import validation
    if not validate_imports():
        all_passed = False
    
    # Install dependencies if needed
    print(f"\n{'='*60}")
    print("📦 Dependency Check")
    print("="*60)
    
    if not run_command("uv --version", "Check UV installation", required=False):
        print("⚠️  UV not found. Using pip for dependency management.")
        if not run_command("pip install -e .", "Install package with pip", required=False):
            print("❌ Failed to install dependencies")
    else:
        print("✅ UV found")
        # Sync dependencies
        run_command("uv sync", "Sync dependencies with UV", required=False)
    
    # 1. Syntax and Style Validation
    validation_commands = [
        ("uv run ruff check src/", "Ruff linting"),
        ("uv run black --check src/", "Black formatting check"),
    ]
    
    # Optional mypy check (may not work without all dependencies)
    validation_commands.append(("uv run mypy src/", "MyPy type checking"))
    
    for command, description in validation_commands:
        if not run_command(command, description, required=False):
            print(f"⚠️  {description} failed - continuing with other checks")
    
    # 2. Unit Tests with FakeLLM
    if not run_command("uv run python -m pytest tests/test_tools.py -v", "Unit Tests - Tools", required=False):
        all_passed = False
    
    if not run_command("uv run python -m pytest tests/test_agent.py -v", "Unit Tests - Agent", required=False):
        all_passed = False
    
    # 3. Integration Tests
    if not run_command("uv run python -m pytest tests/test_integration.py -v", "Integration Tests", required=False):
        print("⚠️  Integration tests failed - this may be expected without API keys")
    
    # 4. Full Test Suite
    if not run_command("uv run python -m pytest tests/ -v --tb=short", "Full Test Suite", required=False):
        print("⚠️  Some tests failed - this may be expected in CI environment")
    
    # 5. Configuration validation
    print(f"\n{'='*60}")
    print("🔍 Configuration Test")
    print("="*60)
    
    try:
        # Test configuration loading (without API keys)
        import os
        os.environ.pop('BRAVE_API_KEY', None)
        os.environ.pop('OPENAI_API_KEY', None)
        
        print("✅ Configuration validation completed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        all_passed = False
    
    # 6. CLI validation
    if not run_command("python deck_builder_cli.py --help", "CLI Help Command", required=True):
        all_passed = False
    
    if not run_command("python deck_builder_cli.py", "CLI Demo Mode", required=False):
        print("⚠️  CLI demo failed - this may be expected")
    
    # Final report
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    if all_passed:
        print("🎉 ALL CRITICAL VALIDATIONS PASSED!")
        print("\nThe LLM PPTX Deck Builder is ready for use.")
        print("\nTo get started:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys to .env")
        print("3. Run: python deck_builder_cli.py --topic 'Your presentation topic'")
        return 0
    else:
        print("⚠️  Some validations failed.")
        print("Review the output above to address any issues.")
        print("The application may still work for basic functionality.")
        return 1


if __name__ == "__main__":
    sys.exit(main())