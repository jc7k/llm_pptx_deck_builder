#!/usr/bin/env python3
"""
Main CLI interface for the LLM PPTX Deck Builder.

This script provides a command-line interface for generating PowerPoint presentations
using AI-powered research and content generation.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables early to prevent LangChain warnings
load_dotenv()
if not os.environ.get("USER_AGENT") and os.environ.get("user_agent"):
    os.environ["USER_AGENT"] = os.environ["user_agent"]
elif not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "LLM-PPTX-Deck-Builder/1.0"

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate AI-powered PowerPoint presentations with web research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deck_builder_cli.py --topic "AI in Education trends for 2024"
  python deck_builder_cli.py --topic "Cybersecurity best practices" --template corporate.pptx
  python deck_builder_cli.py --topic "Climate change impacts" --output climate_presentation.pptx
        """
    )
    
    parser.add_argument(
        "--topic",
        help="Topic or description for the presentation"
    )
    
    parser.add_argument(
        "--template",
        help="Path to PowerPoint template file (.pptx)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path for the presentation"
    )
    
    parser.add_argument(
        "--max-slides", 
        type=int,
        default=12,
        help="Maximum number of content slides to generate (default: 12)"
    )
    
    parser.add_argument(
        "--audience",
        help="Target audience for the presentation"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        help="Expected presentation duration in minutes"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and API keys"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.validate_only and not args.topic:
        parser.error("--topic is required unless using --validate-only")
    
    # Import here to avoid config errors during help
    try:
        from src.dependencies import validate_api_keys
        from src.settings import settings
        from src.deck_builder_agent import build_deck_sync
    except ImportError as e:
        print(f"❌ Import Error: {e}", file=sys.stderr)
        print("Please run 'uv sync' to install dependencies")
        return 1
    
    # Validate API keys
    try:
        validate_api_keys()
        if args.verbose:
            print("✓ API keys validated successfully")
    except ValueError as e:
        print(f"❌ Configuration Error: {e}", file=sys.stderr)
        print("\nPlease ensure you have set the following environment variables:")
        print("- BRAVE_API_KEY: Your Brave Search API key")
        print("- OPENAI_API_KEY: Your OpenAI API key")
        print("\nYou can create a .env file based on .env.example")
        return 1
    except Exception as e:
        print(f"❌ Configuration Error: {e}", file=sys.stderr)
        return 1
    
    if args.validate_only:
        print("✓ Configuration is valid. All API keys are present.")
        return 0
    
    # Validate template file if provided
    if args.template:
        if not os.path.exists(args.template):
            print(f"❌ Template file not found: {args.template}", file=sys.stderr)
            return 1
        if not args.template.lower().endswith('.pptx'):
            print(f"❌ Template file must be a .pptx file: {args.template}", file=sys.stderr)
            return 1
        if args.verbose:
            print(f"✓ Template file found: {args.template}")
    
    # Build the presentation
    print(f"🚀 Starting presentation generation for: {args.topic}")
    print("This may take several minutes...")
    
    if args.verbose:
        print(f"Configuration:")
        print(f"  - Topic: {args.topic}")
        print(f"  - Template: {args.template or 'Default'}")
        print(f"  - Output: {args.output or 'Auto-generated'}")
        print(f"  - Max search results: {settings.max_search_results}")
        print(f"  - Max documents: {settings.max_documents}")
    
    try:
        # Set up verbose progress callback
        if args.verbose:
            from src.deck_builder_agent import set_progress_callback
            
            def progress_callback(step: str, message: str):
                print(f"[{time.strftime('%H:%M:%S')}] {step}: {message}", flush=True)
            
            set_progress_callback(progress_callback)
        
        result = build_deck_sync(
            user_request=args.topic,
            template_path=args.template,
            output_path=args.output
        )
        
        if result["success"]:
            print("✅ Presentation generated successfully!")
            print(f"📄 Output file: {result['output_path']}")
            print(f"📊 Slides created: {result['slide_count']}")
            print(f"📚 References included: {result['references_count']}")
            
            if args.verbose and result.get("messages"):
                print("\nWorkflow steps completed:")
                for i, message in enumerate(result["messages"], 1):
                    print(f"  {i}. {message}")
            
            return 0
            
        else:
            print("❌ Presentation generation failed!")
            print(f"Error: {result.get('error_message', 'Unknown error')}")
            print(f"Status: {result.get('status', 'Unknown status')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def demo():
    """Run a demo with sample content."""
    print("🎯 Running LLM PPTX Deck Builder Demo")
    print("=" * 50)
    
    demo_topics = [
        "Artificial Intelligence trends in 2024",
        "Remote work best practices",
        "Sustainable technology solutions"
    ]
    
    print("Sample topics you could try:")
    for i, topic in enumerate(demo_topics, 1):
        print(f"  {i}. {topic}")
    
    print("\nTo generate a presentation, run:")
    print(f'python deck_builder_cli.py --topic "AI in Education trends for 2024"')
    print("\nMake sure you have set your API keys in a .env file:")
    print("cp .env.example .env")
    print("# Edit .env with your actual API keys")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show demo
        demo()
        sys.exit(0)
    else:
        sys.exit(main())