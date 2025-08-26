#!/usr/bin/env python3
"""Test the improved content generation quality."""

import json

# Define the validation function inline for testing
def validate_slide_content(slide_data):
    """Validate slide content quality."""
    generic_phrases = [
        "key aspect", "important consideration", "implications and next steps",
        "key point", "main topic", "various factors", "different aspects",
        "several elements", "multiple components", "diverse range"
    ]
    
    bullets = slide_data.get("bullets", [])
    
    if len(bullets) < 2:
        return False
    
    for bullet in bullets:
        bullet_lower = bullet.lower()
        
        if any(phrase in bullet_lower for phrase in generic_phrases):
            return False
        
        has_specifics = (
            any(char.isdigit() for char in bullet) or
            "%" in bullet or
            "$" in bullet or
            len(bullet.split()) > 5
        )
        
        if not has_specifics and len(bullet) < 30:
            return False
    
    return True

def test_validation():
    """Test the content quality validator."""
    
    # Test case 1: Generic content (should fail)
    generic_slide = {
        "title": "Key Concepts",
        "bullets": [
            "Key aspect of Key Concepts",
            "Important considerations for Key Concepts",
            "Implications and next steps"
        ]
    }
    
    # Test case 2: Good content with specifics (should pass)
    good_slide = {
        "title": "AI Job Market Impact 2025",
        "bullets": [
            "AI projected to create 170 million new jobs by 2030, offsetting 85 million displaced roles",
            "75% of knowledge workers already using AI tools, with 46% adoption in last 6 months",
            "Workers with AI skills earn 56% wage premium compared to non-AI skilled peers",
            "Healthcare and finance sectors see 40% productivity gains from AI augmentation"
        ]
    }
    
    # Test case 3: Mixed quality (should fail due to generic content)
    mixed_slide = {
        "title": "Current Trends",
        "bullets": [
            "AI adoption increased by 250% in 2024",
            "Various factors affecting implementation",  # Generic
            "Key aspects to consider"  # Generic
        ]
    }
    
    # Test case 4: Short but specific (should pass)
    short_good = {
        "title": "Market Size",
        "bullets": [
            "Global AI market valued at $500 billion in 2025",
            "Expected CAGR of 38% through 2030"
        ]
    }
    
    print("Testing content quality validation:")
    print("-" * 50)
    
    test_cases = [
        ("Generic content", generic_slide, False),
        ("Good specific content", good_slide, True),
        ("Mixed quality", mixed_slide, False),
        ("Short but specific", short_good, True)
    ]
    
    for name, slide, expected in test_cases:
        result = validate_slide_content(slide)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"{status} - {name}: Expected {expected}, got {result}")
        if result != expected:
            print(f"  Bullets: {slide['bullets'][:2]}")
    
    print("\n" + "=" * 50)
    print("Validation test complete!")

if __name__ == "__main__":
    test_validation()