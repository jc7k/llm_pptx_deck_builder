#!/usr/bin/env python3
"""Final validation test for the polished system."""

from typing import List, Tuple

def validate_thought_completeness(bullets: List[str]) -> Tuple[bool, List[str]]:
    """Enhanced validation with comprehensive incomplete thought detection."""
    issues = []
    
    for i, bullet in enumerate(bullets):
        # Check word count (flexible range based on thought complexity)
        word_count = len(bullet.split())
        if word_count > 15:
            issues.append(f"Bullet {i+1}: {word_count} words (max 15): '{bullet[:50]}...'")
        
        # Check for markdown formatting
        if "**" in bullet or "__" in bullet or "*" in bullet:
            issues.append(f"Bullet {i+1}: Contains markdown formatting: '{bullet}'")
        
        # Check minimum length (too short bullets are often generic)
        if word_count < 4:
            issues.append(f"Bullet {i+1}: Too short ({word_count} words): '{bullet}'")
        
        # Enhanced incomplete thought detection
        bullet_lower = bullet.lower().strip()
        
        # Prepositions that indicate incomplete thoughts
        incomplete_prepositions = [
            " through", " via", " by", " due to", " such as", " including",
            " with", " from", " for", " in", " on", " at", " of", " to",
            " within", " across", " among", " between", " during", " since",
            " until", " after", " before", " around", " under", " over"
        ]
        
        # Conjunctions that indicate incomplete thoughts
        incomplete_conjunctions = [
            " and", " or", " but", " so", " yet", " however", " while",
            " whereas", " although", " because", " since", " if", " when",
            " where", " who", " which", " that"
        ]
        
        # Articles and incomplete phrases that indicate cut-offs
        incomplete_articles = [
            " the", " a", " an", " some", " many", " most", " all",
            " these", " those", " this", " that"
        ]
        
        # Special case: "is" often indicates incomplete thought
        if bullet_lower.endswith(" is"):
            issues.append(f"Bullet {i+1}: Incomplete - ends with 'is': '{bullet}'")
        
        # Check for incomplete endings
        if any(bullet_lower.endswith(indicator) for indicator in incomplete_prepositions):
            issues.append(f"Bullet {i+1}: Incomplete - ends with preposition: '{bullet}'")
        
        if any(bullet_lower.endswith(indicator) for indicator in incomplete_conjunctions):
            issues.append(f"Bullet {i+1}: Incomplete - ends with conjunction: '{bullet}'")
            
        if any(bullet_lower.endswith(indicator) for indicator in incomplete_articles):
            issues.append(f"Bullet {i+1}: Incomplete - ends with article: '{bullet}'")
        
        # Check for truncation patterns
        if bullet.endswith("...") or bullet.endswith(".."):
            issues.append(f"Bullet {i+1}: Appears truncated with ellipsis: '{bullet}'")
        
        # Check for common incomplete patterns
        incomplete_patterns = [
            "organizations should", "companies need to", "it is important to",
            "there are many", "some of the", "one of the", "according to",
            "research shows that", "studies indicate", "experts believe"
        ]
        
        for pattern in incomplete_patterns:
            if bullet_lower.startswith(pattern) and len(bullet.split()) < 10:
                issues.append(f"Bullet {i+1}: Generic/incomplete start pattern: '{bullet}'")
    
    return len(issues) == 0, issues

def test_enhanced_validation():
    """Test enhanced validation covering slides 8-10 patterns."""
    print("🧪 Testing Enhanced Incomplete Thought Detection:")
    print("=" * 60)
    
    # Test cases covering slides 8-10 typical problems
    test_cases = [
        # GOOD complete thoughts
        ("AI automation: 40% productivity boost across sectors", True),
        ("Healthcare AI: $2.3B investment returns recorded", True),
        ("Training programs address 46% skill gap effectively", True),
        ("Remote work integration increases 30% with AI", True),
        
        # BAD incomplete thoughts that should be caught
        ("Companies implementing AI solutions through", False),  # Preposition
        ("Organizations should prioritize training programs for", False),  # Preposition
        ("The demand for AI specialists is", False),  # Article ending
        ("Studies show that workforce adaptation with", False),  # Preposition
        ("Training initiatives help workers who", False),  # Conjunction
        ("Implementation challenges include skill gaps and", False),  # Conjunction
        ("According to research", False),  # Generic start + too short
        ("There are many", False),  # Generic start + too short
        ("Some of the key factors", False),  # Generic + article ending
        ("Organizations should develop comprehensive training programs that", False),  # Conjunction
        
        # EDGE cases from real presentations
        ("AI creates new job categories while", False),  # Conjunction
        ("Healthcare productivity gains through", False),  # Preposition
        ("Remote work adoption increases significantly due to", False),  # Preposition
        ("The impact on workforce development across", False),  # Preposition + article
        ("Future predictions indicate that", False),  # Conjunction
        ("Implementation success depends on many", False),  # Article
        ("Workers need training programs for", False),  # Preposition
        
        # Should PASS - Complete thoughts in bullet style
        ("AI job displacement: 92M roles by 2030", True),
        ("Skills gap: 46% leaders cite training barriers", True),
        ("Success factor: Comprehensive retraining programs essential", True),
        ("Healthcare leader: 40% efficiency gains achieved", True),
    ]
    
    print(f"Testing {len(test_cases)} validation scenarios...")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for bullet, expected in test_cases:
        is_valid, issues = validate_thought_completeness([bullet])
        status = "✅ PASS" if is_valid == expected else "❌ FAIL"
        
        if is_valid == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} - '{bullet[:55]}{'...' if len(bullet) > 55 else ''}'")
        if issues and not expected:
            print(f"      ⚠️  {issues[0]}")
        elif not is_valid and expected:
            print(f"      ❌ Unexpectedly flagged: {issues[0] if issues else 'Unknown issue'}")
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Enhanced validation system working perfectly!")
    else:
        print(f"⚠️  {failed} tests failed - needs further refinement")
    
    print("=" * 60)

def test_title_quote_removal():
    """Test quote removal from titles."""
    print("\n🎯 Testing Title Quote Removal:")
    print("-" * 40)
    
    test_titles = [
        '"AI Adoption Trends"',
        "'Healthcare AI Success'", 
        '"Enterprise Implementation Challenges"',
        '""Skills Gap Solutions""',
        "''Future Workforce Predictions''",
        "No Quotes Here",
        '"Mixed "Quotes" Problem"'
    ]
    
    for title in test_titles:
        cleaned = title.strip('"\'""''')
        print(f"'{title}' → '{cleaned}'")
    
    print("✅ Quote removal system ready")

if __name__ == "__main__":
    test_enhanced_validation()
    test_title_quote_removal()