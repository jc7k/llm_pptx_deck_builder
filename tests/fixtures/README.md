# Test Fixtures

This directory contains test assets for the llm_pptx_deck_builder test suite.

## Directory Structure

```
fixtures/
├── templates/     # PowerPoint template files for testing
│   └── *.pptx    # Place test template files here
└── README.md     # This file
```

## Templates Directory

The `templates/` subdirectory contains PowerPoint template files used for testing the template functionality:

- Place your `.pptx` template files here
- Templates should include various slide layouts to test compatibility
- Templates can include custom themes, branding, and master slides

## Usage in Tests

Test files can reference these fixtures using:

```python
import os
from pathlib import Path

# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEMPLATES_DIR = FIXTURES_DIR / "templates"

# Use a template in tests
template_path = TEMPLATES_DIR / "corporate_template.pptx"
```

## Adding New Fixtures

When adding new test fixtures:
1. Place them in the appropriate subdirectory
2. Use descriptive names
3. Keep file sizes reasonable for version control
4. Document any special characteristics of the fixture