# Contributing to Surveillance Enhancement System

Thank you for your interest in contributing! This document provides guidelines for contributions.

## Code of Conduct

Be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a detailed issue with:
   - System information
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable

### Suggesting Features

1. Check existing feature requests
2. Create an issue with:
   - Clear description of the feature
   - Use cases
   - Potential implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

### Testing

```bash
pytest tests/
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions
- Update API documentation if needed

## Development Setup

```bash
git clone https://github.com/pandoradiscoverer/surveillance-enhancement.git
cd surveillance-enhancement
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements-dev.txt
pre-commit install