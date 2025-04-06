# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Format: `black main.py`
- Type check: `mypy main.py`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_file.py::test_function`
- Run application: `python main.py`

## Code Style Guidelines

- Formatting: Use Black with default settings
- Imports: Group imports by standard library, third-party, and local
- Types: Use type annotations for all functions and classes
- Naming: 
  - snake_case for functions/variables
  - CamelCase for classes
  - ALL_CAPS for constants
- Error handling: Use explicit try/except blocks with specific exceptions
- Line length: 88 characters (Black default)
- Documentation: Include docstrings for all functions, classes, and modules
- Environment: Use python-dotenv for environment variables

## Dependencies

- LangChain with Anthropic for LLM integration
- LangGraph for building the conversational flow
- Use IPython for visualization capabilities