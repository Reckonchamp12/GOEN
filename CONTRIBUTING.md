# Contributing to GOEN

Thank you for your interest in contributing! Please follow these guidelines.

## Development Setup

```bash
git clone https://github.com/your-org/goen.git
cd goen
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Code Style

- Follow PEP 8.
- Use type annotations for all public functions.
- Write docstrings for all public classes and methods (NumPy style).
- Keep line length ≤ 100 characters.

## Submitting Changes

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes with a clear message.
4. Push and open a pull request against `main`.

## Reporting Issues

Please include:
- Python and PyTorch versions.
- Full traceback.
- Minimal reproducible example.
