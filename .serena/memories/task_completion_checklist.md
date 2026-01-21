# Task Completion Checklist

When completing a task in this project, ensure:

## Before Committing
1. **Run tests**: `pytest -q` - All tests should pass
2. **Manual test**: If UI changes, run `python app.py` and verify visually
3. **Check for regressions**: If modifying scanner/OCR, consider running `python e2e_test.py test_vids`

## Code Quality
- Configuration changes go in `config.py`, not inline
- Database operations use the connection pattern (`get_db_connection()`)
- Invalidate caches when modifying data (e.g., `_invalidate_known_people_cache()`)
- Handle graceful shutdown if adding long-running processes

## No Linting/Formatting Required
- Project does not have explicit linting/formatting tools configured
- Follow existing code style by example

## Testing Changes
- Add tests for new functionality in appropriate `tests/test_*.py` file
- Use existing fixtures from `tests/conftest.py` when applicable
- For UI changes, consider adding Playwright E2E tests in `tests/test_e2e_ui.py`
