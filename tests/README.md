# SLICES Test Suite

This directory contains unit tests and integration tests for the SLICES project.

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pytest tests/ --cov=slices --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/test_data_io.py -v
```

### Run specific test class
```bash
pytest tests/test_data_io.py::TestCsvToParquetAll -v
```

### Run specific test function
```bash
pytest tests/test_data_io.py::TestCsvToParquetAll::test_successful_conversion -v
```

### Run tests matching a pattern
```bash
pytest tests/ -k "conversion" -v
```

### Run with verbose output
```bash
pytest tests/ -vv
```

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_placeholder.py` - Basic sanity tests
- `test_data_io.py` - Tests for CSV-to-Parquet conversion
- `test_extractor_config.py` - Tests for ExtractorConfig dataclass
- `test_base_extractor.py` - Tests for BaseExtractor abstract class

## Writing New Tests

### Test File Naming
- Test files must start with `test_` or end with `_test.py`
- Test functions must start with `test_`
- Test classes must start with `Test`

### Using Fixtures
Fixtures from `conftest.py` are automatically available:

```python
def test_something(sample_batch):
    # sample_batch fixture is automatically injected
    assert sample_batch["timeseries"].shape[0] == 4
```

### Example Test Structure
```python
class TestMyFeature:
    """Test suite for MyFeature."""
    
    def test_basic_functionality(self):
        """Test basic case."""
        result = my_function()
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## Test Coverage

To view test coverage in browser:
```bash
pytest tests/ --cov=slices --cov-report=html
open htmlcov/index.html  # macOS
```

Target: >80% coverage on core modules
