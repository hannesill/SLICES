# SLICES Test Suite

Comprehensive test suite for the SLICES project, covering data extraction, processing, and task building.

## Running Tests

### Run all tests
```bash
uv run pytest tests/ -v
```

### Run with coverage report
```bash
uv run pytest tests/ --cov=slices --cov-report=html --cov-report=term
```

### Run specific test file
```bash
uv run pytest tests/test_data_io.py -v
```

### Run specific test class
```bash
uv run pytest tests/test_data_io.py::TestCsvToParquetAll -v
```

### Run specific test function
```bash
uv run pytest tests/test_data_io.py::TestCsvToParquetAll::test_successful_conversion -v
```

### Run tests matching a pattern
```bash
uv run pytest tests/ -k "mortality" -v
```

### Run with verbose output
```bash
uv run pytest tests/ -vv
```

## Test Structure

| File | Purpose | Coverage |
|------|---------|----------|
| `conftest.py` | Shared fixtures and pytest configuration | - |
| `test_package.py` | Package structure, imports, and dependencies | Package-level validation |
| `test_extractor_config.py` | ExtractorConfig validation | Input validation, defaults |
| `test_base_extractor.py` | BaseExtractor abstract class | Core extraction logic, `run()` method, dense conversion |
| `test_data_io.py` | CSV-to-Parquet conversion | File I/O, data integrity |
| `test_dataset_datamodule.py` | ICUDataset and ICUDataModule | Data loading, imputation, normalization, patient-level splits |
| `test_extractor_integration.py` | Extractor + task system integration | Multi-task extraction, label computation |
| `test_task_builders.py` | Task label extraction | Mortality tasks, boundary conditions |
| `test_timeseries_extraction.py` | Time-series extraction pipeline | Hourly binning, feature mapping, edge cases |

## Test Categories

### Unit Tests
- `test_extractor_config.py` - Configuration validation
- `test_base_extractor.py` - Core extractor methods
- `test_data_io.py` - Data I/O utilities
- `test_task_builders.py` - Task builders

### Integration Tests
- `test_extractor_integration.py` - Full extraction pipeline
- `test_dataset_datamodule.py` - Data loading pipeline

### Edge Case Tests
- `test_timeseries_extraction.py::TestTimeSeriesEdgeCases` - Empty data, extreme values
- `test_task_builders.py::TestMortalityBoundaryConditions` - Boundary conditions

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

    @pytest.fixture
    def setup_data(self, tmp_path):
        """Create test data."""
        # Setup code
        return data

    def test_basic_functionality(self, setup_data):
        """Test basic case."""
        result = my_function(setup_data)
        assert result == expected

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError, match="expected error"):
            my_function(invalid_input)

    def test_boundary_condition(self, setup_data):
        """Test boundary condition."""
        result = my_function(setup_data, boundary_value)
        assert result == expected_boundary_result
```

### Best Practices
1. **Test one thing per test** - Each test should verify a single behavior
2. **Use descriptive names** - Test names should describe what they test
3. **Include edge cases** - Test boundary conditions, empty inputs, extreme values
4. **Use fixtures** - Share setup code via fixtures
5. **Mock external dependencies** - Use `unittest.mock.patch` for external calls
6. **Test error handling** - Verify exceptions are raised correctly

## Test Coverage

To view test coverage in browser:
```bash
uv run pytest tests/ --cov=slices --cov-report=html
open htmlcov/index.html  # macOS
```

**Target: >80% coverage on core modules**

## Key Test Scenarios

### Data Extraction
- Valid/invalid parquet paths
- Feature mapping loading
- Hourly binning and aggregation
- Negative hour filtering
- Multiple itemids per feature

### Data Processing
- Imputation strategies (forward_fill, zero, mean, none)
- Normalization
- Dense timeseries conversion
- Observation mask handling

### Task Building
- Mortality prediction windows (24h, 48h, hospital, ICU)
- Boundary conditions (exact boundaries)
- Empty data handling
- Multiple tasks extraction

### Data Loading
- Patient-level splits (no leakage)
- Reproducible splits with seeds
- Batch collation
- DataLoader configuration
