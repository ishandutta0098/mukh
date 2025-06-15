# Installation

## Requirements

Mukh requires Python 3.10 or later. Make sure you have a compatible version installed:

```bash
python --version
```

## Installation via pip

The recommended way to install Mukh is using pip:

```bash
pip install mukh
```

## Development Installation

If you want to contribute to Mukh or use the latest development version, you can install it from source:

```bash
# Clone the repository
git clone https://github.com/ishandutta0098/mukh.git
cd mukh

# Install in development mode
pip install -e ".[dev]" --use-pep517
```

## Verify Installation

After installation, verify that Mukh is working correctly:

```python
import mukh
print(mukh.__version__)
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'mukh'`

**Solution**: Make sure you've installed the package correctly:
```bash
pip list | grep mukh
```

### Getting Help

If you encounter any issues during installation:

1. Check the [GitHub Issues](https://github.com/ishandutta0098/mukh/issues)
2. Create a new issue with your system information and error details

## Next Steps

Once you have Mukh installed, check out the [Quick Start Guide](quick-start.md) to begin using the library. 