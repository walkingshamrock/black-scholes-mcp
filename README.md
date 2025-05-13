
# Black-Scholes MCP Server

This project provides a Model Context Protocol (MCP) server for calculating the price and Greeks of European options using the Black-Scholes model.

## Features

- Calculate the Black-Scholes price for European call and put options
- Compute option Greeks and higher-order Greeks:
  - Delta
  - Vega
  - Theta
  - Gamma
  - Rho
  - Lambda
  - Epsilon
  - Vanna
  - Charm
  - Vomma
  - Veta
  - Speed
  - Zomma
  - Color
  - Ultima
  - Vera

## Usage

### Installation and Usage

1. Install dependencies (if using `uv`):
   ```sh
   uv pip install -r requirements.txt
   ```
   Or use your preferred Python package manager.

2. Install this MCP server to Claude:
   ```sh
   uv run mcp install main.py
   ```
   This command will add the configuration to `claude_desktop_config.json` so that Claude can use this MCP server.

3. (Optional) Run the MCP server directly:
   ```sh
   python main.py
   ```

4. Use the MCP tools to calculate option prices and Greeks by providing the following arguments:
   - `S`: Spot price
   - `K`: Strike price
   - `T`: Time to maturity (in years)
   - `r`: Risk-free rate (annual, decimal)
   - `q`: Dividend yield (annual, decimal)
   - `vol`: Volatility (annual, decimal)
   - `type`: "call" or "put"

## Running Tests

To run the tests for this project:

1. Install the package in development mode:
   ```sh
   pip install -e .
   ```

2. Run tests using unittest:
   ```sh
   python -m unittest discover -s tests
   ```

   Or with pytest (after installing pytest from requirements.txt):
   ```sh
   python -m pytest
   ```

3. To run specific test modules:
   ```sh
   python -m unittest tests.calculators.test_black_scholes_price
   ```
   
   Or with pytest:
   ```sh
   python -m pytest tests/calculators/test_black_scholes_price.py
   ```

## Acknowledgements

This project uses the [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) for MCP server implementation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
