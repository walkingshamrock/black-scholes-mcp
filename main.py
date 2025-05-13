import math
import json
from typing import Literal, Annotated, cast, Any, List
from pydantic import Field

ToolAnnotations = Any

from fastmcp import FastMCP

# Import the calculator registration functions
from calculators.black_scholes_price import register_price_tool
from calculators.black_scholes_delta import register_delta_tool
from calculators.black_scholes_gamma import register_gamma_tool
from calculators.black_scholes_vega import register_vega_tool
from calculators.black_scholes_theta import register_theta_tool
from calculators.black_scholes_rho import register_rho_tool
from calculators.black_scholes_vanna import register_vanna_tool
from calculators.black_scholes_charm import register_charm_tool
from calculators.black_scholes_lambda import register_lambda_tool
from calculators.black_scholes_epsilon import register_epsilon_tool
from calculators.black_scholes_vomma import register_vomma_tool
from calculators.black_scholes_veta import register_veta_tool
from calculators.black_scholes_speed import register_speed_tool
from calculators.black_scholes_zomma import register_zomma_tool
from calculators.black_scholes_color import register_color_tool
from calculators.black_scholes_ultima import register_ultima_tool
from calculators.black_scholes_vera import register_vera_tool

# Initialize FastMCP instance
mcp = FastMCP("black-scholes")

def register_all_tools() -> List:
    """Register all Black-Scholes calculation tools with the MCP server."""
    registered_tools = []
    
    # Register the option pricing tool
    price_tool = register_price_tool(mcp)
    registered_tools.append(price_tool)
    
    # Register the first-order Greeks
    delta_tool = register_delta_tool(mcp)
    registered_tools.append(delta_tool)
    
    gamma_tool = register_gamma_tool(mcp)
    registered_tools.append(gamma_tool)
    
    vega_tool = register_vega_tool(mcp)
    registered_tools.append(vega_tool)
    
    theta_tool = register_theta_tool(mcp)
    registered_tools.append(theta_tool)
    
    rho_tool = register_rho_tool(mcp)
    registered_tools.append(rho_tool)
    
    # Register higher-order Greeks
    vanna_tool = register_vanna_tool(mcp)
    registered_tools.append(vanna_tool)
    
    charm_tool = register_charm_tool(mcp)
    registered_tools.append(charm_tool)
    
    # Register additional Greek calculators
    lambda_tool = register_lambda_tool(mcp)
    registered_tools.append(lambda_tool)
    
    epsilon_tool = register_epsilon_tool(mcp)
    registered_tools.append(epsilon_tool)
    
    vomma_tool = register_vomma_tool(mcp)
    registered_tools.append(vomma_tool)
    
    veta_tool = register_veta_tool(mcp)
    registered_tools.append(veta_tool)
    
    speed_tool = register_speed_tool(mcp)
    registered_tools.append(speed_tool)
    
    zomma_tool = register_zomma_tool(mcp)
    registered_tools.append(zomma_tool)
    
    color_tool = register_color_tool(mcp)
    registered_tools.append(color_tool)
    
    ultima_tool = register_ultima_tool(mcp)
    registered_tools.append(ultima_tool)
    
    vera_tool = register_vera_tool(mcp)
    registered_tools.append(vera_tool)
    
    return registered_tools

# Register all tools
tools = register_all_tools()

if __name__ == "__main__":
    mcp.run()
