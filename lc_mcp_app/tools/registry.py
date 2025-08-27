from __future__ import annotations

import time
from typing import Any, Optional

from ..clients.mcp_client import MCPClientError, get_mcp_client
from ..middleware.metrics import record_error, record_tool_execution
from ..observability.logging import get_logger
from .superconductivity_calculator import SuperconductivityCalculatorTool

logger = get_logger("tools.registry")


class ToolRegistry:
    """Registry for managing tools and their capabilities."""
    
    def __init__(self):
        self.tools: dict[str, Any] = {
            "superconductivity_calculator": SuperconductivityCalculatorTool(),
        }
        self._capabilities: Optional[list[dict]] = None
    
    def get(self, name: str) -> Any:
        """Get a tool by name."""
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        return tool
    
    def get_tool_info(self) -> dict[str, Any]:
        """Get information about registered tools."""
        return {
            "count": len(self.tools),
            "tools": [{"name": name} for name in self.tools.keys()]
        }
    
    async def get_capabilities(self) -> list[dict]:
        """Get capabilities from MCP server."""
        if self._capabilities is None:
            try:
                mcp_client = await get_mcp_client()
                # This would normally fetch from MCP server
                # For now, return a placeholder
                self._capabilities = [
                    {
                        "type": "tool",
                        "name": name,
                        "description": f"Tool: {name}",
                        "inputSchema": {}
                    }
                    for name in self.tools.keys()
                ]
            except Exception as e:
                logger.error("Failed to get capabilities", error=str(e))
                self._capabilities = []
        return self._capabilities


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


async def initialize_tools() -> None:
    """Initialize tools and registry."""
    registry = get_registry()
    logger.info("Initializing tools", tool_count=len(registry.tools))
    
    # Perform any async initialization needed
    try:
        # Pre-fetch capabilities
        await registry.get_capabilities()
        logger.info("Tools initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize tools", error=str(e))
        # Don't fail startup if tool init fails
        pass


def get_tool(name: str) -> Any:
    """Get a tool by name (backward compatibility)."""
    return get_registry().get(name)


def execute_tool(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered tool and record metrics."""
    t0 = time.time()
    try:
        result = get_tool(name)(params)
        record_tool_execution(name, "success", time.time() - t0)
        return result
    except MCPClientError:
        record_error("MCPClientError", "tool")
        logger.exception("Tool execution failed: %s", name)
        raise
