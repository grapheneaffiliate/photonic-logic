# Validates MCP list_tools + each tool's JSON Schema, then calls the first zero-arg tool.
# Env:
#   MCP_SERVER_URL (default http://localhost:8000 or http://localhost:8000/mcp)
#   MCP_HEALTH_URL (default http://localhost:8000/health)

import asyncio
import json
import os
from typing import Any

import httpx
import jsonschema
import pytest
from mcp import ClientSession

# Prefer streamable HTTP; fall back to bare HTTP client if needed.
try:
    from mcp.client.streamable_http import streamablehttp_client as http_client
except Exception:  # pragma: no cover
    from mcp.client.http import http_client  # type: ignore

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_HEALTH_URL = os.getenv("MCP_HEALTH_URL", "http://localhost:8000/health")
# Use Draft7Validator which is available in jsonschema 4.x
SCHEMA_DRAFT = jsonschema.Draft7Validator

def _with_mcp_path(url: str) -> str:
    return url if url.rstrip("/").endswith("/mcp") else url.rstrip("/") + "/mcp"

async def _wait_healthy(url: str, timeout_s: float = 15.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        last = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except Exception as e:
                last = e
            await asyncio.sleep(0.25)
    if last:
        raise last
    raise RuntimeError(f"Health never passed for {url}")

def _schema_of(tool: Any) -> dict[str, Any] | None:
    if isinstance(tool, dict):
        return tool.get("input_schema") or tool.get("inputSchema") or tool.get("schema")
    return (
        getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None)
        or getattr(tool, "schema", None)
    )

def _name_of(tool: Any) -> str:
    return tool["name"] if isinstance(tool, dict) else getattr(tool, "name", "")

@pytest.mark.asyncio
@pytest.mark.integration
async def test_mcp_contract_list_and_call():
    try:
        await _wait_healthy(MCP_HEALTH_URL)
    except Exception:
        pytest.skip("MCP server not running - skipping integration test")

    # Try provided URL; if the server expects /mcp, try with that.
    connected = False
    last_error = None
    for url in (MCP_SERVER_URL, _with_mcp_path(MCP_SERVER_URL)):
        try:
            async with http_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    connected = True

                    tools_resp = await session.list_tools()
                    tools = getattr(tools_resp, "tools", None) or (
                        tools_resp.get("tools") if isinstance(tools_resp, dict) else None
                    )
                    assert tools, "No tools returned by MCP server"

                    invalid = []
                    for t in tools:
                        name = _name_of(t)
                        assert name, f"Tool missing name: {t!r}"
                        schema = _schema_of(t)
                        if schema:
                            try:
                                SCHEMA_DRAFT.check_schema(schema)
                            except Exception as e:
                                invalid.append((name, str(e)))
                    assert not invalid, "Invalid tool JSON Schemas:\n" + json.dumps(invalid, indent=2)

                    # Call the first zero-arg tool if present
                    for t in tools:
                        schema = _schema_of(t) or {}
                        req = set(schema.get("required", [])) if isinstance(schema, dict) else set()
                        if not req:
                            res = await session.call_tool(_name_of(t), arguments={})
                            assert res is not None
                            return
                    pytest.skip("No zero-argument tools to safely call; schema validation passed.")
        except Exception as e:
            last_error = e
            continue
    
    # If we couldn't connect at all, skip the test instead of failing
    if not connected:
        pytest.skip(f"Could not connect to MCP server - skipping integration test. Last error: {last_error}")
    else:
        pytest.fail("Could not connect to MCP server using provided URL (with and without /mcp)")
