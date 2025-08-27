@echo off
echo ========================================
echo Starting MCP Test Environment
echo ========================================

REM Start MCP Agent Server in new window
echo Starting MCP Agent Server (port 8000)...
start "MCP Agent Server" cmd /k "python -m mcp_agent.server"

REM Start LangChain MCP App Server in new window  
echo Starting LangChain MCP App Server (port 8001)...
start "LangChain MCP Server" cmd /k "python -m lc_mcp_app.server"

REM Wait for servers to start
echo Waiting for servers to start...
timeout /t 5 /nobreak > nul

REM Run tests
echo.
echo ========================================
echo Running Tests
echo ========================================
echo.

REM Run core tests that should pass
echo Running core server tests...
pytest tests/mcp_server/test_server.py -v --tb=short

echo.
echo Running health endpoint tests...
pytest tests/test_health.py -v --tb=short

echo.
echo Running OpenAI API tests...
pytest tests/test_openai_api.py -v --tb=short

echo.
echo ========================================
echo Test run complete!
echo ========================================
echo.
echo Servers are still running in separate windows.
echo Close them manually when done.
echo.
pause
