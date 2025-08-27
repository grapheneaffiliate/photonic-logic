# PowerShell script to run MCP servers and tests
param(
    [switch]$Docker,
    [switch]$SkipSetup,
    [switch]$TestsOnly
)

Write-Host "MCP System Test Runner" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan

# Function to check if a port is in use
function Test-Port {
    param($Port)
    $connection = New-Object System.Net.Sockets.TcpClient
    try {
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Setup environment
if (-not $SkipSetup) {
    Write-Host "`nSetting up environment..." -ForegroundColor Yellow
    
    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        Write-Host "Creating .env file from example..."
        Copy-Item ".env.example" ".env"
    }
    
    # Install dependencies
    Write-Host "Installing dependencies..."
    pip install -e ".[dev]" 2>$null
    pip install responses jsonschema jsonrpcserver pydantic pydantic-settings 2>$null
}

if ($Docker) {
    Write-Host "`nUsing Docker Compose..." -ForegroundColor Yellow
    
    # Stop any existing containers
    docker-compose down 2>$null
    
    # Start services
    Write-Host "Starting services with Docker Compose..."
    docker-compose up -d --build
    
    # Wait for services to be ready
    Write-Host "Waiting for services to be ready..."
    $maxAttempts = 30
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        if ((Test-Port 8000) -and (Test-Port 8001)) {
            Write-Host "Services are ready!" -ForegroundColor Green
            break
        }
        Start-Sleep -Seconds 2
        $attempt++
        Write-Host "." -NoNewline
    }
    Write-Host ""
    
} elseif (-not $TestsOnly) {
    Write-Host "`nStarting servers manually..." -ForegroundColor Yellow
    
    # Kill any existing Python processes on our ports
    Write-Host "Cleaning up existing processes..."
    Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*mcp_agent.server*" -or $_.CommandLine -like "*lc_mcp_app.server*"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Start-Sleep -Seconds 2
    
    # Start MCP Agent Server
    Write-Host "Starting MCP Agent Server (port 8000)..."
    $mcp = Start-Process powershell -ArgumentList "-Command", "python -m mcp_agent.server" -PassThru -WindowStyle Hidden
    
    # Start LangChain MCP App Server
    Write-Host "Starting LangChain MCP App Server (port 8001)..."
    $lc = Start-Process powershell -ArgumentList "-Command", "python -m lc_mcp_app.server" -PassThru -WindowStyle Hidden
    
    # Wait for servers to start
    Write-Host "Waiting for servers to start..."
    $maxAttempts = 15
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        if ((Test-Port 8000) -and (Test-Port 8001)) {
            Write-Host "Servers are ready!" -ForegroundColor Green
            break
        }
        Start-Sleep -Seconds 2
        $attempt++
        Write-Host "." -NoNewline
    }
    Write-Host ""
}

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Yellow
Write-Host "================" -ForegroundColor Yellow

# Run different test suites
$testSuites = @(
    @{Name="Server Tests"; Path="tests/mcp_server/test_server.py"},
    @{Name="Health Tests"; Path="tests/test_health.py"},
    @{Name="OpenAI API Tests"; Path="tests/test_openai_api.py"},
    @{Name="MCP Contract Tests"; Path="tests/test_mcp_contract.py"},
    @{Name="MCP Conformance Tests"; Path="tests/test_mcp_conformance.py"}
)

$results = @()
foreach ($suite in $testSuites) {
    Write-Host "`nRunning $($suite.Name)..." -ForegroundColor Cyan
    $output = pytest $suite.Path -v --tb=short 2>&1
    $passed = $output | Select-String "passed" | Select-Object -First 1
    if ($passed) {
        $results += "$($suite.Name): $passed"
        Write-Host $passed -ForegroundColor Green
    } else {
        $failed = $output | Select-String "failed" | Select-Object -First 1
        $results += "$($suite.Name): $failed"
        Write-Host $failed -ForegroundColor Red
    }
}

# Summary
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
foreach ($result in $results) {
    if ($result -like "*passed*") {
        Write-Host $result -ForegroundColor Green
    } else {
        Write-Host $result -ForegroundColor Red
    }
}

# Cleanup
if (-not $Docker -and -not $TestsOnly) {
    Write-Host "`nCleaning up..." -ForegroundColor Yellow
    Read-Host "Press Enter to stop servers and exit"
    
    if ($mcp) { Stop-Process -Id $mcp.Id -Force -ErrorAction SilentlyContinue }
    if ($lc) { Stop-Process -Id $lc.Id -Force -ErrorAction SilentlyContinue }
}

if ($Docker) {
    Write-Host "`nDocker containers are still running. To stop them, run:" -ForegroundColor Yellow
    Write-Host "docker-compose down" -ForegroundColor White
}
