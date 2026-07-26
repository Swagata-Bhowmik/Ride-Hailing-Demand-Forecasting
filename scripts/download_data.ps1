# Downloads the 12-month NYC TLC FHVHV window (2025-05 .. 2026-04) into data/.
# Skips files that already exist with a plausible size. Safe to re-run (resumable by skip).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$base = 'https://d37ci6vzurychx.cloudfront.net/trip-data'
$dataDir = Join-Path $PSScriptRoot '..\data'
$dataDir = (Resolve-Path $dataDir).Path

$months = @('2025-05','2025-06','2025-07','2025-08','2025-09','2025-10',
            '2025-11','2025-12','2026-01','2026-02','2026-03','2026-04')

foreach ($m in $months) {
    $name = "fhvhv_tripdata_$m.parquet"
    $out  = Join-Path $dataDir $name
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 100MB)) {
        Write-Output "SKIP  $name (already present, $([math]::Round((Get-Item $out).Length/1MB,1)) MB)"
        continue
    }
    $url = "$base/$name"
    Write-Output "GET   $name ..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -TimeoutSec 1800
        Write-Output "OK    $name ($([math]::Round((Get-Item $out).Length/1MB,1)) MB)"
    } catch {
        Write-Output "FAIL  $name : $($_.Exception.Message)"
    }
}
Write-Output "ALL-DONE download_data.ps1"
