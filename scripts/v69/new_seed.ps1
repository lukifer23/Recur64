# Draw the V69 experiment master seed from operating-system entropy and record it.
# Refuses to overwrite an existing seed. Never re-run to "search" for a seed.
param([Parameter(Mandatory = $true)][string]$SeedFile)
$ErrorActionPreference = 'Stop'
if (Test-Path $SeedFile) { throw "seed file exists; V69 seeds are never redrawn: $SeedFile" }
New-Item -ItemType Directory -Force (Split-Path $SeedFile) | Out-Null
$b = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($b)
$hex = -join ($b | ForEach-Object { $_.ToString('x2') })
[IO.File]::WriteAllText($SeedFile, $hex)
(Get-Item $SeedFile).IsReadOnly = $true
$sha = [System.Security.Cryptography.SHA256]::Create().ComputeHash($b)
$fp = (-join ($sha | ForEach-Object { $_.ToString('x2') })).Substring(0, 16)
$stamp = (Get-Date).ToUniversalTime().ToString('o')
"source=System.Security.Cryptography.RandomNumberGenerator (OS CSPRNG)`nrecorded_utc=$stamp`nfingerprint=$fp" |
  Out-File "$SeedFile.record" -Encoding ascii
Write-Output "seed fingerprint $fp recorded at $stamp"
