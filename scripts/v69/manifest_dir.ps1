# Append-only manifest of a V69 artifact directory (verification taken now; not a launch-time receipt).
param([Parameter(Mandatory = $true)][string]$Artifacts, [Parameter(Mandatory = $true)][string]$Src, [Parameter(Mandatory = $true)][string]$OutRel, [string]$Kind = 'supplementary verification')
$ErrorActionPreference = 'Stop'
$out = Join-Path $Artifacts ($OutRel.Replace('/', '\'))
if (Test-Path $out) { throw "exists (append-only): $out" }
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
$entries = [ordered]@{}
foreach ($f in (Get-ChildItem (Join-Path $Artifacts $Src) -Recurse -File | Sort-Object FullName)) {
  $rel = $f.FullName.Substring($Artifacts.Length).TrimStart('\').Replace('\', '/')
  $entries[$rel] = [ordered]@{ sha256 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower(); bytes = $f.Length }
}
$obj = [ordered]@{ kind = "$Kind of $Src/ taken at G1-R1 start; not a launch-time receipt"; created_utc = (Get-Date).ToUniversalTime().ToString('o'); files = $entries }
($obj | ConvertTo-Json -Depth 5) | Out-File $out -Encoding ascii
"entries=$($entries.Count)"
