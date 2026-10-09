# G1 step: append-only supplementary manifest of the D3 artifacts (verification taken at G1 start).
param([Parameter(Mandatory = $true)][string]$Artifacts)
$ErrorActionPreference = 'Stop'
$out = Join-Path $Artifacts 'g1\d3_supplementary_manifest.json'
if (Test-Path $out) { throw "exists (append-only): $out" }
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
$entries = [ordered]@{}
foreach ($f in (Get-ChildItem (Join-Path $Artifacts 'd3') -Recurse -File | Sort-Object FullName)) {
  $rel = $f.FullName.Substring($Artifacts.Length).TrimStart('\').Replace('\', '/')
  $entries[$rel] = [ordered]@{ sha256 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower(); bytes = $f.Length }
}
$obj = [ordered]@{ kind = 'supplementary verification of D3 artifacts taken at G1 start; not a launch-time receipt'; created_utc = (Get-Date).ToUniversalTime().ToString('o'); files = $entries }
($obj | ConvertTo-Json -Depth 5) | Out-File $out -Encoding ascii
"entries=$($entries.Count)"
