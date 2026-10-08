# D1 step 2: supplementary immutable manifest of the E1 artifacts (verification NOW, not a
# launch-time receipt). Read-only on E1. Refuses to overwrite.
param([Parameter(Mandatory = $true)][string]$Artifacts)
$ErrorActionPreference = 'Stop'
$out = Join-Path $Artifacts 'd1\e1_supplementary_manifest.json'
if (Test-Path $out) { throw "exists: $out" }
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
$files = @()
foreach ($sub in 'fits', 'eval', 'report', 'intervention', 'init', 'spec', 'qual', 'audit') {
  $d = Join-Path $Artifacts $sub
  if (Test-Path $d) { $files += Get-ChildItem $d -Recurse -File }
}
$entries = [ordered]@{}
foreach ($f in ($files | Sort-Object FullName)) {
  $rel = $f.FullName.Substring($Artifacts.Length).TrimStart('\').Replace('\', '/')
  $entries[$rel] = [ordered]@{ sha256 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower(); bytes = $f.Length }
}
$obj = [ordered]@{
  kind = 'supplementary verification of E1 artifacts taken at D1 start; NOT a historical launch-time receipt'
  created_utc = (Get-Date).ToUniversalTime().ToString('o')
  files = $entries
}
($obj | ConvertTo-Json -Depth 5) | Out-File $out -Encoding ascii
"entries=$($entries.Count)"
$entries.Keys | Where-Object { $_ -match 'final/(model|opt_decay|opt_nodecay)\.mpk|final/meta.json' } | ForEach-Object { "$_ $($entries[$_].sha256.Substring(0,12))" }
