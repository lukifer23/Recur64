# Freeze the executable specification before any scientific fit.
# Copies docs/v69/{MODEL_SPEC,CONTRACT}.md into the V69 artifact namespace and records the hashes of
# everything a fit is bound to. Refuses to overwrite an existing freeze.
param([Parameter(Mandatory = $true)][string]$Artifacts)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$spec = Join-Path $Artifacts 'spec'
$frozen = Join-Path $spec 'frozen.json'
if (Test-Path $frozen) { throw "spec already frozen: $frozen" }
New-Item -ItemType Directory -Force $spec | Out-Null
Copy-Item (Join-Path $repo 'docs\v69\MODEL_SPEC.md') (Join-Path $spec 'MODEL_SPEC.md')
Copy-Item (Join-Path $repo 'docs\v69\CONTRACT.md') (Join-Path $spec 'CONTRACT.md')
function Sha256Of($p) { (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() }
$inv = Get-Content (Join-Path $spec 'param_inventory.json') -Raw | ConvertFrom-Json
$obj = [ordered]@{
  frozen_utc                 = (Get-Date).ToUniversalTime().ToString('o')
  model_spec_sha256          = Sha256Of (Join-Path $spec 'MODEL_SPEC.md')
  contract_sha256            = Sha256Of (Join-Path $spec 'CONTRACT.md')
  canonical_init_file_sha256 = Sha256Of (Join-Path $Artifacts 'init\canonical_init.bin')
  canonical_init_tensors_sha256 = $inv.tensors_sha256
  parameter_inventory_sha256 = Sha256Of (Join-Path $spec 'param_inventory.json')
  total_parameters           = $inv.total_parameters
  intervention_map_sha256    = Sha256Of (Join-Path $Artifacts 'intervention\map.json')
  qual_summary_sha256        = Sha256Of (Join-Path $Artifacts 'qual\qual_summary.json')
  dataset_manifest_sha256    = Sha256Of (Join-Path $Artifacts 'gen-001\MANIFEST.sha256.json')
  audit_v2_receipt_sha256    = Sha256Of (Join-Path $Artifacts 'audit\gen-001_audit_receipt_v2.json')
  git_head_at_freeze         = (git -C $repo rev-parse HEAD)
  git_dirty_files_at_freeze  = @(git -C $repo status --porcelain).Count
}
($obj | ConvertTo-Json) | Out-File $frozen -Encoding ascii
Write-Output ($obj | ConvertTo-Json)
