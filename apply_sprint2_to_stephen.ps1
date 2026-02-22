# Applies the Sprint 2 commit to the 'stephen' branch of
# brocarli/CPE178_E01_212526_Traffic_Sign_Recognition
#
# Run this ONE command from ANY folder in PowerShell:
#
#   irm https://raw.githubusercontent.com/StephG88/Traffic_Sign_Recognition_DRAFT/copilot/copy-sprint-2-commit/apply_sprint2_to_stephen.ps1 | iex
#
# It will clone the repo (if needed), apply the patch, and push automatically.

$ErrorActionPreference = "Stop"

$repoUrl   = "https://github.com/brocarli/CPE178_E01_212526_Traffic_Sign_Recognition.git"
$repoName  = "CPE178_E01_212526_Traffic_Sign_Recognition"
$patchUrl  = "https://raw.githubusercontent.com/StephG88/Traffic_Sign_Recognition_DRAFT/copilot/copy-sprint-2-commit/sprint2.patch"
$patchFile = "$env:TEMP\sprint2.patch"

# Clone repo if it doesn't exist in the current directory
if (-not (Test-Path $repoName)) {
    Write-Host "Cloning $repoName..."
    git clone $repoUrl
}

Set-Location $repoName

Write-Host "Switching to 'stephen' branch..."
git checkout stephen

Write-Host "Downloading Sprint 2 patch..."
Invoke-WebRequest -Uri $patchUrl -OutFile $patchFile

Write-Host "Applying Sprint 2 patch..."
git am $patchFile

Write-Host "Pushing to origin/stephen..."
git push origin stephen

Remove-Item $patchFile -ErrorAction SilentlyContinue
Write-Host "Done! Sprint 2 is now on the 'stephen' branch."
