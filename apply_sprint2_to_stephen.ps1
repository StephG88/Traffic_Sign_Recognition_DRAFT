# Applies the Sprint 2 commit to the 'stephen' branch of
# brocarli/CPE178_E01_212526_Traffic_Sign_Recognition
#
# Run this script from inside a local clone of brocarli's repo:
#   git clone https://github.com/brocarli/CPE178_E01_212526_Traffic_Sign_Recognition.git
#   cd CPE178_E01_212526_Traffic_Sign_Recognition
#   .\apply_sprint2_to_stephen.ps1

$ErrorActionPreference = "Stop"

Write-Host "Downloading Sprint 2 patch..."
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/StephG88/Traffic_Sign_Recognition_DRAFT/copilot/copy-sprint-2-commit/sprint2.patch" -OutFile "sprint2.patch"

Write-Host "Switching to 'stephen' branch..."
git checkout stephen

Write-Host "Applying Sprint 2 patch..."
git am sprint2.patch

Write-Host "Pushing to origin/stephen..."
git push origin stephen

Write-Host "Cleaning up..."
Remove-Item sprint2.patch

Write-Host "Done! Sprint 2 is now on the 'stephen' branch."
