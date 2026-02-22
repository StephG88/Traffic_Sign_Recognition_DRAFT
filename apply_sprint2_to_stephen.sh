#!/bin/bash
# Applies the Sprint 2 commit to the 'stephen' branch of
# brocarli/CPE178_E01_212526_Traffic_Sign_Recognition
#
# Steps:
#   1. Download sprint2.patch from:
#      https://github.com/StephG88/Traffic_Sign_Recognition_DRAFT/blob/copilot/copy-sprint-2-commit/sprint2.patch
#   2. Clone brocarli's repo and run this script from inside it:
#        git clone https://github.com/brocarli/CPE178_E01_212526_Traffic_Sign_Recognition.git
#        cd CPE178_E01_212526_Traffic_Sign_Recognition
#        bash apply_sprint2_to_stephen.sh /path/to/sprint2.patch

set -e

PATCH_FILE="${1:-sprint2.patch}"

if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: patch file '$PATCH_FILE' not found."
    echo "Usage: bash apply_sprint2_to_stephen.sh /path/to/sprint2.patch"
    exit 1
fi

echo "Switching to 'stephen' branch..."
git checkout stephen

echo "Applying Sprint 2 patch from $PATCH_FILE..."
git am "$PATCH_FILE"

echo "Pushing to origin/stephen..."
git push origin stephen

echo "Done! Sprint 2 is now on the 'stephen' branch."
