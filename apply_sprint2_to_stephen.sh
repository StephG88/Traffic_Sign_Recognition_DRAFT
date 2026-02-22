#!/bin/bash
# Applies the Sprint 2 commit to the 'stephen' branch of
# brocarli/CPE178_E01_212526_Traffic_Sign_Recognition
#
# Run this script inside a local clone of brocarli's repo:
#   git clone https://github.com/brocarli/CPE178_E01_212526_Traffic_Sign_Recognition.git
#   cd CPE178_E01_212526_Traffic_Sign_Recognition
#   bash apply_sprint2_to_stephen.sh

set -e

echo "Switching to 'stephen' branch..."
git checkout stephen

echo "Downloading Sprint 2 patch..."
curl -sL https://raw.githubusercontent.com/StephG88/Traffic_Sign_Recognition_DRAFT/copilot/copy-sprint-2-commit/sprint2.patch -o /tmp/sprint2.patch

echo "Applying patch..."
git am /tmp/sprint2.patch

echo "Pushing to origin/stephen..."
git push origin stephen

echo "Done! Sprint 2 is now on the 'stephen' branch."
