#!/usr/bin/env bash
#
# remove_pycaches.sh
# Recursively removes all __pycache__ directories under the project root.
# Place this script in your project’s root and run: ./remove_pycaches.sh

# Resolve the directory this script lives in (the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "🧹 Cleaning __pycache__ directories under: $PROJECT_ROOT"

# Find and delete all __pycache__ directories
find "$PROJECT_ROOT" -type d -name "__pycache__" -print -exec rm -rf {} +

echo "✅ All __pycache__ directories removed."
