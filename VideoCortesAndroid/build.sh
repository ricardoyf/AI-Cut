#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

./gradlew --no-daemon :app:assembleDebug

mkdir -p "$PROJECT_DIR/build"
cp "$PROJECT_DIR/app/build/outputs/apk/debug/app-debug.apk" "$PROJECT_DIR/build/videocortes.apk"
echo "$PROJECT_DIR/build/videocortes.apk"
