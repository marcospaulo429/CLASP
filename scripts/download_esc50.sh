#!/usr/bin/env bash
# Downloads ESC-50 dataset into data/noise_sources/esc50/
# Usage: bash scripts/download_esc50.sh [target_dir]

set -euo pipefail

TARGET="${1:-data/noise_sources/esc50}"
ZIP="$TARGET/esc50_master.zip"

mkdir -p "$TARGET"

echo "Downloading ESC-50 archive (~600MB)..."
nohup wget -c --progress=bar \
  https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip \
  -O "$ZIP" &

wait $!

echo "Extracting..."
unzip -q "$ZIP" -d "$TARGET"

# Move audio files to expected location
mv "$TARGET/ESC-50-master/audio" "$TARGET/audio"
mv "$TARGET/ESC-50-master/meta"  "$TARGET/meta"
rm -rf "$TARGET/ESC-50-master" "$ZIP"

echo "Done. $(ls "$TARGET/audio" | wc -l) WAV files in $TARGET/audio/"
echo ""
echo "Use with:"
echo "  uv run python scripts/build_spoken_squad_pkl.py \\"
echo "    --device cuda \\"
echo "    --noise-prob 0.5 \\"
echo "    --noise-snr 15.0 \\"
echo "    --noise-types white reverb ambient \\"
echo "    --esc50-dir $TARGET \\"
echo "    --vision-batch-size 32 \\"
echo "    --text-batch-size 128"
