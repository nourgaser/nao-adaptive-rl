#!/usr/bin/env bash
# Build a PDF next to every Markdown file in docs/, and the demo slides.
# Needs pandoc (or PANDOC=/path/to/pandoc) and tectonic.
#
#   docs/build_pdfs.sh
set -euo pipefail
cd "$(dirname "$0")"
PANDOC="${PANDOC:-pandoc}"
TECTONIC="${TECTONIC:-tectonic}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Smaller code, wrapped when too wide, so commands and diagrams stay on the page.
cat > "$tmp/header.tex" <<'TEX'
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\},fontsize=\small}
\usepackage{etoolbox}
\AtBeginEnvironment{verbatim}{\small}
TEX

for md in *.md; do
  name="${md%.md}"
  "$PANDOC" "$md" -s -o "$tmp/$name.tex" \
    -V geometry:margin=2.2cm -V fontsize=11pt -V colorlinks=true \
    -V linkcolor=blue -V urlcolor=blue --syntax-highlighting=tango \
    --include-in-header="$tmp/header.tex"
  "$TECTONIC" -X compile "$tmp/$name.tex" --outdir "$tmp" 2>&1 \
    | grep -E "error|Overfull" || true
  cp "$tmp/$name.pdf" "$name.pdf"
  echo "docs/$name.pdf"
done

(cd ../demo-slides && "$TECTONIC" -X compile demo.tex >/dev/null && echo "demo-slides/demo.pdf")
