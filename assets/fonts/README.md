# Fonts

The puzzle engine embeds 31 fonts: `RobotoMono-Bold.ttf` renders every board
label, and all of them serve as glyph styles in Character Recognition. **The font
files are not distributed with this repository.** Download them before building
the engine (`scripts/setup.sh` does this):

```bash
uv run --no-project --with fonttools==4.62.1 python scripts/download_fonts.py
```

The script fetches each source from a pinned release or commit and checks every
file against its sha256, so the engine renders the same puzzles everywhere:

- `DejaVuSans.ttf`, `RobotoBlack.ttf` and the three `ZillaSlab-*.ttf` files are
  used as released (DejaVu 2.37, googlefonts/roboto-2, Zilla Slab v1.002).
- The other 26 files are static instances (one weight each) of the variable fonts
  in the [Google Fonts repository](https://github.com/google/fonts), cut with
  fontTools 4.62.1.

All fonts are open source under the licenses below, whose texts are in
[`licenses/`](licenses). Copyright lines are those embedded in the font files.

| Files | Copyright | License |
|---|---|---|
| `DejaVuSans.ttf` | (c) 2003 Bitstream, Inc.; (c) 2006 Tavmjong Bah; DejaVu changes are public domain | Bitstream Vera / Arev license ([text](licenses/DejaVu-LICENSE.txt)) |
| `Inter-{Bold,Light,Regular}.ttf` | 2016 The Inter Project Authors | SIL OFL 1.1 ([text](licenses/OFL-inter.txt)) |
| `NotoSans-*.ttf` | 2022 The Noto Project Authors | SIL OFL 1.1 ([text](licenses/OFL-notosans.txt)) |
| `NotoSerif-*.ttf` | 2022 The Noto Project Authors | SIL OFL 1.1 ([text](licenses/OFL-notoserif.txt)) |
| `Oswald-*.ttf` | 2016 The Oswald Project Authors | SIL OFL 1.1 ([text](licenses/OFL-oswald.txt)) |
| `PlayfairDisplay-*.ttf` | 2017 The Playfair Display Project Authors | SIL OFL 1.1 ([text](licenses/OFL-playfairdisplay.txt)) |
| `RobotoBlack.ttf` | 2015 Google Inc. | Apache License 2.0 ([text](licenses/Apache-2.0.txt)) |
| `RobotoMono-{Bold,Regular}.ttf` | 2015 The Roboto Mono Project Authors | SIL OFL 1.1 ([text](licenses/OFL-robotomono.txt)) |
| `RobotoSlab-*.ttf` | 2018 The Roboto Slab Project Authors | Apache License 2.0 ([text](licenses/Apache-2.0.txt)) |
| `ZillaSlab-*.ttf` | 2017 The Mozilla Foundation | SIL OFL 1.1 ([text](licenses/OFL-zillaslab.txt)) |
