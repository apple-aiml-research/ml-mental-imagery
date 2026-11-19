#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Download the fonts the puzzle engine embeds into assets/fonts/ (once, before building the engine).

    uv run --no-project --with fonttools==4.62.1 python scripts/download_fonts.py

The fonts are not distributed with this repository; they are open-source fonts (SIL Open Font
License 1.1, Apache License 2.0, Bitstream Vera license; see assets/fonts/README.md). Five files
are used as released. The other 26 are static instances of variable fonts from the Google Fonts
repository, cut with fontTools 4.62.1 with fixed head-table timestamps, so the engine renders the
same puzzles everywhere. Every source and every output is checked against its sha256; fonts
already in place are kept.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile

FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts"
)
FONTTOOLS_VERSION = "4.62.1"  # instances match FONT_SHA256 only with this version

GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts"
# Variable fonts: path in the google/fonts repository -> (commit, sha256).
VARIABLE_FONTS = {
    "ofl/inter/Inter[opsz,wght].ttf": (
        "e1d6480102fed30739fead0faee463101f892c8f",
        "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031",
    ),
    "ofl/notosans/NotoSans[wdth,wght].ttf": (
        "2984c575fdce412ee02b2baaba67672b9a9434d8",
        "bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d",
    ),
    "ofl/notosans/NotoSans-Italic[wdth,wght].ttf": (
        "2984c575fdce412ee02b2baaba67672b9a9434d8",
        "58e6e0ebd1931b29a365aa2d3e2ee9a9e831a3af7cf3ad1462d4e72154f0b291",
    ),
    "ofl/notoserif/NotoSerif[wdth,wght].ttf": (
        "49af86b54514f1390e126f3139ced33824f5d72e",
        "4d8e6761424656867019081a1a01336f3cb086982682698714054fc33f782713",
    ),
    "ofl/notoserif/NotoSerif-Italic[wdth,wght].ttf": (
        "b233f71caa64d2f82b3a6667d3a571f33ce76c3f",
        "e87acbc6c0efd0d9a20d6a8cbbda2b266c14be3a3a6f5af8ec9d7b2460570ad1",
    ),
    "ofl/oswald/Oswald[wght].ttf": (
        "b6f0fe1740573b70ee367fbaba04b7586be85af3",
        "5b38c246e255a12f5712d640d56bcced0472466fc68983d2d0410ec0457c2817",
    ),
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf": (
        "1e1aa08e994ff7db50116e86ccc7b52a4e4ae5b8",
        "c40f2293766a503bc70cce9e512ef844a4ccb7cbcde792fe2ea31d191917d8d6",
    ),
    "ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf": (
        "1e1aa08e994ff7db50116e86ccc7b52a4e4ae5b8",
        "a5e26dc5e2e77fb2803a0bf02fd4f81ee136ec8dea863ccdb0c59a263b21378b",
    ),
    "ofl/robotomono/RobotoMono[wght].ttf": (
        "b01086459d9d41ece904d39155b93f7bd3a676f2",
        "66a80e79d17e4c7cabd162e2916578a4cc08fd19eef6e2a643305eae9c567b2b",
    ),
    "apache/robotoslab/RobotoSlab[wght].ttf": (
        "892dc85555606db5a08545c7874a4dc561d3ce50",
        "786ae192477447d33c6672c3055fba7cbfe45184c9a79e77a14f15716ca05b16",
    ),
}

# Static instances: font -> (variable font, axis location, head.modified timestamp).
INSTANCES = {
    "Inter-Light.ttf": ("ofl/inter/Inter[opsz,wght].ttf", {"opsz": 14, "wght": 300}, 3860268074),
    "Inter-Regular.ttf": ("ofl/inter/Inter[opsz,wght].ttf", {"opsz": 14, "wght": 400}, 3860268076),
    "Inter-Bold.ttf": ("ofl/inter/Inter[opsz,wght].ttf", {"opsz": 14, "wght": 700}, 3860268079),
    "NotoSans-Light.ttf": (
        "ofl/notosans/NotoSans[wdth,wght].ttf",
        {"wght": 300, "wdth": 100},
        3860268051,
    ),
    "NotoSans-Regular.ttf": (
        "ofl/notosans/NotoSans[wdth,wght].ttf",
        {"wght": 400, "wdth": 100},
        3860268055,
    ),
    "NotoSans-Bold.ttf": (
        "ofl/notosans/NotoSans[wdth,wght].ttf",
        {"wght": 700, "wdth": 100},
        3860268060,
    ),
    "NotoSans-Black.ttf": (
        "ofl/notosans/NotoSans[wdth,wght].ttf",
        {"wght": 900, "wdth": 100},
        3860268064,
    ),
    "NotoSans-Italic.ttf": (
        "ofl/notosans/NotoSans-Italic[wdth,wght].ttf",
        {"wght": 400, "wdth": 100},
        3860268147,
    ),
    "NotoSans-BoldItalic.ttf": (
        "ofl/notosans/NotoSans-Italic[wdth,wght].ttf",
        {"wght": 700, "wdth": 100},
        3860268152,
    ),
    "NotoSerif-Regular.ttf": (
        "ofl/notoserif/NotoSerif[wdth,wght].ttf",
        {"wght": 400, "wdth": 100},
        3860268068,
    ),
    "NotoSerif-Bold.ttf": (
        "ofl/notoserif/NotoSerif[wdth,wght].ttf",
        {"wght": 700, "wdth": 100},
        3860268071,
    ),
    "NotoSerif-Italic.ttf": (
        "ofl/notoserif/NotoSerif-Italic[wdth,wght].ttf",
        {"wght": 400, "wdth": 100},
        3860268155,
    ),
    "NotoSerif-BoldItalic.ttf": (
        "ofl/notoserif/NotoSerif-Italic[wdth,wght].ttf",
        {"wght": 700, "wdth": 100},
        3860268157,
    ),
    "Oswald-Light.ttf": ("ofl/oswald/Oswald[wght].ttf", {"wght": 300}, 3860268086),
    "Oswald-Regular.ttf": ("ofl/oswald/Oswald[wght].ttf", {"wght": 400}, 3860268087),
    "Oswald-Bold.ttf": ("ofl/oswald/Oswald[wght].ttf", {"wght": 700}, 3860268087),
    "PlayfairDisplay-Regular.ttf": (
        "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
        {"wght": 400},
        3860268080,
    ),
    "PlayfairDisplay-Bold.ttf": (
        "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
        {"wght": 700},
        3860268082,
    ),
    "PlayfairDisplay-Black.ttf": (
        "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
        {"wght": 900},
        3860268083,
    ),
    "PlayfairDisplay-Italic.ttf": (
        "ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf",
        {"wght": 400},
        3860268159,
    ),
    "PlayfairDisplay-BoldItalic.ttf": (
        "ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf",
        {"wght": 700},
        3860268160,
    ),
    "RobotoMono-Regular.ttf": ("ofl/robotomono/RobotoMono[wght].ttf", {"wght": 400}, 3860268071),
    "RobotoMono-Bold.ttf": ("ofl/robotomono/RobotoMono[wght].ttf", {"wght": 700}, 3860268071),
    "RobotoSlab-Light.ttf": ("apache/robotoslab/RobotoSlab[wght].ttf", {"wght": 300}, 3860268084),
    "RobotoSlab-Regular.ttf": ("apache/robotoslab/RobotoSlab[wght].ttf", {"wght": 400}, 3860268085),
    "RobotoSlab-Bold.ttf": ("apache/robotoslab/RobotoSlab[wght].ttf", {"wght": 700}, 3860268086),
}

# Fonts used as released: font -> (URL, member of the zip archive or None for a plain file).
DEJAVU_ZIP = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
ZILLA_ZIP = (
    "https://github.com/mozilla/zilla-slab/releases/download/v1.002/Zilla-Slab-Fonts-v1.002.zip"
)
RELEASED = {
    "DejaVuSans.ttf": (DEJAVU_ZIP, "dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf"),
    "RobotoBlack.ttf": (
        "https://raw.githubusercontent.com/googlefonts/roboto-2/"
        "eaef3058ec9d4f426c013b29fae53345e1c36f67/src/hinted/Roboto-Black.ttf",
        None,
    ),
    "ZillaSlab-Bold.ttf": (ZILLA_ZIP, "zilla-slab/ttf/ZillaSlab-Bold.ttf"),
    "ZillaSlab-BoldItalic.ttf": (ZILLA_ZIP, "zilla-slab/ttf/ZillaSlab-BoldItalic.ttf"),
    "ZillaSlab-Light.ttf": (ZILLA_ZIP, "zilla-slab/ttf/ZillaSlab-Light.ttf"),
}

# sha256 of every font the engine embeds (engine/base/render.rs, engine/games/character_recognition/gen.rs).
FONT_SHA256 = {
    "DejaVuSans.ttf": "7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954",
    "Inter-Bold.ttf": "c00dece3c9ee432d169e43d683da78e6de9423b40eb51bf8b1c2763755f36720",
    "Inter-Light.ttf": "87083c01110e899bdba710ef2feab88dcd6d48ffe69ed0f1b61d6ddae0b96c07",
    "Inter-Regular.ttf": "8aaaca1dc5fb8019c974a0f307c896c06b199203d14bf5b980dc21eba6468585",
    "NotoSans-Black.ttf": "e3aa6a97c181a133df5d5db4b11b3fb31dd0204c4700a7ef5194c0a0453d1f77",
    "NotoSans-Bold.ttf": "4bf61866d6752ad9fa21915bbeedde2387f0767df5dee3a5b2bcea3bf9a5d172",
    "NotoSans-BoldItalic.ttf": "06aa2a6f66474ba1ba9c8e31cc6be067ef87c53eda7ec84a97ee5ee6db9a7c73",
    "NotoSans-Italic.ttf": "3dbed22af247bc5af27ecaaabb7e3a14d9ab66d0b9be720e85b7f6dfab957915",
    "NotoSans-Light.ttf": "01257b7c9446082e4d04b0468964346f1346b9b0a7a03ff2d5a02aa81b102a95",
    "NotoSans-Regular.ttf": "8e6954baac65a49c9decd3fb130402422f0a88551a836d21aa148766d4c3bcd7",
    "NotoSerif-Bold.ttf": "ef019735fac963ed56a456db470ac6c2a33125d91fe336b029e0a70f22ef345f",
    "NotoSerif-BoldItalic.ttf": "2f1da5cf29562ddbf92cb0243b69e3e34205d77b298add7896996c6f25cb87af",
    "NotoSerif-Italic.ttf": "3e94896c20c1cb5397abdf547056ed2373223da36d5b6ace682e04a3f5ed0fdc",
    "NotoSerif-Regular.ttf": "c555e03e94e966e407435a4b416f926cd13372de0bda8e01aafbc1777b8e601b",
    "Oswald-Bold.ttf": "31f06903007c884692af0b206ce3945ecbaa9d00628f8f66c48cc99df7a81fc5",
    "Oswald-Light.ttf": "ffe6dc8a4b14929716ea3ebcc9e1928d9d8f91f11b8b8aa1da50379ec000d6e7",
    "Oswald-Regular.ttf": "6878cd79b88b3fb05d2efe769f8b1eab0d4f2a5fc4f283cd4b91349fc89174ef",
    "PlayfairDisplay-Black.ttf": "17183f8f4bdb87d21011c93b14a87b534aaf5ed1619c1e9bf86fc3fa0b1df481",
    "PlayfairDisplay-Bold.ttf": "9abdd5b4a4ef7b54c5ee0c53c8ba1d9c09d83338dfebdda70609550ec0d881f1",
    "PlayfairDisplay-BoldItalic.ttf": "8fc0522343884073245cc90faddcb006a5e47ee79a7094636313d5f0752ff876",
    "PlayfairDisplay-Italic.ttf": "a5855fd97dba938ce89b480e36b677da122dd1614c14e0eb4176a295396de029",
    "PlayfairDisplay-Regular.ttf": "19fbfa9dd41d2131c3546c87242fe36016c0b809ceb7a85ebd2c5d9ef4117bce",
    "RobotoBlack.ttf": "6ca9c892798579f5bf87a497bd9b9e087ecae16c75ee2209ae7bc86e27e8ff64",
    "RobotoMono-Bold.ttf": "0a5be511abbf07e67abff0022aff472ce929ff23b6bcbfe222600a309985102e",
    "RobotoMono-Regular.ttf": "66fae7528868e4cd910bab16a46b29669c3a7b709283a3e246a49bd161287b09",
    "RobotoSlab-Bold.ttf": "a7fc0d5c028d18ca1a4e6cbc65b1b78f0adb1a2b1d682130a7f313d070cc44f5",
    "RobotoSlab-Light.ttf": "704dee37ebbd4e803367b751dfb51c8e30e2a226e88aad81003fc633764e4836",
    "RobotoSlab-Regular.ttf": "81152645a0de9fdc27e30f7b5a2b5fbbbbf4e2efd233457fca6ae6274d1e1d54",
    "ZillaSlab-Bold.ttf": "4ec3a04a4eef37074b42ef542e4d874e13646668cfe65256e0bf100441cf8719",
    "ZillaSlab-BoldItalic.ttf": "7951036c92de71e3c88ea01abe031d5bdaddf4639022a3e1cb92322f2e88d2cf",
    "ZillaSlab-Light.ttf": "4c3dfcd4a5bb71ee28b6c2ec8e8bfec720f619bc00e12a13a88f0f233af7f851",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, cache: dict) -> bytes:
    if url not in cache:
        print(f"  downloading {url}", flush=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "ml-mental-imagery-download-fonts"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            cache[url] = response.read()
    return cache[url]


def instantiate(variable_font: bytes, location: dict, head_modified: int) -> bytes:
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = instancer.instantiateVariableFont(TTFont(io.BytesIO(variable_font)), location)
    font.recalcTimestamp = False  # fixed timestamp: the bytes must match FONT_SHA256
    font["head"].modified = head_modified
    buffer = io.BytesIO()
    font.save(buffer)
    return buffer.getvalue()


def write_verified(name: str, data: bytes) -> None:
    if sha256(data) != FONT_SHA256[name]:
        raise SystemExit(f"{name}: sha256 {sha256(data)} does not match {FONT_SHA256[name]}")
    with open(os.path.join(FONT_DIR, name), "wb") as f:
        f.write(data)
    print(f"  wrote {name}", flush=True)


def is_in_place(name: str) -> bool:
    path = os.path.join(FONT_DIR, name)
    return os.path.exists(path) and sha256(open(path, "rb").read()) == FONT_SHA256[name]


def main() -> None:
    missing = [name for name in FONT_SHA256 if not is_in_place(name)]
    if not missing:
        print(f"all {len(FONT_SHA256)} fonts are in {FONT_DIR}")
        return
    if any(name in INSTANCES for name in missing):
        import fontTools

        if fontTools.version != FONTTOOLS_VERSION:
            raise SystemExit(
                f"fontTools {fontTools.version} found; run with fontTools {FONTTOOLS_VERSION}:\n"
                "  uv run --no-project --with fonttools==4.62.1 python scripts/download_fonts.py"
            )
    os.makedirs(FONT_DIR, exist_ok=True)
    cache: dict[str, bytes] = {}
    for name in missing:
        if name in RELEASED:
            url, member = RELEASED[name]
            data = download(url, cache)
            write_verified(name, zipfile.ZipFile(io.BytesIO(data)).read(member) if member else data)
        else:
            source, location, head_modified = INSTANCES[name]
            commit, source_sha256 = VARIABLE_FONTS[source]
            variable_font = download(f"{GOOGLE_FONTS}/{commit}/{source}", cache)
            if sha256(variable_font) != source_sha256:
                raise SystemExit(f"{source} at {commit}: unexpected sha256 {sha256(variable_font)}")
            write_verified(name, instantiate(variable_font, location, head_modified))
    print(f"all {len(FONT_SHA256)} fonts are in {FONT_DIR}")


if __name__ == "__main__":
    main()
