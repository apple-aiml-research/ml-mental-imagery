#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Download OPTED and recreate the Anagram word list, engine/games/anagram/dictionary.txt.

    python scripts/download_words.py [--output PATH]

The word list is not distributed with this repository. It is a subset of the headwords of OPTED,
The Online Plain Text English Dictionary by Ralph Sutherland
(https://www.mso.anu.edu.au/~ralph/OPTED/), which is based on the Project Gutenberg eBook of
Webster's Unabridged Dictionary (eBook #29765, https://www.gutenberg.org/ebooks/29765), in turn
based on the 1913 edition of Webster's Unabridged Dictionary. The source text is in the public
domain; see engine/games/anagram/SOURCE.md. This script:

1. downloads the 26 OPTED letter files and checks each against its sha256;
2. takes every headword (the <B>...</B> text that starts an entry), splits it into words (a
   headword such as "Signaled (/) or Signalled" holds several), lowercases them and keeps the
   distinct words of 6-10 ASCII letters, sorted;
3. keeps the words whose bit is set in SELECTION_MASK (bit i, least significant first, of byte
   i // 8 selects the i-th sorted word);
4. writes them one per line and checks the file against its sha256, so the engine generates the
   same puzzles everywhere. A word list already in place with that sha256 is kept.

Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "engine", "games", "anagram", "dictionary.txt")

OPTED_URL = "https://www.mso.anu.edu.au/~ralph/OPTED/v003/wb1913_{letter}.html"
# OPTED v0.03 letter file -> sha256.
OPTED_SHA256 = {
    "a": "1fabf10a54dd800e8ac9c85473b09abbd671ac9cbc0745d66034172db2d5914f",
    "b": "047a9936fe5fc38b7a2445c1f89b5a7123a9f1e20a25181d84379960bb7a83b3",
    "c": "29cf007e80eb1e6d27267485ce255964ce4e58ae241598c63e174e10614d6a5e",
    "d": "93d85dcf02a7c5d4505b9f139b9dbaf71db0453223af198f9466efbe4c063892",
    "e": "f912dca0072209766eeb58e9377722f5576ef734b7bec290318b0f62f4cc5208",
    "f": "357e0138fda0ca5f5483c83e707c7b4102dd68b720e5396382f475c9a1026809",
    "g": "ed495eb4067c29cb07e17018ce90f214c698a062dbf6702a602ff76d923f9b5e",
    "h": "57aacae0d68c11b91a939032ae0970bd335ce29c8c1bdb1ebf924476fbe07588",
    "i": "52d704f09d25c2e42d2f71540210177434f2680b7791d5541ddbb042df8874e3",
    "j": "610ae95ca2fe7f864c80f39e50efc6a143fb62394eb6208820cae8022916c0ee",
    "k": "982ed239d0ca6d5b3ec8df10453395d582210cdc59448629c7a5a3a88fdbb78d",
    "l": "4a855b02711f7b2317a6ee24c7d0f77f3941a408df080b47feb5a624ff959e89",
    "m": "877e63f7a5259134e4dd7f8ec88c3f9918175c9b3094ce2d9cb0539e8eea0c10",
    "n": "95026a42f14f7a705ae163c56179d8045f7a56fbfb7db16531b24375b20e60e5",
    "o": "b177a089b037eb47b5d6e17d40250e444d38f9dc38c96360b0a618bd76582fc3",
    "p": "dcae0cd217d4b82a996b3e6193998781de3f9a2822a0c16cc193dd5d0bbd6014",
    "q": "fe12deb58e04bfe962ed580a499a27eca291b7ba83042a9edee68d58ad67f513",
    "r": "dab4b42036aff350b9d34b5986283c918650009bf82a2d13419325f1516654e1",
    "s": "5886987092cc1cb1f4fb0c3c648c6616af6785fa022a94ddd88abeadbdccf678",
    "t": "0c0f7af5a3c6245f6f3095674caf6480b29a4e3c73a7d8458618f0b8b4503ca9",
    "u": "2c124fbec2c9e1ff470c4c6c495899d0b48817c09f1e4a7a7bb6232ece6dd0ce",
    "v": "ef5c668ba9268c23b12fe8a120c2484dc68819d4b817ea35e2264f094545d8be",
    "w": "bbedb488b83da66950cb522f5364dea1ef2fcec671cce97158e015b19648d313",
    "x": "d833708dcbe76130ade4c06b2a0f15a737a48c203ae569ea78b89f8c8a08fe1d",
    "y": "1738e15702bc164c9c046d62bc0b68907cf0bcc4e3a4f62056dd773c04065c8f",
    "z": "8ebf2ea859374220b8f6a208d4998b28f10ca3ca6f23b3064cf26f312adbe8af",
}
NUM_CANDIDATES = 72412  # distinct headword words of 6-10 ASCII letters
NUM_WORDS = 34011
DICTIONARY_SHA256 = "f8a212c9022c8a036ab3b46c3cd05a8e22ed8fd25b76c31c0666feee21f9c067"

# One bit per candidate word (base64 of 9,052 bytes).
SELECTION_MASK = (
    "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"
    "DBgAWgAAASwAMBqAAA=="
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(letter: str) -> bytes:
    url = OPTED_URL.format(letter=letter)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if sha256(data) != OPTED_SHA256[letter]:
        raise RuntimeError(f"{url}: sha256 {sha256(data)}, expected {OPTED_SHA256[letter]}")
    return data


def candidate_words(pages: list[bytes]) -> list[str]:
    """The distinct words of 6-10 ASCII letters in the headwords, sorted."""
    words = set()
    for page in pages:
        text = page.decode("mac_roman")  # OPTED was prepared on a Macintosh
        for headword in re.findall(r"<P><B>(.*?)</B>", text):
            for token in re.findall(r"[^\W\d_]+", headword):
                word = token.lower()
                if re.fullmatch(r"[a-z]{6,10}", word):
                    words.add(word)
    return sorted(words)


def select(candidates: list[str]) -> list[str]:
    mask = base64.b64decode(SELECTION_MASK)
    if len(candidates) != NUM_CANDIDATES or len(mask) != (NUM_CANDIDATES + 7) // 8:
        raise RuntimeError(f"{len(candidates)} candidate words, expected {NUM_CANDIDATES}")
    return [word for i, word in enumerate(candidates) if mask[i >> 3] >> (i & 7) & 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="where to write the word list")
    output = parser.parse_args().output

    if os.path.exists(output):
        with open(output, "rb") as f:
            if sha256(f.read()) == DICTIONARY_SHA256:
                print(f"{output} is already in place")
                return

    pages = []
    for letter in sorted(OPTED_SHA256):
        print(f"downloading OPTED {letter.upper()}")
        pages.append(download(letter))
    words = select(candidate_words(pages))
    data = "".join(word + "\n" for word in words).encode("ascii")
    if len(words) != NUM_WORDS or sha256(data) != DICTIONARY_SHA256:
        raise RuntimeError(
            f"word list: {len(words)} words, sha256 {sha256(data)}; "
            f"expected {NUM_WORDS} words, {DICTIONARY_SHA256}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, output)
    print(f"wrote {len(words)} words to {output}")


if __name__ == "__main__":
    main()
