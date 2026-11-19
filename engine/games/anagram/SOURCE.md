# Anagram word list

`dictionary.txt` holds 34,011 lowercase ASCII words of 6-10 letters, one per
line. It is a subset of OPTED's headwords, without slurs, sexual terms,
profanity and other offensive words.

The word list is not distributed with this repository. Recreate it from OPTED
before building the engine (`scripts/setup.sh` does this):

```bash
python scripts/download_words.py
```

The script downloads the OPTED letter files, takes their headwords of 6-10
ASCII letters and keeps those its selection mask marks. Every download and the
resulting word list are checked against their sha256.

The words come from:

- [OPTED](https://www.mso.anu.edu.au/~ralph/OPTED/) (The Online Plain Text
  English Dictionary) by Ralph Sutherland, a plain-text version of
- [The Project Gutenberg eBook of *Webster's Unabridged Dictionary*](https://www.gutenberg.org/ebooks/29765)
  (eBook #29765), which is based on
- *Webster's Unabridged Dictionary*, 1913 edition.

## License

Project Gutenberg lists eBook #29765 as "Public domain in the USA". OPTED is
public domain, and asks that redistributed content stays free, stays available as
plain text at no cost, and credits OPTED, Project Gutenberg and the 1913
*Webster's Unabridged Dictionary*. See the
[OPTED page](https://www.mso.anu.edu.au/~ralph/OPTED/) and the
[Project Gutenberg License](https://www.gutenberg.org/policy/license.html).
