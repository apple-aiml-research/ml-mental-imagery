# The website

The [project's website](https://apple-aiml-research.github.io/ml-mental-imagery/) for
"Do multimodal models imagine electric sheep?". The `website` branch tracks the code on `main` and adds the static site
in `docs/`, which GitHub Pages serves as it is (`docs/.nojekyll`), with no build step. Website changes belong in `docs/`;
keep the other files synchronized with `main`.

## Looking at it locally

```
python3 -m http.server --directory docs 8000    # then open http://localhost:8000
```

## Publishing it

Commit the changes to `docs/` on this branch and push it; Pages serves the new commit. Pages needs pointing at the
branch once, by an admin of the repository: in its settings, Pages, deploy from a branch, `website`, folder `/docs`.
Or with the GitHub CLI (`-X PUT` instead to change it once Pages is on):

```
gh api -X POST repos/apple-aiml-research/ml-mental-imagery/pages \
  -f "source[branch]=website" -f "source[path]=/docs"
```
