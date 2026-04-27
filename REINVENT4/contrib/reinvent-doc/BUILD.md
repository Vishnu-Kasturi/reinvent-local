# Build docs locally

From inside `doc/`:

```bash
pip install -r requirements.txt
sphinx-build -b html . _build/html
python -m http.server 8000 --directory _build/html
```

Open `http://localhost:8000` in a browser.

