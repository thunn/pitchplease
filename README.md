# (WIP) pitchplease
A minimal package for when you have some audio and want to know the pitch – just remember to say please!

# Project

(Planned) Supported pitch methods:
- [ ] [torchcrepe](https://github.com/maxrmorrison/torchcrepe)
- [ ] [FCPE](https://github.com/CNChTu/FCPE)
- [ ] [RMVPE](https://github.com/Dream-High/RMVPE)

# Development 

```
uv pip install -e .
pre-commit install
```

## Run tests

```
uv sync --all-groups
uv run pytest
```
