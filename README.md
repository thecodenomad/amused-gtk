# amused-gtk

An very first pass gtk frontend for amused library.

### Build

Easiest: Use GNOME Builder to build

If you use `foundry`:

`foundry init && foundry run`

### Local Dev (backend)

#### Install dependencies

`poetry install`

#### Run example:

`poetry run python amused_example.py`

#### Scripts

Updates to the `pyproject.toml` requires the `update_python_dependencies.sh` to update the flatpak dependencies.
