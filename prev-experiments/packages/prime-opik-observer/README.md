# Prime Opik observer

This is the isolated telemetry environment used by the experiment wrappers. It keeps Opik and its dependency graph outside the PrimeRL GPU environment.

```bash
uv sync --locked --python 3.12
uv run --no-sync python opik_bridge.py --help
```

At runtime, credentials are read from a mode-`600` file outside the repository. Never commit that file. Required variables are `OPIK_API_KEY`, `OPIK_URL_OVERRIDE`, and `OPIK_WORKSPACE`.

