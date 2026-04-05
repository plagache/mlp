## Other commands

```bash
uv python list
uv pip list
du -sh .venv
```

### Monitory GPU

```bash
watch -n0.1 nvidia-smi
watch -n0.1 rocm-smi
```

### Open Machine directory on a http server

```bash
uv run python -m http.server 3635 --bind 0.0.0.0
```
open in `http://machine_ip:3635`
