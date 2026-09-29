# Training and evaluation

See the main README for [installation](../README.md#installation), [datasets](../README.md#datasets), [training recipes](../README.md#training), and [evaluation](../README.md#evaluation).

Use the scripts in [scripts/train](../scripts/train) and [scripts/test](../scripts/test) for each model's settings. Argument descriptions, base defaults, and supported choices are available through `--help`:

```bash
python -m visionhope.tasks.cli train classification --help
python -m visionhope.tasks.cli inference classification --help
```

Replace `classification` with `detection` or `segmentation` for the other tasks.
