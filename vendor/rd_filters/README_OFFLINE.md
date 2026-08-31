# rd_filters — offline / DGX install

Bundled copy of [PatWalters/rd_filters](https://github.com/PatWalters/rd_filters).

## On laptop (with internet)

```bash
cd reinvent-local-main
git clone https://github.com/PatWalters/rd_filters vendor/rd_filters
rm -rf vendor/rd_filters/.git   # optional — avoids nested repo
```

Copy the whole `reinvent-local-main` folder (or just `vendor/rd_filters/`) to DGX via WinSCP.

## On DGX (no git+ pip needed)

```bash
conda activate reinvent_qsar
cd ~/path/to/reinvent-local-main
pip install -e ./vendor/rd_filters
```

## Run filter script

```bash
python Preprocess/scripts/apply_rd_filters.py \
  --input_csv your_input.csv \
  --output_prefix results/your_run_rd
```

Outputs: `your_run_rd_flagged.csv` and `your_run_rd_passed.csv`
