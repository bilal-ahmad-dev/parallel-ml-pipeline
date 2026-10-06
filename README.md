# Parallel ML Pipeline: Serial vs Multiprocessing vs MPI

Builds the same machine learning pipeline three ways on a synthetic 2-million-row dataset and measures speedup and efficiency on a laptop (Intel i7-1355U, WSL2).

## Layout
- `data/generate_data.py`: makes the dataset (with injected duplicates and missing values)
- `serial/pipeline.py`: serial baseline (load, clean, features, linear regression)
- `parallel_multiprocessing/`: `pipeline_mp.py` (data pipeline) and `pipeline_select.py` (12-model selection)
- `distributed_mpi/pipeline_mpi.py`: the model selection with MPI (mpi4py + Open MPI)
- `benchmarks/`: `benchmark.py` and the result files

## Run
```bash
python3 data/generate_data.py --rows 2000000 --out data/dataset.csv
python3 serial/pipeline.py --data data/dataset.csv
cd parallel_multiprocessing && python3 pipeline_select.py --workers 8 --subsample 50000
cd ../distributed_mpi && mpirun --use-hwthread-cpus --bind-to none -n 8 python3 pipeline_mpi.py --subsample 50000
```

## Results (median of repeated runs)
| Task | Serial / 1 worker | Best parallel | Speedup |
|---|---|---|---|
| Data pipeline | 12.0 s | 4.45 s (4 workers) | 2.7x (only ~1.2x is pure parallelism) |
| Model selection, multiprocessing | 47.0 s | 10.9 s (8 workers) | 4.3x |
| Model selection, MPI | 44.8 s | 10.3 s (8 processes) | 4.3x |

All versions give the same RMSE (5.1424 for the data pipeline, 5.7759 for model selection).

## Takeaways
- Serial parts (CSV load, preparation) limit speedup (Amdahl's Law).
- With only 12 models, more than 8 workers does not help (load imbalance).
- MPI communication cost stays below about 5% of run time.
