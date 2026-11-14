# numcompute_stream benchmarks

## A. Loop vs vectorised

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| best_split (2,000 rows x 5 features) | 0.002561 | 0.01572 | 6.14x |
| tree predict (50,000 rows, depth 8) | 0.002596 | 0.04903 | 18.89x |
| running mean/var (50 chunks, 100,000 x 3) | 0.01367 | 0.1656 | 12.11x |
| confusion matrix (20 chunks, 200,000 labels) | 0.008478 | 0.03502 | 4.13x |

Each pair is checked for identical results by `numcompute.benchmarking.compare` before the speedup is reported. Mean of 3 timed runs after 1 warm-up.

## B1. buffer_size sweep - synthetic stream (10,000 rows, chunks of 100, drift at row 5,000)

| buffer_size | final_cum_acc | mean_rolling_acc | post_drift_acc | first_scored_chunk | n_retrains | mean_retrain_ms | total_retrain_ms | max_memory_kib |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.815 | 0.816 | 0.782 | 1 | 100 | 2.894 | 289.439 | 2.742 |
| 250 | 0.845 | 0.845 | 0.756 | 3 | 33 | 7.187 | 237.157 | 16.492 |
| 500 | 0.858 | 0.858 | 0.680 | 5 | 20 | 10.642 | 212.848 | 27.711 |
| 1000 | 0.854 | 0.854 | 0.521 | 10 | 10 | 16.604 | 166.036 | 56.039 |
| 2000 | 0.814 | 0.814 | 0.546 | 20 | 5 | 28.379 | 141.893 | 111.430 |

## B2. buffer_size sweep - Housing.csv (545 rows, chunks of 25, shuffled with seed 0)

| buffer_size | final_cum_acc | mean_rolling_acc | first_scored_chunk | n_retrains | mean_retrain_ms | total_retrain_ms | max_memory_kib |
|---|---|---|---|---|---|---|---|
| 50 | 0.673 | 0.680 | 2 | 10 | 2.029 | 20.293 | 12.586 |
| 100 | 0.719 | 0.720 | 4 | 5 | 4.095 | 20.474 | 21.398 |
| 200 | 0.725 | 0.720 | 8 | 2 | 8.003 | 16.007 | 48.805 |
| 400 | 0.731 | 0.732 | 16 | 1 | 15.920 | 15.920 | 99.609 |

**Environment**

- Date: 2026-11-14
- Python: 3.14.6
- NumPy: 2.5.2
- OS: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 204 Stepping 3, GenuineIntel (6 logical cores)
