## Simulate test data

Download SONICOM HRTFs and put them in the same folder (e.g., `./data/`).

```bash
python simulate_data.py data sim_lv1 --regexp "*CompMinPhase_48kHz.sofa" --lvl 1
```

## Upsample the low-resolution data

### 100 points

```bash
python upsample.py sim_lv1 pred_lv1 --sph-order 15
```

### 19 points

```bash
python upsample.py sim_lv2 pred_lv2 --sph-order 4
```

### 5 points

```bash
python upsample.py sim_lv3 pred_lv3 --sph-order 4 --use-rigid-toa --minphase
```

### 3 points

```bash
python upsample.py sim_lv4 pred_lv4 --sph-order 2 --use-rigid-toa --lr-augment   
```

## Evaluate the results

```bash
python eval.py pred_lv1 data
python eval.py pred_lv2 data
python eval.py pred_lv3 data
python eval.py pred_lv4 data
```
