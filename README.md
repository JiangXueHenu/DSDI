<!--
 * @Description:
 * @Author: Xue Jiang
 * @Email: jx19303768670@gmail.com

-->

It's a pytorch implementation of paper "Boundary-Enhanced Time Series Data Imputation with
Long-Term Dependency Diffusion Models" .

## Requirements

```shell
pip install -r requirements.txt
```

## Datasets

- ETT
- AQI
- DACMI

## How to run

### Train DSDI from scratch

```shell
bash ./scripts/ETT_point.sh
bash ./scripts/ETT_block.sh
```

### Test DSDI from trained model

First, set the scratch is `False` in scripts.
Then, run these scripts.

```shell
bash ./scripts/ETT_point.sh
bash ./scripts/ETT_block.sh
```
