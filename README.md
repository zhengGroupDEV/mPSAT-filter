# mPSAT-filter

Microplastics (MPs) spectral reconstruction and identification using machine learning.

## Development

1. install python-poetry
2. `git clone https://github.com/zhengGroupDEV/mPSAT-filter.git`, `cd mPSAT-filter`
3. install dependencies, run `pdm install` and activate your virtual environment
4. generate dataset `python -m mpfilter.ds_generate -i data/ds_mpc_mpb_24.ftr -o data/dataset/ds_1k_0.2`
5. train or evaluate using powershell scripts in `scripts`, e.g., to train reconstruction model AE `./scripts/train_seq.ps1 -model ae`, you can change configuration in script and config file.

## Dataset
You can find dataset in `data` dir, the datasets used in this project are from:

[1] Primpke S, Wirth M, Lorenz C, et al. Reference database design for the automated analysis of microplastic samples based on Fourier transform infrared (FTIR) spectroscopy. Analytical and Bioanalytical Chemistry, 2018, 410(21): 5131-5141. DOI: 10.1007/s00216-018-1156-x

[2] Chabuka B K, Kalivas J H. Application of a hybrid fusion classification process for identification of microplastics based on Fourier transform infrared spectroscopy. Applied Spectroscopy, 2020, 74(9): 1167-1183. DOI: 10.1177/0003702820923993

[3] Suja Sukumaran (Thermo Fisher Scientific)

[4] Liu Y, Yao W, Qin F, et al. Spectral classification of large-scale blended (micro)plastics using FT-IR raw spectra and image-based machine learning[J]. Environmental Science & Technology, 2023, 57(16): 6656-6663. DOI: 10.1021/acs.est.2c08952

