# Shape Analysis 101

### To use
1. Create conda environment from yaml file:
    `conda env create -f shapeanalysis101_env.yml`

2. Activate environment:
    `source activate shapeanalysis101`
    
3. Start jupyter server:
    `jupyter notebook`
    
4. Explore and modify the various `.ipynb` notebooks

5. Answer sheets can be found on the `solutions` folder. To run them, simply move the [Solution] notebook you would like to run to the base `shapeanalysis101` repository.

### Troubleshooting

If the `shapeanalysis101_env.yml` fails to install for whichever reason, try and use `env_wo_gpu.yml` instead as follows:

```
conda env create -f env_wo_gpu.yml
source activate shapeanalysis101_2
```

You can then resume at step 3 above.

### To cite
If you reuse pieces of code from this course for your own research, please acknowledge them as follows:
* For the spline models library `spline_curve_model`:
> Uhlmann group's Spline Fitting Toolbox v1.0

* For `2B - Continuous Shape Analysis`: 
> Song, A., Uhlmann, V., Fageot, J., & Unser, M. (2020). Dictionary learning for two-dimensional Kendall shapes. SIAM Journal on Imaging Sciences, 13(1), 141-175.

* For `3 - Shape Embeddings`: 
> Hugger, J., Uhlmann, V., (2021). Shape Embeddings for Biological Morphology Quantification. Preprint.


### Further reading
* Dryden, I. L., & Mardia, K. V. (2016). Statistical shape analysis: with applications in R. John Wiley & Sons.
* Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. & Krishnan, D. (2020). Supervised contrastive learning. arXiv:2004.11362.

### Acknowledgements
Part of the material from this course was adapted from Paula Balcells' Bachelor Thesis work carried out in the Uhlmann group. We also thank Anna Song for useful discussions and Jean Feydy for sharing his excellent teaching material (https://www.jeanfeydy.com/Teaching/), which inspired part of this course's structure. 
