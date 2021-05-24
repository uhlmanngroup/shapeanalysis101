# Shape Analysis 101

### To retrieve data
```diff
@@ TODO: write a bash script to automate this @@
```

1. Create a `data` folder in the repository

2. Create two subfolders `data/BBBC020` and `data/BBBC010`

3. Populate 
    * a subfolder `data/BBBC010/images` with https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip 
    * a subfolder `data/BBBC010/masks` with https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip
    * a subfolder `data/BBBC020/images` with https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip
    * a subfolder `data/BBBC020/masks_cells` https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_cells.zip
    * a subfolder `data/BBBC020/masks_nuclei` with https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_nuclei.zip

### To use
1. Create conda environment from yaml file:
    `conda env create -f shapeanalysis101_env.yml`

2. Activate environment:
    `source activate shapeanalysis101`
    
3. Start jupyter server:
    `jupyter notebook`
    
4. Explore and modify the various `.ipynb` notebooks

### Sources
* Dryden, I. L., & Mardia, K. V. (2016). Statistical shape analysis: with applications in R. John Wiley & Sons.
* Song, A., Uhlmann, V., Fageot, J., & Unser, M. (2020). Dictionary learning for two-dimensional Kendall shapes. SIAM Journal on Imaging Sciences, 13(1), 141-175.

### Acknowledgements
We thank Anna Song for useful discussions and Jean Feydy for sharing his excellent teaching material (https://www.jeanfeydy.com/Teaching/), which inspired part of this repository's content.
