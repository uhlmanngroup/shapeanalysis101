mkdir data
cd data

mkdir BBBC010
cd BBBC010

mkdir images
wget https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v2_images.zip
unzip BBBC010_v2_images.zip -d images/
rm BBBC010_v2_images.zip

wget https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_foreground_eachworm.zip
unzip BBBC010_v1_foreground_eachworm.zip
mv BBBC010_v1_foreground_eachworm masks
rm BBBC010_v1_foreground_eachworm.zip

cd ..
mkdir BBBC020
cd BBBC020

wget https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip
unzip BBBC020_v1_images.zip 
mv BBBC020_v1_images images
rm BBBC020_v1_images.zip

wget https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_cells.zip
unzip BBBC020_v1_outlines_cells.zip
mv BBC020_v1_outlines_cells masks_cells
rm BBBC020_v1_outlines_cells.zip 

mkdir masks_nuclei 
wget https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_nuclei.zip
unzip BBBC020_v1_outlines_nuclei.zip
mv BBC020_v1_outlines_nuclei masks_nuclei
rm BBBC020_v1_outlines_nuclei.zip 

cd ..

