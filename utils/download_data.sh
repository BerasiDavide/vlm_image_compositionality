CURRENT_DIR=$(pwd)

mkdir -p data
cd data

# Download datasets and splits
wget -c http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip -O mitstates.zip
wget -c http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip -O utzap.zip
wget -c https://www.senthilpurushwalkam.com/publication/compositional/compositional_split_natural.tar.gz -O compositional_split_natural.tar.gz

# MIT-States
unzip -q mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
#rename "s/ /_/g" mit-states/images/*   ## Doesn't work everywhere
for file in mit-states/images/*; do mv "$file" "$(echo "$file" | sed 's/ /_/g')"; done

# UT-Zappos50k
unzip -q utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/

# Download new splits for Purushwalkam et. al
tar -zxvf compositional_split_natural.tar.gz

rm -r mitstates.zip utzap.zip compositional_split_natural.tar.gz

cd $CURRENT_DIR
python3 datasets/reorganize_utzap.py

mv data/ut-zap50k data/ut-zappos
