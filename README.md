<h2 align="center">CVPR 2025<br>Not Only Text: Exploring Compositionality of Visual Representations
in Vision-Language Models</h2>

<p align="center">
  <a href="https://openreview.net/profile?id=~Davide_Berasi1">Davide Berasi</a>,
  <a href="https://scholar.google.com/citations?user=SxQwDD8AAAAJ&authuser=1">Matteo Farina</a>, 
  <a href="https://scholar.google.com/citations?user=bqTPA8kAAAAJ&authuser=1">Massimiliano Mancini</a>, 
  <a href="https://scholar.google.com/citations?user=xf1T870AAAAJ&authuser=1">Elisa Ricci</a> and
  <a href="https://scholar.google.it/citations?user=7cgpfGYAAAAJ&hl=1">Nicola Strisciuglio</a>
</p>

>**Abstract.** *Vision-Language Models (VLMs) learn a shared feature space for text and images, enabling the comparison of inputs of different modalities. While prior works demonstrated that VLMs organize natural language representations into regular structures encoding composite meanings, it remains unclear if compositional patterns also emerge in the visual embedding space. In this work, we investigate compositionality in the image domain, where the analysis of compositional properties is challenged by noise and sparsity of visual data. We address these problems and propose a framework, called Geodesically Decomposable Embeddings (GDE), that approximates image representations with geometry-aware compositional structures in the latent space. We demonstrate that visual embeddings of pre-trained VLMs exhibit a compositional arrangement, and evaluate the effectiveness of this property in the tasks of compositional classification and group robustness. GDE achieves stronger performance in compositional classification compared to its counterpart method that assumes linear geometry of the latent space. Notably, it is particularly effective for group robustness, where we achieve higher results than task-specific solutions. Our results indicate that VLMs can automatically develop a human-like form of compositional reasoning in the visual domain, making their underlying processes more interpretable.*


## Set up
Clone the repo and enter in it:
```
git clone https://github.com/BerasiDavide/vlm_image_compositionality
cd vlm_image_compositionality
```
Create the conda environment and activate it:
```
conda env create -f environment.yml
conda activate img_comp
```

## Compositional Classification

- Download the MIT-States and UT-Zappos datasets:
```
sh utils/download_classification_data.sh
```
- Compute the VLM embeddings and store them in the correct directory:
```
python -m datasets.compute_embeddings 'mit-states' 'ViT-L-14' 'openai'
python -m datasets.compute_embeddings 'ut-zappos' 'ViT-L-14' 'openai'
```
RK: Any pre-trained OpenCLIP model can be used ([available models](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb#scrollTo=uLFS29hnhlY4)).


- Run a classification experiment with `classification.py`. For example, you can perform classification on Ut-Zappos, in open world scenario, with GDE decompositions of ('ViT-L-14', 'openai') image embeddings as classifiers, with:
```
python -m classification \
  --dataset 'ut-zappos' \
  --model_architecture 'ViT-L-14' \
  --model_pretraining 'openai' \
  --experiment_name 'GDE' \
  --modality_IW 'image' \
  --open_world
```
RK: This applies a uniform noise distribution for the decomposition. Use instead `classification_optim.py` to leverage the CLIP image-to-text distribution with optimal temperature.


## Group Robustness

- Download the Waterbirds dataset:
```
sh utils/download_waterbirds.sh
``` 
Download the CelebA dataset at [this link](https://www.kaggle.com/jessicali9530/celeba-dataset) and position it in the ```data/``` folder. Ensure to have this structure:
```
data 
└───celeba-dataset
    │   list_attr_celeba.csv
    |   list_eval_partition.csv
    └───img_align_celeba
        └───img_align_celeba
```
Then run:
```
python datasets/reorganize_celebA.py
```
- Compute the VLM embeddings and store them in the correct directory:
```
python -m datasets.compute_embeddings 'waterbirds' 'ViT-L-14' 'openai'
python -m datasets.compute_embeddings 'celebA' 'ViT-L-14' 'openai'
```
- Run a group robustness experiment with `group_robustness.py`. For example, you can evaluate group robustness on Waterbirds, with GDE decompositions of ('ViT-L-14', 'openai') image embeddings, with:
```
python -m group_robustness \
  --dataset 'waterbirds' \
  --model_architecture 'ViT-L-14' \
  --model_pretraining 'openai' \
  --experiment_name 'GDE' \
  --modality_IW 'image'
```
RK: Similarly to classification, use `group_robustness_optim.py` to apply the CLIP image-to-text distribution with optimal temperature as the noise weights.

## Citation
Please cite this work as follows if you find it useful!
```bibtex
@article{berasi2025not,
  title={Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models},
  author={Davide Berasi and Matteo Farina and Massimiliano Mancini and Elisa Ricci and Nicola Strisciuglio},
  journal={arXiv preprint arXiv:2503.17142},
  year={2025}
}
```