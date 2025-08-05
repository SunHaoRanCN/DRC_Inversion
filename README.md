# Audio Dynamic Range Compression (DRC) inversion
Official repository of the paper: Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Resoration Audio Dynamics

## Authors
- **Haoran Sun** <haoran.sun@etu-upsaclay.fr>
- **Dominique Fourer** <dominique.fourer@univ-evry.fr>
- **Hichem Maaref** <hichem.maaref@univ-evry.fr>

## Affiliations
Laboratoire IBISC (EA 4526), Univ. Evry Paris-Saclay, Évry-Courcouronnes, France

## Setup
This repository requires Python 3.10+ and Pytorch 1.10+. Other packages are listed in 'requirements.txt'.
To install the requirements in your environment:
```
pip install -r requirements.txt
```

## Train
To retrain the model, run:
```
python inference.py -t [task (classification or regression] -m train -i [input path]
```

## Evaluation
Download the pretrained AST and MEE models [GoogleDrive](https://drive.google.com/drive/folders/1LwsGQVpnZOczGa8e-NY45pMHWGjAP6aS?usp=sharing).
To evaluate the pretrained model, run:
```
python inference.py -t [task (classification or regression)] -m eval -i [input path] -o [output path]
```

## License
Distributed under the MIT License.


## Acknowledgements
- [PyTorch](https://pytorch.org/)
