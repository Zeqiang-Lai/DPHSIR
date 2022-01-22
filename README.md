# DPHSIR

[Paper]() | [Pretrained Model]()

**Deep Plug-and-Play Prior for Hyperspectral Image Restoration (Neurocomputing 2022)**

Zeqiang Lai, Kaixuan Wei, Ying Fu

## :sparkles: News

- **In Progress**: Release training code of GRUNet.
- **2021-01-22**: Add a command line client for testing single image or list of images in folders.
- **2021-01-21**: Release demo code for each task.

## Requirement

- Pytorch >= 1.8
- OpenCV

## Getting Started

1.  **Install the requirments**

```shell
conda install -c conda-forge opencv
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

2. **Clone the repo**

```shell
git clone https://github.com/Zeqiang-Lai/DPHSIR.git
```

3. **Run cli or playgrounds**

```shell
# run cli
python cli/main.py -i [input_path] [task]
# run playground
python playgrounds/deblur.py
```

## Citation

If you find our work useful for your research, please consider citing our paper :)

```bibtex
@article{lai2022deep,
  title={Deep Plug-and-Play Prior for Hyperspectral Image Restoration},
  author={Lai, Zeqiang and Wei, Kaixuan and Fu, Ying},
  journal={Neurocomputing},
  volume={},
  number={},
  pages={},
  year={2022},
  publisher={Elsevier}
}
```

## Acknowledgement

- We use some code from [DPIR](https://github.com/cszn/DPIR).
