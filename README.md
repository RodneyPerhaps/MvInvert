## Official PyTorch implementation of Towards High-Fidelity Face Self-occlusion Recovery via Multi-view Residual-based GAN Inversion ##

<p align="center"> 
<img src="data/eyecandy.png">
</p>

### Environment setup ###

```
conda env create -f environment.yml
```

### Run inference on portrait images ###

```
python main_poisson.py --img data/MoFA-test/5.jpg 
```

### Citation ###
If you find this project is useful to your research, please cite the following paper:

```
@inproceedings{chen2022mvinvert,
  title={Towards High-Fidelity Face Self-Occlusion Recovery via Multi-View Residual-Based GAN Inversion},
  author={Chen Jinsong, Han Hu, and Shan Shiguang},
  booktitle={AAAI},
  pages={294--302},
  year={2022}
}
```
