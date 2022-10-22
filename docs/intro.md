#  OmniGibson: the Interactive Gibson Environment

### Large Scale Interactive Simulation Environments for Robot Learning

OmniGibson, the Interactive Gibson Environment, is a simulation environment providing fast visual rendering and physics simulation (based on Bullet). 
It is packed with a dataset with hundreds of large 3D environments reconstructed from real homes and offices, and interactive objects that can be pushed and actuated. 
OmniGibson allows researchers to train and evaluate robotic agents that use RGB images and/or other visual sensors to solve indoor (interactive) navigation and mobile manipulation tasks such as opening doors, picking and placing objects, or searching for objects.
With the latest extension, OmniGibson 2.0 supports new types of [object state changes](extended_states.md) (cook, soak, slice, freeze, etc), that can enable new types of simulated activities!
OmniGibson implements all features required to evaluate AI solutions in the BEHAVIOR benchmark: [sampling logic activity descriptions](sampling.md), [checking logic states](extended_states.md), connecting to the BEHAVIOR dataset of 3D objects and evaluating BEHAVIOR metrics (information about the [dataset](https://stanfordvl.github.io/behavior/objects.html) and the [metrics](https://stanfordvl.github.io/behavior/metrics.html) can be found in the documentation of the BEHAVIOR repository).


### Citation
If you use OmniGibson or its assets and models, consider citing the following publications:

```
@inproceedings{shen2021omnigibson,
      title={OmniGibson 1.0: a Simulation Environment for Interactive Tasks in Large Realistic Scenes}, 
      author={Bokui Shen and Fei Xia and Chengshu Li and Roberto Mart\'in-Mart\'in and Linxi Fan and Guanzhi Wang and Claudia Pérez-D'Arpino and Shyamal Buch and Sanjana Srivastava and Lyne P. Tchapmi and Micael E. Tchapmi and Kent Vainio and Josiah Wong and Li Fei-Fei and Silvio Savarese},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems},
      year={2021},
      pages={accepted},
      organization={IEEE}
}
```

```
@misc{li2021omnigibson,
      title={OmniGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks}, 
      author={Chengshu Li and Fei Xia and Roberto Martín-Martín and Michael Lingelbach and Sanjana Srivastava and Bokui Shen and Kent Vainio and Cem Gokmen and Gokul Dharan and Tanish Jain and Andrey Kurenkov and Karen Liu and Hyowon Gweon and Jiajun Wu and Li Fei-Fei and Silvio Savarese},
      year={2021},
      eprint={2108.03272},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

### Code Release
The GitHub repository of OmniGibson can be found [here](https://github.com/StanfordVL/OmniGibson). Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. 

### Documentation
The documentation for OmniGibson can be found [here](http://svl.stanford.edu/omnigibson/docs/). It includes installation guide (including data download instructions), quickstart guide, code examples, and APIs.

If you want to know more about OmniGibson, you can also check out [our webpage](http://svl.stanford.edu/omnigibson), the [OmniGibson 2.0 arxiv preprint](https://arxiv.org/abs/2108.03272) and the [OmniGibson 1.0 arxiv preprint](https://arxiv.org/abs/2012.02924).
