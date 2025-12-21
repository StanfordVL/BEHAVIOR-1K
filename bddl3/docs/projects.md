Projects using Gibson/iGibson
===================================

It is exciting to see people using Gibson Environment in embodied AI research. Here is a list of projects using Gibson v1 or iGibson:

- K. Chen, J. P. de Vicente, G. Sepulveda, F. Xia, A. Soto, M. Vazquez, and S. Savarese. [A behavioral approach to visual navigation with graph localization networks](https://arxiv.org/pdf/1903.00445.pdf). In RSS, 2019.
- Hirose, Noriaki, et al. [Deep Visual MPC-Policy Learning for Navigation.](https://arxiv.org/pdf/1903.02749.pdf) arXiv preprint arXiv:1903.02749 (2019). IROS 2019.
- Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox. [Scaling Local Control to Large-Scale Topological Navigation](https://arxiv.org/pdf/1909.12329.pdf)
- X. Meng, N. Ratliff, Y. Xiang, and D. Fox, [Neural autonomous navigation with riemannian motion policy,](https://arxiv.org/pdf/1904.01762.pdf) in IEEE International Conference on Robotics and Automation (ICRA), 2019.
- Kang, Katie, et al. [Generalization through simulation: Integrating simulated and real data into deep reinforcement learning for vision-based autonomous flight.](https://arxiv.org/abs/1902.03701) arXiv preprint arXiv:1902.03701 (2019). ICRA 2019.
- Sax, Alexander, et al. [Mid-level visual representations improve generalization and sample efficiency for learning active tasks.](https://arxiv.org/pdf/1812.11971.pdf) arXiv preprint arXiv:1812.11971 (2018).
- Shen, William B., et al. [Situational Fusion of Visual Representation for Visual Navigation.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Situational_Fusion_of_Visual_Representation_for_Visual_Navigation_ICCV_2019_paper.pdf) arXiv preprint arXiv:1908.09073 (2019). ICCV 2019.
- Li, Chengshu, et al. [HRL4IN: Hierarchical Reinforcement Learning for Interactive Navigation with Mobile Manipulators.](https://arxiv.org/pdf/1910.11432.pdf) arXiv preprint arXiv:1910.11432 (2019).
- Watkins-Valls, David, et al. [Learning Your Way Without a Map or Compass: Panoramic Target Driven Visual Navigation.](https://arxiv.org/pdf/1909.09295.pdf) arXiv preprint arXiv:1909.09295 (2019).
- Akinola, Iretiayo, et al. [Accelerated Robot Learning via Human Brain Signals.](https://arxiv.org/pdf/1910.00682.pdf) arXiv preprint arXiv:1910.00682(2019).
- Xia, Fei, et al. [Interactive Gibson: A Benchmark for Interactive Navigation in Cluttered Environments.](https://arxiv.org/pdf/1910.14442.pdf) arXiv preprint arXiv:1910.14442 (2019).
- Pérez-D'Arpino, et al. [Robot Navigation in Constrained Pedestrian Environments using Reinforcement Learning](https://arxiv.org/pdf/2010.08600.pdf). Preprint arXiv:2010.08600, 2020.
- Andrey Kurenkov, et al[Visuomotor Mechanical Search: Learning to Retrieve Target Objects in Clutter](https://arxiv.org/abs/2008.06073). IROS 2020.
- Andrey Kurenkov, et al. [Multi-Layer Semantic and Geometric Modeling with Neural Message Passing in 3D Scene Graphs for Hierarchical Mechanical Search](https://ai.stanford.edu/mech-search/hms/).
- Joanne Truong, et al. [Learning Navigation Skills for Legged Robots with Learned Robot Embeddings](https://arxiv.org/pdf/2011.12255.pdf).




These papers tested policies trained in Gibson v1 on real robots in the physical world:

- Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox. [Scaling Local Control to Large-Scale Topological Navigation](https://arxiv.org/pdf/1909.12329.pdf)
- X. Meng, N. Ratliff, Y. Xiang, and D. Fox, [Neural autonomous navigation with riemannian motion policy,](https://arxiv.org/pdf/1904.01762.pdf) in IEEE International Conference on Robotics and Automation (ICRA), 2019.
- Kang, Katie, et al. [Generalization through simulation: Integrating simulated and real data into deep reinforcement learning for vision-based autonomous flight.](https://arxiv.org/abs/1902.03701) arXiv preprint arXiv:1902.03701 (2019). ICRA 2019.
- Hirose, Noriaki, et al. [Deep Visual MPC-Policy Learning for Navigation.](https://arxiv.org/pdf/1903.02749.pdf) arXiv preprint arXiv:1903.02749 (2019). IROS 2019.


If you use Gibson, iGibson or their assets, please consider citing the following papers for iGibson, the Interactive Gibson Environment:

```
@article{shenigibson,
  title={iGibson, a Simulation Environment for Interactive Tasks in Large Realistic Scenes},
  author={Shen*, Bokui and Xia*, Fei and Li*, Chengshu and Mart{\'i}n-Mart{\'i}n*, Roberto and Fan, Linxi and Wang, Guanzhi and Buch, Shyamal and D’Arpino, Claudia and Srivastava, Sanjana and Tchapmi, Lyne P and  Vainio, Kent and Fei-Fei, Li and Savarese, Silvio},
  journal={arXiv preprint arXiv:2012.02924},
  year={2020}
}
```


````
@article{xia2020interactive,
           title={Interactive Gibson Benchmark: A Benchmark for Interactive Navigation in Cluttered Environments},
           author={Xia, Fei and Shen, William B and Li, Chengshu and Kasimbeg, Priya and Tchapmi, Micael Edmond and Toshev, Alexander and Mart{\'\i}n-Mart{\'\i}n, Roberto and Savarese, Silvio},
           journal={IEEE Robotics and Automation Letters},
           volume={5},
           number={2},
           pages={713--720},
           year={2020},
           publisher={IEEE}
         }
````


and the following paper for Gibson v1:

````text
@inproceedings{xia2018gibson,
  title={Gibson env: Real-world perception for embodied agents},
  author={Xia, Fei and Zamir, Amir R and He, Zhiyang and Sax, Alexander and Malik, Jitendra and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9068--9079},
  year={2018}
}
````
