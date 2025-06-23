# Assets

## Introduction

Assets includes necessary files for constructing a scene in iGibson simulator. The files include robot models, interactive objects, articulated objects and mesh files for tests. These files are too large to include in a version control system so we distribute them separately. The assets file can be downloaded to the path set in `your_installation_path/igibson/global_config.yaml` (default to `your_installation_path/igibson/data/assets`) with running 

```bash 
python -m igibson.utils.assets_utils --download_assets
```

The folder structure will look like below (in the future we might add more models):
```
assets
├── models
│   ├── ant
│   ├── atlas
│   ├── botlab
│   ├── cabinet
│   ├── cabinet2
│   ├── cassie
│   ├── fetch
│   ├── humanoid
│   ├── husky
│   ├── jr1_urdf
│   ├── jr2_urdf
│   ├── kinova
│   ├── laikago
│   ├── locobot
│   ├── mjcf_primitives
│   ├── quadrotor
│   ├── quadruped
│   ├── rbo
│   ├── scene_components
│   ├── turtlebot
│   └── ycb
├── networks
│   └── model.pth
├── test
│   └── mesh
└── example_configs
    └── {}.yaml (deprecated, will be removed)

```
`models` contains robot models and interactive objects, `networks` contain the neural network filler used in Gibson V1, `test` contains files for performing tests of installation. 

## Models

The robots folders correspond to [robot](./robots.md) models. 

| Agent Name     | Folder | 
|:-------------: | :-------------: |
| Mujoco Ant      |   `ant` |
| Mujoco Humanoid |   `humanoid` | 
| Husky Robot     |   `husky` |
| Minitaur Robot  |   `quadruped` |
| Quadrotor       |   `quadrotor` |
| TurtleBot       |   `turtlebot` |
| Freight         |  `fetch` |
| Fetch           |  `fetch` |
| JackRabbot      |  `jr2_urdf` |
| LocoBot         |   `locobot` |

We also include [YCB objects](http://www.ycbbenchmarks.com/object-models/) in `ycb` folder, [RBO models](https://tu-rbo.github.io/articulated-objects/) in `rbo` folder, and a few commonly used primitives for home environments such as doors (in `scene_components`) and cabinets (in `cabinet` and `cabinet2`). You can refer to [objects](./objects.md) page to see how to use these models in gibson scenes. Don't forget to cite related papers when using these assets.

## Pretrained network

`networks` folder contains a pretrained network checkpoint `model.pth`, which is the "goggle" network used in Gibson V1. It is used to fix artifacts caused by imperfect reconstruction. In previous version we used image based rendering and there are more visible artifacts, therefore the "goggle" network is important. In this version, mesh rendering creates photorealstic visuals, so we recommend not using it (to gain more framerate). 

## Test meshes

In `test` folder we save a mesh file to test the renderer is compiled correctly, it is used by `test/test_render.py`.

## Example configs

Sample config files to be used with the demo.