# Datasets

In this page you will find information about:

- [How to download OmniGibson 2.0 scenes and the BEHAVIOR Dataset of Objects](#download-omnigibson-2-0-scenes-and-the-behavior-dataset-of-objects)
- [How to download OmniGibson 1.0 scenes](#download-omnigibson-1-0-scenes)
- [How to download Gibson and Stanford 2D-3D-Semantics scenes](#download-gibson-and-stanford-2d-3d-semantics-scenes)
- [How to download Matterport3D scenes](#download-matterport3d-scenes)

## Download OmniGibson 2.0 Scenes and the BEHAVIOR Dataset of Objects

What will you download?
- **OmniGibson 2.0 Dataset of Scenes**: New versions of the fully interactive scenes, more densely populated with objects.
- **BEHAVIOR Object Dataset**: Dataset of object models annotated with physical and semantic properties. The 3D models are free to use within OmniGibson 2.0 for BEHAVIOR (due to artists' copyright, models are encrypted and allowed only to be used with OmniGibson 2.0). You can download a bundle of the OmniGibson 2.0 dataset of scenes and the BEHAVIOR dataset of objects here.

To download both in a bundle, you need to follow the following steps:
- Fill out the license agreement in this [form](https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform)
- You will receive a key (omnigibson.key). Move it into the subfolder of the OmniGibson repository that contains the dataset, for example, OmniGibson/omnigibson/data
- Download the behavior data bundle (og_dataset) from [here](https://storage.googleapis.com/gibson_scenes/behavior_data_bundle.zip)
- Unzip og_dataset into the folder: `unzip behavior_data_bundle.zip -d OmniGibson/omnigibson/data`

After this process, you will be able to sample and use the scenes and objects in OmniGibson, for example, to evaluate your embodied AI solutions in the [BEHAVIOR benchmark](https://behavior.stanford.edu/).

## Download OmniGibson 1.0 Scenes

What will you download?
- **OmniGibson 1.0 Dataset of Scenes**: We annotated fifteen 3D reconstructions of real-world scans and converted them into fully interactive scene models. In this process, we respect the original object-instance layout and object-category distribution. The object models are extended from open-source datasets ([ShapeNet Dataset](https://www.shapenet.org/), [Motion Dataset](http://motiondataset.zbuaa.com/), [SAPIEN Dataset](https://sapien.ucsd.edu/)) enriched with annotations of material and dynamic properties. 

The following image shows the fifteen fully interactive scenes: 

![placeholder.jpg](images/og_scene.png)

To download the dataset, you need to first configure where the dataset is to be stored. You can change it in `your_installation_path/omnigibson/global_config.yaml` (default and recommended: `og_dataset: your_installation_path/omnigibson/data/og_dataset`). OmniGibson scenes can be downloaded with one single line:

```bash
python -m omnigibson.utils.assets_utils --download_og_dataset
```

If the script fails to work, you can download from this [direct link](https://storage.googleapis.com/gibson_scenes/og_dataset.tar.gz) and extract to `your_installation_path/omnigibson/data/og_dataset`.

A description of the file structure and format of the files in the dataset can be found [here](https://github.com/StanfordVL/OmniGibson/tree/master/omnigibson/utils/data_utils). 

**Cubicasa / 3D Front Dataset Support:** We provide support for Cubicasa and 3D Front Dataset providing more than 10000 additional scenes (with less furniture than our fifteen scenes). To import them into OmniGibson, follow the instructions [here](https://github.com/StanfordVL/OmniGibson/tree/master/omnigibson/utils/data_utils/ext_scene). 

## Download Gibson and Stanford 2D-3D-Semantics scenes

What will you download?
- **Gibson static scenes**: more than 500 reconstructions of homes and offices with a Matterport device. These models keep the texture observed with the sensor, but contain some irregularities, specially with reflective surfaces and thin elements like chairs' legs.
- **Stanford 2D-3D-Semantics scenes**: 7 reconstructions of Stanford offices.

Files included in the dataset:

- All scenes, 572 scenes (108GB): gibson_v2_all.tar.gz
- 4+ partition, 106 scenes, with textures better packed (2.6GB): gibson_v2_4+.tar.gz
- Stanford 2D-3D-Semantics, 7 scenes (1.4GB): 2d3ds_for_omnigibson.zip

We have updated these datasets to be used with OmniGibson so that users can keep developing and studying pure navigation solutions. The following link will bring you to a license agreement and then to a downloading URL: [form](https://forms.gle/36TW9uVpjrE1Mkf9A)

After filling in the agreement, you will obtain a downloading `URL`. 
You can download the data manually and store it in the path set in `your_installation_path/omnigibson/global_config.yaml` (default and recommended: `g_dataset: your_installation_path/omnigibson/data/g_dataset`).
Alternatively, you can run a single command to download the dataset, decompress, and place it in the correct folder:
```bash
python -m omnigibson.utils.assets_utils --download_dataset URL
```

The Gibson Environment Dataset consists of 572 models and 1440 floors. We cover a diverse set of models including households, offices, hotels, venues, museums, hospitals, construction sites, etc. A diverse set of visualization of all spaces in Gibson can be seen [here](http://gibsonenv.stanford.edu/database/).
The following image shows some of the environments:
 
![spaces.png](images/spaces.png)

**Gibson Dataset Metadata:** Each space in the database has some metadata with the following attributes associated with it. The metadata is available in this [JSON file](https://raw.githubusercontent.com/StanfordVL/GibsonEnv/master/gibson/data/data.json). 
```
id                      # the name of the space, e.g. ""Albertville""
area                    # total metric area of the building, e.g. "266.125" sq. meters
floor                   # number of floors in the space, e.g. "4"
navigation_complexity   # navigation complexity metric, e.g. "3.737" (see the paper for definition)
room                    # number of rooms, e.g. "16"
ssa                     # Specific Surface Area (A measure of clutter), e.g. "1.297" (see the paper for definition)
split_full              # if the space is in train/val/test/none split of Full partition 
split_full+             # if the space is in train/val/test/none split of Full+ partition 
split_medium            # if the space is in train/val/test/none split of Medium partition 
split_tiny              # if the space is in train/val/test/none split of Tiny partition 
```
- Floor Number: Total number of floors in each model. We calculate floor numbers using distinctive camera locations. We use `sklearn.cluster.DBSCAN` to cluster these locations by height and set minimum cluster size to `5`. This means areas with at least `5` sweeps are treated as one single floor. This helps us capture small building spaces such as backyard, attics, basements.
- Area: Total floor area of each model. We calculate total floor area by summing up area of each floor. This is done by sampling point cloud locations based on floor height, and fitting a `scipy.spatial.ConvexHull` on sample locations.
- SSA: Specific surface area. The ratio of inner mesh surface and volume of convex hull of the mesh. This is a measure of clutter in the models: if the inner space is placed with large number of furnitures, objects, etc, the model will have high SSA. 
- Navigation Complexity: The highest complexity of navigating between arbitrary points within the model. We sample arbitrary point pairs inside the model, and calculate `A∗` navigation distance between them. `Navigation Complexity` is equal to `A*` distance divide by `straight line distance` between the two points. We compute the highest navigation complexity for every model. Note that all point pairs are sample within the *same floor*.
- Subjective Attributes: We examine each model manually, and note the subjective attributes of them. This includes their furnishing style, house shapes, whether they have long stairs, etc.

**Gibson Dataset Format:** Each space in the database has its own folder. All the modalities and metadata for each space are contained in that folder. 
```
mesh_z_up.obj             # 3d mesh of the environment, it is also associated with an mtl file and a texture file, omitted here
floors.txt                # floor height
floor_render_{}.png       # top down views of each floor
floor_{}.png              # top down views of obstacles for each floor
floor_trav_{}.png         # top down views of traversable areas for each floor  
```

For the maps, each pixel represents 0.01m, and the center of the image correspond to `(0,0)` in the mesh, as well as in the pybullet coordinate system. 


## Download Matterport3D Scenes
What will you download?
- Matterport3D Dataset: 90 scenes (3.2GB)

Please fill in this [form](http://dovahkiin.stanford.edu/matterport/public/MP_TOS.pdf) and send it to [matterport3d@googlegroups.com](mailto:matterport3d@googlegroups.com). Please put "use with OmniGibson simulator" in your email.

You'll then recieve a python script via email in response. Run `python download_mp.py --task_data omnigibson -o .` with the received script to download the data (3.2GB). Afterwards, move each of the scenes to the path set in `your_installation_path/omnigibson/global_config.yaml` (default and recommended: `g_dataset: your_installation_path/omnigibson/data/g_dataset`).

Reference: [Matterport3D webpage](https://niessner.github.io/Matterport/).