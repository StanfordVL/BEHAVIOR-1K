# :octicons-rocket-16: **Speed Optimization**

This is an Ad Hoc page for tips & tricks about speed optimization. The page is currently under construction. 

A lot of factors could affect the speed of OmniGibson. Here are a few of them:


## Macros

`macros.py` contains some macros that affects the overall configuration of OmniGibson. Some of them will have an effect on the performance of OmniGibson:

1. `ENABLE_HQ_RENDERING`: While it is set to False by default, setting it to True will give us better rendering quality as well as other more advanced rendering features (e.g. isosurface for fluids), but with the cost of dragging down performance.

2. `USE_GPU_DYNAMICS`: setting it to True is required for more advanced features like particles and fluids, but it will lower the performance of OmniGibson.

3. `RENDER_VIEWER_CAMERA`: viewer camera refers to the camera that shows the default viewport in OmniGibson, if you don't want to view the entire scene (e.g. during training), you can set this to False andit will boost OmniGibson performance.

4. `ENABLE_FLATCACHE`: setting it to True will boost OmniGibson performance.


## Miscellaneous

1. Assisted and sticky grasping is slower than physical grasp because they need to perform ray casting.

2. Setting a high `physics_frequency` vs. `action_frequency` will drag down OmniGibson's performance.