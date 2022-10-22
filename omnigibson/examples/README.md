### Code Examples

The following examples illustrate the use of OmniGibson.

If you are interested in just getting started as an end-user, you only need check out `./environments`.

If you are looking for examples of BEHAVIOR, the benchmark of household activities that uses OmniGibson, please check the BEHAVIOR repository at https://github.com/StanfordVL/behavior.

- environments: how to instantiate OmniGibson environments with interactive or static scenes, optionally with a scene selector.
- learning: how to train RL policies for robot navigation using stable baselines 3, and how to save and replay demos of agents for imitation learning.
- objects: how to create, load, and place objects to predefined locations or using a logic sampler (e.g. onTop(A, B)), how to change texture as a function of the temperature, and how to generate the minimum volume bounding boxes of objects.
- object_states: how to change various objects states, including dusty, stained, (water sources) toggled on, (cleaning tool) soaked, sliced, and temprature, and how to save and reload object states.
- observations: how to generate different observation modalities such as RGB, depth, LiDAR, segmentation, etc.
- renderer: how to use the renderer directly, without the physics engine.
- robots: how to (keyboard) control robots with differential drive controllers, IK controllers and sampling-based motion planners.
- ros: how to run ROS with OmniGibson as if it is the real world.
- scenes: how to load interactive and non-interactive scenes, how to use domain randomization (of object models and/or texture), and how to create a tour video of the scenes.
- vr: how to use OmniGibson with VR.
- web_ui: how to start a web server that hosts OmniGibson environments.
