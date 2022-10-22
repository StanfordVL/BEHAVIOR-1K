# Viewer

### Overview

We developed an easy-to-use OmniGibson-human interface called **Viewer** for users to inspect and interact with our scenes and objects. The Viewer will automatically pop up if you use `gui_non_interactive` or `gui_interactive` mode in `Simulator`.

To play with an example of viewer, you can run the following command:

```bash
python -m omnigibson.examples.environments.env_int_example
```
or 
```bash
python -m omnigibson.examples.environments.env_nonint_example
```
after downloading the Rs_int or Rs scenes (interactive or non-interactive, see [the installation instructions](installation.md))
![viewer.png](images/viewer.png)

In example image above, you can see on the top left corner `px 0.4 py -0.9 pz 1.2`, which indicates the camera position, `[1.0, 0.1, -0.1]`, which indicates the camera viewing direction, and `manip mode`, which indicates the current control mode you are in. There are three modes that we explain below.

Keyboard control includes the following
- `W`, `A`, `S`, `D`, `T`, `G`: translate forward/left/backward/right/up/down
- `Q`, `E`: rotate left/right
- `M`: choose between different control mode (navigation, manipulation and planning)
- `R`: start/stop recording
- `P`: pause/resume recording
- `H`: show help menu
- `ESC`: exit

We have three control modes (navigation, manipulation and planning) and the mouse control is different for each control mode. You may switch between these control modes by pressing `M`.

Mouse control in navigation mode
- Left click and drag: rotate camera
- CTRL + left click and drag: translate camera
- Middle click and drag: translate camera closer/further away in the viewing direction

Mouse control in manipulation mode
- Left click and drag: create ball-joint connection to the clicked object and move it
- Middle click and drag: create fixed-joint connection to the clicked object and move it
- CTRL + click and drag: move the object further/closer

Mouse control in planning mode
- Left click and drag: create (click), visualize (drag) and plan / execute (release) a base motion subgoal for the robot base to reach the physical point that corresponds to the clicked pixel
- Middle click: create create, and plan / execute an arm motion subgoal for the robot end-effector to reach the physical point that corresponds to the clicked pixel

In manipulation and planning modes, a visual indicator will be visualized in the `Viewer` to assist control (e.g. the blue sphere at the bottom in the image above).

The code can be found in [omnigibson/render/viewer.py](https://github.com/StanfordVL/OmniGibson/tree/master/omnigibson/render/viewer.py).
