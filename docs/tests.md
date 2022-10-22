# Tests and Examples
### Tests
We provide tests in [this directory](https://github.com/StanfordVL/OmniGibson/tree/master/tests) in our repository. 
You can run them executing:
```bash
pytest --ignore disabled --ignore benchmark
```
It will take a few minutes. If all tests pass, you will see something like this
```bash
=============================== test session starts ================================
platform linux -- Python 3.5.6, pytest-4.6.9, py-1.5.3, pluggy-0.13.1
plugins: openfiles-0.3.0, doctestplus-0.1.3, arraydiff-0.2
collected 27 items

test_binding.py ..                                                           [  7% ]
test_navigate_env.py ..                                                      [ 14% ]
test_object.py ....                                                          [ 29% ]
test_render.py ...                                                           [ 40% ]
test_robot.py ..........                                                     [ 77% ]
test_scene_importing.py ....                                                 [ 92% ]
test_simulator.py .                                                          [ 96% ]
test_viewer.py
```
