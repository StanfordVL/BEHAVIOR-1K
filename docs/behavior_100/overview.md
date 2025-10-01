---
icon: material/graph-outline
---

!!! warning "Deprecated"
    This information is about BEHAVIOR-100, a legacy version of BEHAVIOR built on iGibson and kept for posterity reasons. We recommend all new users to use BEHAVIOR-1K, which features 1,000 human-grounded household activities (vs. 100), 50 diverse scenes, thousands of annotated objects, and runs on the more powerful OmniGibson simulator with richer physics and visual realism.

<iframe 
  src="https://player.vimeo.com/video/1101627941?badge=0&autopause=0&autoplay=1&muted=1&loop=1&title=0&byline=0&portrait=0&controls=0" 
  width="640" 
  height="320" 
  frameborder="0" 
  allow="autoplay; fullscreen" 
  allowfullscreen>
</iframe>


### What is BEHAVIOR-100?

BEHAVIOR-100 is the first generation of BEHAVIOR, a benchmark for embodied AI with 100 activities in simulation, spanning a range of everyday household chores such as cleaning, maintenance, and food preparation. These activities are designed to be realistic, diverse and complex, aiming to reproduce the challenges that agents must face in the real world. 

##### Reference
- [BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments](https://arxiv.org/abs/2108.03332). Sanjana Srivastava\*, Chengshu Li\*, Michael Lingelbach\*, Roberto Martín-Martín\*, Fei Xia, Kent Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, C. Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, Li Fei-Fei, Conference on Robot Learning (CoRL) 2021.

### Building blocks of BEHAVIOR-100
Building BEHAVIOR-100 poses three fundamental difficulties for each activity: definition, instantiation in a simulator, and evaluation. BEHAVIOR addresses these with three building blocks. First, we propose a predicate logic-based description language (BDDL) for expressing an activity’s initial and goal conditions, enabling generation of diverse instances for any activity. Second, we identify the simulator-agnostic features required by an underlying environment to support BEHAVIOR-100, and demonstrate in one such simulator, i.e., iGibson 2.0. Third, we introduce a set of metrics to measure task progress and efficiency, absolute and relative to human demonstrators. We include 500 human demonstrations in virtual reality (VR) to serve as the human ground truth. 

Do you want to benchmark your solution? Follow the instructions [here](https://stanfordvl.github.io/behavior/installation.html) to get started. The main components are:

##### BEHAVIOR-100 benchmark [documentation](https://stanfordvl.github.io/behavior/intro.html).
##### iGibson simulator [codebase](https://github.com/StanfordVL/iGibson) and [documentation](http://svl.stanford.edu/igibson/docs/).
##### Combined BEHAVIOR-100 iGibson2.0 [scene and object assets](https://storage.googleapis.com/gibson_scenes/behavior_data_bundle.zip).
##### BDDL specification language [codebase and documentation](https://github.com/StanfordVL/bddl).
##### BEHAVIOR-100 VR human demonstration [dataset](./dataset.md).

<!-- You will download and install the required infrastructure: [a new version of iGibson](http://svl.stanford.edu/igibson/docs/installation.html), our simulation environment for interactive tasks extended now to new object states for BEHAVIOR, the BEHAVIOR Dataset of Objects and the iGibson2.0 Dataset of Scenes (combined in our [benchmarking bundle](https://storage.googleapis.com/gibson_scenes/behavior_data_bundle.zip)), with object and house models to use the benchmark, and our [starter code](https://github.com/StanfordVL/behavior/), with examplest to train againts in the tasks.  -->

## What makes BEHAVIOR-100 different?

!!! info "🏠 100 Household Activities in Realistically Simulated Homes"
    
    Activities include cleaning, preparing food, tidying, polishing, installing elements, etc. The activities are obtained from the American Time Use Survey and approximate the real distribution of tasks performed by humans in their everyday lives.
    
    <iframe width="100%" height="281" style="max-width: 500px; border-radius: 8px; margin: 1rem 0;"
            src="https://player.vimeo.com/video/1101630779?autoplay=1&loop=1&autopause=0&muted=1&controls=0&title=0&byline=0&portrait=0&badge=0"
            frameborder="0" allowfullscreen>
    </iframe>
    
    **Resources:** [Activity images and videos](./demo_gallery.md)

!!! warning "🤖 Decision Making based on Onboard Sensing for Navigation and Manipulation"
    
    These long-horizon activities require that the robot understand the scene, plan a strategy, and execute it by controlling the motion of the embodied agent based on observations. We provide [three embodiments](https://stanfordvl.github.io/behavior/agents.html), both [a rich actuation space and a set of action primitives](https://stanfordvl.github.io/behavior/agents.html#embodiments-actuation-sensing-grasping), and [realistic RGB-D and proprioceptive signals](https://stanfordvl.github.io/behavior/agents.html#observations): as close as it gets to the challenges of real-world.
    
    <iframe width="100%" height="281" style="max-width: 500px; border-radius: 8px; margin: 1rem 0;"
            src="https://player.vimeo.com/video/1101630792?autoplay=1&loop=1&autopause=0&muted=1&controls=0&title=0&byline=0&portrait=0&badge=0"
            frameborder="0" allowfullscreen>
    </iframe>
    
    **Resources:** [Benchmark documentation](https://stanfordvl.github.io/behavior/intro.html)

!!! danger "⚙️ More Complex Interactions than just Pick-and-Place"
    
    Accomplishing the BEHAVIOR activities requires changing more than the position of the objects in the environment: they need to be cooked, frozen, soaked, cleaned, and more. All these new types of state changes are supported by the provided simulator, iGibson 2.0, and enable new and unique types of activities.
    
    <iframe width="100%" height="281" style="max-width: 500px; border-radius: 8px; margin: 1rem 0;"
            src="https://player.vimeo.com/video/1101630897?autoplay=1&loop=1&autopause=0&muted=1&controls=0&title=0&byline=0&portrait=0&badge=0"
            frameborder="0" allowfullscreen>
    </iframe>
    
    **Resources:** [More about the simulator iGibson 2.0](http://svl.stanford.edu/igibson/)




