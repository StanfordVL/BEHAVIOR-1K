# :material-file-question: **Frequently Asked Questions**

## **What is the relationship between BEHAVIOR-1K and OmniGibson?**

[BEHAVIOR-1K](https://behavior.stanford.edu/behavior-1k) is the high-level project that proposes an embodied AI benchmark of 1,000 tasks. To measure agents' performance accurately and reproducibly on these tasks, a simulation platform with comprehensive features and capabilities is necessary. This necessitates the need for a simulation platform that can support all the semantics required in the BEHAVIOR-1K tasks. OmniGibson meets this need as a feature-complete simulation platform, allowing us to instantiate and evaluate these tasks fully in a simulated environment.

## **How is OmniGibson connected to Nvidia's Omniverse?**

OmniGibson is built upon NVIDIA's Isaac Sim/Omniverse physics backend, leveraging its robust physics simulation capabilities. On top of this powerful engine, OmniGibson provides modular and user-friendly APIs, along with additional features such as controllers, robots, object states, and more. These added functionalities enable OmniGibson to facilitate the necessary physical interactions and simulations required by the diverse range of tasks included in the BEHAVIOR-1K task suite.

## **Why should I use OmniGibson?**
The core strengths of OmniGibson lie in its exceptional physical and visual realism, two critical factors in the development of embodied AI agents. 

- `Physical Realism`: to our knowledge, OmniGibson is the only simulation platform that supports large-scale scene interaction with 
    - cloth
    - fluids
    - semantic object states (e.g. temperature, particle interactions)
    - complex physical interactions (transition rules).
- `Visual Realism`: OmniGibson is built on NVIDIA's Isaac Sim/Omniverse physics backend, which provides industry-leading real-time ray tracing capabilities, resulting in highly realistic visual simulations.

While OmniGibson may not currently be the fastest simulation platform available, we are actively working to enhance its performance. Our efforts include optimizing speeds and leveraging NVIDIA's cloning features to enable large-scale parallelization. 

## **What is the relationship between Gibson, iGibson, and OmniGibson?**

[Gibson](http://gibsonenv.stanford.edu/) is a collection of high-fidelity, large-scale scene scans, primarily designed for navigation tasks within static environments. 

[iGibson](https://svl.stanford.edu/igibson/), building upon this foundation, introduced interactivity by creating 15 fully interactive scenes. However, iGibson's implementation in PyBullet limited its capabilities, lacking support for high-fidelity rendering, cloth simulation, and fluid dynamics.

[OmniGibson](https://behavior.stanford.edu/omnigibson/) aims to provide a comprehensive and feature-complete simulation platform. It includes 50 much larger-scale, fully interactive scenes with thousands of curated objects. Leveraging real-time ray tracing and NVIDIA's Omniverse backend, OmniGibson offers exceptional visual realism while supporting advanced simulations involving cloth, fluids, and various other complex physical interactions.