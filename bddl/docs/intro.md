
# BDDL: BEHAVIOR Domain Definition Language 

The BEHAVIOR Domain Definition Language (BDDL) is the domain-specific language of the BEHAVIOR benchmark for embodied AI agents in simulation. BEHAVIOR's 100 activities are realistic, diverse, and complex, and BDDL facilitates data-driven definition of these activities. BDDL is object-centric and based in predicate logic, and can express an activity's initial and goal conditions symbolically. The codebase includes 100 such symbolic definitions and functionality for parsing them and any other BDDL file, including a custom one; compiliing the symbolic definition to be grounded in a physically simulated environment; checking success and progress efficiently at every simulator step; and solving the goal condition to measure finer-grained progress for evaluation. 


### Citation
If you use BEHAVIOR, consider citing the following publication:

```
@inproceedings{srivastava2021behavior,
      title={BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, Ecological Environments}, 
      author={Sanjana Srivastava* and Chengshu Li* and Michael Lingelbach* and Roberto Martín-Martín* and Fei Xia and Kent Vainio and Zheng Lian and Cem Gokmen and Shyamal Buch and C. Karen Liu and Silvio Savarese and Hyowon Gweon and Jiajun Wu and Li Fei-Fei},
      year={2021},
      booktitle={5th Annual Conference on Robot Learning}
}
```

### Code Release
The GitHub repository of BDDL can be found here: [BDDL GitHubRepo](https://github.com/StanfordVL/bddl). Bug reports, suggestions for improvement, as well as community developments are encouraged and appreciated. 

### Documentation
The documentation for iGibson can be found here: [BDDL Documentation](https://stanfordvl.github.io/bddl). It includes installation guide, quickstart guide, code examples, and APIs.

If you want to know more about BDDL, you can also check out the [BEHAVIOR paper](https://arxiv.org/abs/2108.03332) and [our webpage](https://behavior.stanford.edu/), where you can create your own BEHAVIOR activity using a visual version of BDDL!
