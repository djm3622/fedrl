# Federated Distributional Reinforcement Learning

Sweden project.

## TODO
1. After bug fixing the method is not better.
2. Make improvements to benefit federation. Local dominates too much.
3. Create data scarcity/imbalance
4. Personalized teacher via LOO barycenters
5. Trust-region KL with dual control
6. 

## Case Study 1.1
This case study looks at a federated DRL bandit problem.

### Requirements
Necessary python packages can be installed using **pip**:
```
pip install -r setup/requirements_case_study_1_1.txt
```

### Experiment (Training/Evaluation)
To execute the trainer and gather results, make the appropiate changes to the configuration file and then use:
```
python case_study_1_1.py configs/case_study_1_1.yaml
```
