# Federated Distributional Reinforcement Learning

Sweden project.

## TODO
1. Abalation study to figure out why our method works better in 1.1, when doesn't it work?

## Case Study 1.1
This case study looks at a federated DRL bandit problem. *Our method converges faster.* 

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
