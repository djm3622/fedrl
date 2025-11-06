# Federated Distributional Reinforcement Learning

Sweden project.

## TODO
1. Implement the federated local training. It should train instances in parrellel on a single gpu.
2. Figure out the weird glitch the agents are learning to avoid the dangerous states. Maybe remove it but it is interesting they are able to do it and it encourages more dangerous behaviour so possibly keep it.

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

## Case Study 2.1
Necessary python packages can be installed using **pip**:
```
pip install -r setup/requirements_case_study_2_1.txt
```

### Experiment (Training/Evaluation)
To execute the trainer and gather results, make the appropiate changes to the configuration file and then use:
```
python case_study_2_1.py configs/case_study_2_1.yaml
```
