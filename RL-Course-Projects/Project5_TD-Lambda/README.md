# FranceLab 5 - Kordel K. France

This project was constructed for the Introduction to Machine Learning course, class 605.649 section 84 at Johns Hopkins
University. FranceLab5 is a machine learning toolkit that implements several reinforcement learning & temporal difference
learning algorithms for the racetrack controls problem. 
regression tasks. Specifically, the toolkit shows the efficacy of the Value Iteration, Q-Learning, and SARSA algorithms
in successfully teaching an autonomous control agent to navigate. FranceLab5 is a software module written in Python 3.7 
that facilitates such algorithms.

##Notes for Graders
All files of concern for this project (with the exception of `main.py`) may be found in the `RL`,
folders. I kept the `Neural Network` folder along with some of the folders associated with cross validation
within the project because my intent was to add a test condition in which a neural network evaluated the value tables 
for each of the algorithms, but I was unable to produce replicable results.

I found it most helpful to keep all code for one algorithm in a single file. Many of the helper functions among the three
algorithms are direct copies of each other, so there is room to optimize lines of code. However, when tuning training 
values, it was useful to have a standalone helper function I could customize for each specific algorithm, so there are 
subtle differences.

I have created blocks of code for you to test and run each algorithm if you choose to do so. In `__main__.py` scroll
to the bottom and find the `main` function. Simply comment or uncomment blocks of code to test if desired. Additionally,
configuration files control the hyperparemeters for each algorithm - files exist under the `RL` folder titled
`demo_config_{algorithm}.py` with a list of hyperparameters that can be altered for each reinforcement learning agent. 
The default values in each "config" file are the tuned values that produce optimal results.

Data produced in my paper were run with at least 10 trials. These are denoted as `LAP_COUNT` within the configuration
files.
  
`__main__.py` is the driver behind importing the track files, initializing training parameters, and performing the 
simulation. `monte_carlo.py` is the file where a Monte Carlo style simulation is performed in order to compare the 
performance of all reinforcement learning algorithms.


## Running FranceLab5
1. **Ensure Python 3.7 is installed on your computer.**
2. **Navigate to the Lab5 directory.** For example, `cd User\Documents\PythonProjects\FranceLab5`.
Do NOT `cd` into the `Lab5` module.
3. **Run the program as a module: `python3 -m Lab5`.**
4. **Input and output files ar located in the `io_files` subdirectory.** 


### FranceLab5 Usage

```commandline
usage: python3 -m Lab5
```


