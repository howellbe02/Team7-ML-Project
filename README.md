## How to Obtain Results
The repository has multiple python and R files to run to obtain the results necessary for our project. KMeans has its own python file to calculate its results, and the skill estimates, Markov chains, and evaluation process have their own files, too. We will detail the proper order to run them in.

### KMeans
Ran ExcelDataModify.py in order to tabulate career averages for offensive and defensive stats for head shots, body shots, leg shots, takedowns, and ground shots, which output the averages into FighterStats.xlsx saved in the data folder. This file is then used by MMAKMeans.py to fit 3 clusters with these stats.

### Baseline
Run BaselineReg.py to receive the results in terms of accuracy for the Baseline logistic regression we created.

### Skill Estimates
All Code for the skill estimates is in the folder Skill Estimates Code, and all output is in the Skill Estimates folder. The Skill estimates folder saves skill estimates by the cutoff date that the train/test split is set on. Skill estimates are also divided by weight class, of which there are 9. In each of these weight class folders, there are excel files for the attack and defense strength of each skill estimate (aside from ground and head shot probability, which is not modeled using attack and defense strengths because this is more of a stylistic characteristic pertaining to an individual fighter). There is also a file that contains the intercept of each skill estimate. There are 14 skill estimate R files for this that all basically have the same code, with slight modifications depending on what type of bayesian generalized linear model they used (binomial, poisson, or gamma) and what match stats were used to estimate them. There is also the file SEFunctions.R which contains helper functions,and Run_All_Skill_Estimate_Code.R which is able to run all of the code from each of the files if desired.

### Markov Chains
Once the skill estimates are obtained, run Main.py to produce predictions for fights starting May 06, 2023 and ending November 04, 2023. Main.py uses methods from AccuracyProbs.py to produce the transition probabilities necessary for the Markov chains, and the chains are then run using methods from MarkovModel.py. This will create output excel files inside ./predictions/ for each weight class and another excel file to include all the fights from multiple weightclasses. A fight is represented by a single row in these excel tables, and the attributes for a fight are its weightclass, fighter i probability of winning, and fighter j probability of winning.

### Merge Fight Outcome Data with Predictions
Once the former step is completed, run PredictionsOutcomesMerge.py to create an excel file in ./predictions/, which will include for each fight (represented by a row in the table) the actual outcome, as well as the date of the fight.

### Judging Simulation
JudgingSimulation.py uses a logistic regression to judge a fight outcome as either a win or a loss. It fits the past winner of a previous fights as the dependent variable, and for the independent variables, the difference between their totals and their opponent's totals for significant strikes they landed, amount of takedowns landed, submissions attempted, and control time.

### Evaluation
Now, we can determine the results of the Markov Chains. Run Evaluation.py, and it will produce an Accuracy statistic for the Markov chains. If fighter i has a probability of winning greater than 50%, it predicts that he will win. If this matches up with the actual outcome, it is considered a successful prediction.
