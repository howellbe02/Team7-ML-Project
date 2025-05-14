## How to Obtain Results
The repository has multiple python and R files to run to obtain the results necessary for our project. KMeans has its own python file to calculate its results, and the skill estimates, Markov chains, and evaluation process have their own files, too. We will detail the proper order to run them in.

### KMeans

### Baseline
Run BaselineReg.py to receive the results in terms of accuracy for the Baseline logistic regression we created.

### Skill Estimates

### Markov Chains
Once the skill estimates are obtained, run Main.py to produce predictions for fights starting May 06, 2023 and ending November 04, 2023. Main.py uses methods from AccuracyProbs.py to produce the transition probabilities necessary for the Markov chains, and the chains are then run using methods from MarkovModel.py. This will create output excel files inside ./predictions/ for each weight class and another excel file to include all the fights from multiple weightclasses. A fight is represented by a single row in these excel tables, and the attributes for a fight are its weightclass, fighter i probability of winning, and fighter j probability of winning.

### Merge Fight Outcome Data with Predictions
Once the former step is completed, run PredictionsOutcomesMerge.py to create an excel file in ./predictions/, which will include for each fight (represented by a row in the table) the actual outcome, as well as the date of the fight.

### Evaluation
Now, we can determine the results of the Markov Chains. Run Evaluation.py, and it will produce an Accuracy statistic for the Markov chains. If fighter i has a probability of winning greater than 50%, it predicts that he will win. If this matches up with the actual outcome, it is considered a successful prediction.
