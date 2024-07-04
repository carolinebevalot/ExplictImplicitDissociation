# ExplictImplicitDissociation
This script calls every script necessary to analyse the differences between contexts.
PATHS['ScriptsPathRoot'] should correspond to the GitHub folder of the Bayesian Battery.
If not, it can be changed in the first lines of this script.

Data files should be in a folder called 'BayesianBattery_complements'
The ones required for the analyses are the following :
* DataFrame with answers from all subjects
  generated with GatherResults.py:
    exp1_results : PATHS['exp1_results']
    exp3_results : PATHS['exp3_results']
 and dataframe with prior reports in T1
    report_df : PATHS['exp1_priorreport']

* Models parameters fitted to subjects' behavior with the Nelder-Mead algorithm
  generated with ModelBasedFitting.py:
    'fitted_parameters_from_complete_bayesian_model':
        'exp1': PATHS['SubjectsFitResults']+'/Gorilla_V4/exp1_parameters_fits_fit_bayesian_param_nov22_best_fit.csv'
        'exp3':PATHS['SubjectsFitResults']+'/Gorilla_V4/exp3_parameters_fits_fit_bayesian_param_dec22_best_fit.csv'

* Logistic weights of priors, sensory likelihood and posteriors in subjects' answers
  computed with ComputeWeightsInLogisticRegression.py:
    Files in folder PATHS['LogregWeightsFolder'] (depending on the model, the type of priors and the evidence level)

* We can check whether fits are equivalent with the Nelder Mead or the Logistic Regression
  with CorrelateNMLogRegWeights.py

* List of subjects included in analysis
  generated with ComputeSubjectsLists_ForContextPaper.py:
    PATHS['SubjectsToIncludeFile']

* Logistic weights of explicited and implicit priors on trials before of after reports
  computed with ComputeWeightsInLogisticRegressionAroundExplicitReports.py :
   PATHS['LogregWeightsFolder']/exp3_implicit_explicit_logreg_after_Q_pen.csv

* Results of the cross validation of the different models
  computed with ModelCrossValidation.py
      'exp1': PATHS['ModelCrossValidation_exp1']
      'exp3': PATHS['ModelCrossValidation_exp3']

* Results of the model identifiability:
    Bayesian models: PATHS['ModelIdentifiability_BFA_BO_C']
    Bayesian vs. Heuristic models: PATHS['ModelIdentifiability_BFD_H']

* Results of parameter recovery of the Bayesian models
    PATHS['ParameterRecoveryResults']

* Recovery
These first analysis steps can be processed again if run_logweights_computations
(to verify logistic weights) and run_model_comparison (to verify model cross validation) are set to True.
Be careful before changing these parameters, these steps are time consumming (hours and days respectively).
