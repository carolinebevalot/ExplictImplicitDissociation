#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:22:56 2023
This script computes the lists of subjects to include and exclude based on the different exclusion criteria :
    X is a threshold used in erratic answers
    - more than X% of erratic answers (answers against the prior and the likelihood whereas they were obvious and congruent)
    - the same proportion on very obvious trials
    - more than 80%  of missing answers 
It also computes the quartile with the lowest and highest SPQ scores

@author: caroline
"""
import argparse
import os.path as op
import datetime
import os
import pandas as pd
import sys

computer = os.uname().nodename
parser = argparse.ArgumentParser(description="Creates the lists of subjects to include")
parser.add_argument('-computer', type=str, default=computer,
                    help='Define on which computer the script is run to adapt paths')
parser.add_argument('-BayesianBattery_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery/',
                    help='Define the path for scripts directory')
parser.add_argument('-BayesianBattery_completements_path', type=str,
                    default='/home/caroline/Desktop/BayesianBattery_complements/',
                    help='Define the path for the data and results')
args = parser.parse_args()
sys.path.append(args.BayesianBattery_path)
sys.path.append(args.BayesianBattery_completements_path)
os.chdir(args.BayesianBattery_path)
print(os.getcwd())

from Analysis.AnalysisOfGorrillaData.VariablesForPsychometricPaperAnalysis\
    import PROP_ERRATIC_ANSWERS_THRESHOLD, PROP_ANSWERS_THRESHOLD, SD_THRESHOLD,\
        SCALE_THRESHOLDS

from PathsForContextPaper import define_paths_for_context_paper
PATHS = define_paths_for_context_paper(computer)

date = datetime.date.today().isoformat().replace('-', '')

subjects_summary = pd.read_csv(PATHS['SubjectsSummaryFile'], index_col=0)

with_logreg = [subject for subject in subjects_summary.index if subject != 'm0bglu5q']
pd.DataFrame(with_logreg,
             columns=['participant_ID']).to_csv(op.join(PATHS['SubjectsList'],
                                                        f'subjects_with_logregweights_n{len(with_logreg)}.csv'))

subjects_to_include_erratic_answers \
    = subjects_summary[
        ~((subjects_summary['prop_erratic_answers_with_gen_priors_exp3']
           > subjects_summary['prop_erratic_answers_with_gen_priors_exp3'].std()
           * SD_THRESHOLD)
          | (subjects_summary['prop_erratic_answers_with_gen_priors_exp1']
             > subjects_summary['prop_erratic_answers_with_gen_priors_exp1'].std()
             * SD_THRESHOLD))].index
pd.DataFrame(subjects_to_include_erratic_answers,
             columns=['participant_ID']).to_csv(
                 op.join(PATHS['SubjectsList'],
                         "subjects_included_on_prop_erratic_answers_with_gen_priors" +
                         f"_inf{str(SD_THRESHOLD).replace('.','')}sd_bothexp_" +
                         f"n{len(subjects_to_include_erratic_answers)}.csv"))


