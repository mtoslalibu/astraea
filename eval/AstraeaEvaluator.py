"""
Astraea evaluator program. 
This program runs demo experiments. 
1) Run continous workload (hey or work), 2) inject problems, 3) runs Astraea, 4) collects evaluation data.
Evaluation data:
a) Traffic split of spans, b) traces sizes after each period
"""

import sys
sys.path.append('../src/')

import AstraeaControllerEval as ace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from matplotlib.pyplot import figure
import glob
import random
import time
from IPython.display import display
from configparser import SafeConfigParser
import os
import subprocess
import atexit
import signal
import logging


pd.set_option('display.max_colwidth', None)
pd.set_option("precision", 1)
pd.options.display.float_format = '{:.1f}'.format
sns.set(style='whitegrid', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.2)                                                  
color = sns.color_palette("Set2", 2)


parser = SafeConfigParser()
parser.read('../conf/eval-config.ini')

period = int(parser.get('experimentation_plane', 'Period'))
totalExpDuration = int(parser.get('experimentation_plane', 'ExperimentDuration'))

samplingPolicy = parser.get('experimentation_plane', 'SpanStatesFileTxt')
samplingPolicyDefault = parser.get('experimentation_plane', 'SpanStatesOrigFileTxt')
sleepPath = parser.get('experimentation_plane', 'SleepInjectionPath')


qps = int(parser.get('experimentation_plane', 'QPS'))
workers = int(parser.get('experimentation_plane', 'Workers'))


resultDir = parser.get('experimentation_plane', 'ResultDir')
elimPercentile = int(parser.get('experimentation_plane', 'EliminationPercentile'))

app = parser.get('experimentation_plane', 'Application')

## app specific parameter
if app == "SocialNetwork":
    all_spans = open("../data/{}".format(parser.get('experimentation_plane', 'AllSpansSN')), "r")
    workloadPath = parser.get('experimentation_plane', 'WorkloadGeneratorPathSN')
elif app == "Media":
    all_spans = open("../data/{}".format(parser.get('experimentation_plane', 'AllSpansMedia')), "r")
    workloadPath = parser.get('experimentation_plane', 'WorkloadGeneratorPathMedia')
else:
    all_spans = open("../data/{}".format(parser.get('experimentation_plane', 'AllSpansTT')), "r")
    workloadPath = parser.get('experimentation_plane', 'WorkloadGeneratorPathTT')


all_spans_list = all_spans.read().split("\n")
logging.debug(all_spans_list)


logging.info("***** Welcome to Astraea evaluator!")

  
process = None
def exit_handler():
    logging.info('My application is ending!')
    os.killpg(0, signal.SIGKILL)
 
    logging.warning("Killed")

# Defining main function
def main():
    atexit.register(exit_handler)

    logging.info("---- Astraea evaluation started! Parameters are a) period: ", period, " b) total: ", totalExpDuration)

    ## first make sure sampling policy is revert to default (100 per span)
    cmd_cp = "cp {} {}".format(samplingPolicyDefault, samplingPolicy)
    logging.info("sampling policy is revert to default: ", cmd_cp)
    os.system(cmd_cp)
    logging.info("Check it out ", os.system("head -n 3 {}".format(samplingPolicy)))

    ## start sending requests
    cmd_wrk = "{}/wrk -D exp -t {} -c {} -d {} -L -s {}/scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R {}".format(workloadPath, workers,workers, totalExpDuration+100, workloadPath, qps)
    logging.info("Sending req in background: ", cmd_wrk)
    process = subprocess.Popen(cmd_wrk, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ### Inject problem
    problem_now = random.choice(all_spans_list)
    cmd_inject = "echo {} > {}".format(problem_now,sleepPath)
    logging.info("Injecting problem: ", cmd_inject)
    os.system(cmd_inject)
    logging.info("Check content now: ", os.system("head -n 3 {}".format(sleepPath)))

    ## sleep for a bit
    time.sleep(period + 15)

    logging.info("Woke up and running Astraea Controller")

    ## run Astraea and collect stats with problem_now
    astraeaCont = ace.AstraeaControllerEval()
    astraeaCont.run_with_evaluator(problem_now, totalExpDuration,resultDir,elimPercentile)



# __name__
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    os.setpgrp() # create new process group, become its leader
    main()
    # try:
    #     main()
    # finally:
    #     os.killpg(0, signal.SIGKILL) # kill all processes in my group