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
print(all_spans_list)


print("***** Welcome to Astraea evaluator!")

  
process = None
def exit_handler():
    print('My application is ending!')
    if process:
        process.terminate()
        print("Killed")

# Defining main function
def main():
    atexit.register(exit_handler)

    print("---- Astraea evaluation started! Parameters are a) period: ", period, " b) total: ", totalExpDuration)

    ## first make sure sampling policy is revert to default (100 per span)
    cmd_cp = "cp {} {}".format(samplingPolicy, samplingPolicyDefault)
    print("sampling policy is revert to default: ", cmd_cp)
    os.system(cmd_cp)
    print("Check it out ", os.system("head -n 3 {}".format(samplingPolicy)))

    ## start sending requests
    cmd_wrk = "{}/wrk -D exp -t 4 -c 4 -d {} -L -s {}/scripts/social-network/compose-post.lua http://localhost:8080/wrk2-api/post/compose -R {}".format(workloadPath, totalExpDuration, workloadPath, qps)
    print("Sending req in background: ", cmd_wrk)
    process = subprocess.Popen(cmd_wrk, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ### Inject problem
    problem_now = random.choice(all_spans_list)
    cmd_inject = "echo {} > {}".format(problem_now,sleepPath)
    print("Injecting problem: ", cmd_inject)
    os.system(cmd_inject)
    print("Check content now: ", os.system("head -n 3 {}".format(sleepPath)))

    ## sleep for a bit
    time.sleep(period + 10)

    print("Woke up and running Astraea Controller")

    ## run Astraea and collect stats with problem_now
    astraeaCont = ace.AstraeaControllerEval()
    astraeaCont.run_with_evaluator(problem_now, totalExpDuration)



# __name__
if __name__=="__main__":
    main()