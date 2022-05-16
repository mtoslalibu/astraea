"""
------------------------
Issue the new sampling policy at every period
------------------------
"""
import os
from ConfigParser import SafeConfigParser

class AstraeaOrc():
    def __init__(self):
        parser = SafeConfigParser()
        parser.read('../conf/astraea-config.ini')
        self.span_states_file = parser.get('application_plane', 'SpanStatesFile')




def issue_sampling_policy(self, splits):
    ### Read file and if it includes the span from before, replace it or insert new
    # Opening JSON file
    if os.path.exists(self.span_states_file):
        f = open(self.span_states_file)
        # returns JSON object as 
        # a dictionary
        data = json.load(f)
        f.close()
    else:
        data = {}

    ## issue all the new sampling values to shared policy file
    for item in splits:
        data[item] = splits[item]

    f_write = open(self.span_states_file, "w+")
    json.dump(data, f_write)
    f_write.close()
