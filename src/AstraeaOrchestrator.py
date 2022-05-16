"""
------------------------
Issue the new sampling policy at every period
------------------------
"""
import os
from configparser import SafeConfigParser

class AstraeaOrc():
    def __init__(self):
        parser = SafeConfigParser()
        parser.read('../conf/astraea-config.ini')
        self.span_states_file = parser.get('application_plane', 'SpanStatesFile')
        self.span_states_file_txt = parser.get('application_plane', 'SpanStatesFileTxt')


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

    def issue_sampling_policy_txt(self, splits):
        # opening the file in read mode
        file = open(self.span_states_file_txt, "r")
        replacement = ""
        # using the for loop
        for line in file:
            line = line.strip()
            line_arr = line.split(" ")
            if line_arr[0] in splits:
                changes = line.replace(line_arr[1], splits[line_arr[0]])
            else:
                changes = line
            
        #     changes = line.replace("hardships", "situations")
            replacement = replacement + changes + "\n"


        file.close()
        # opening the file in write mode
        fout = open(self.span_states_file_txt, "w")
        fout.write(replacement)
        fout.close()
