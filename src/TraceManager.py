## READ traces

import requests
import json
import time;
import requests 
import concurrent.futures
from IPython.display import display, HTML
from collections import defaultdict
from scipy import stats
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score
import random
from scipy.stats import norm, kurtosis, pearsonr
from kneed import KneeLocator, DataGenerator as dg
import random

class Node(object):
    """
    Class of node objects in a doubly linked list to
    represent DOT edges in a treelike form. source
    nodes are parents and destination nodes are children.
    info from edge labels resides in destination nodes/
    children.
    """
    def __init__(self, name='root',  parents="", iden = "", latency = 0, start=0, end=0):
        self.name = name
        self.parents = parents
        self.id = iden
        self.latency = latency
        self.start = start
        self.end = end
        
def get_traces_jaeger_file(file_name):
    """
    Method for reading traces from provided file. 
    Used when replaying traces from files. 
    """
    print("------")
    data = {}
    with open(file_name) as f:
        data = json.load(f)

    # Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
    print("# of traces: ",len(data["data"]))
    return data

## get current time in jaeger format
current_milli_time = lambda: int(round(time.time() * 1000000))

## get traces from API, given service and lookback period
def get_traces_jaeger_api(service = "compose-post-service", period=5):
    """
    Method for fetching traces from provided API.  
    """
    print("------ JAeger api called")

    JaegerAPIEndpoint = "http://localhost:16686/jaeger/api/traces?end={end}&maxDuration&minDuration&service={service}&start={start}&prettyPrint=true"

    
    
    end=current_milli_time()
    ## adjust time -- delayed batch by 10 sec
    delayed_sec = 10
    end = end - delayed_sec * 1000000

    ## period for lookback
    start = end - period * 1000000

    formatted_endpoint = JaegerAPIEndpoint.format(end, service, start)
    print("formatted endpiont now: ", formatted_endpoint)

    response_batch = requests.get(url = formatted_endpoint)

    data = response_batch.json()
    print("# of traces: ",len(data["data"]))
    return data
    
### experimental method to calculate all stats for utility
def traces_to_df_asplos_experimental(all_traces_data, is_train_ticket = True):
    """
    Method for mapping traces to astraea data structure. 
    Input: trace["data"] and boolean flag to indicate application. Boolean is for identifying unique spans (url is used in tt).
    Output: DataFrame w/ columns = ['Name','Count', 'Mean','Var','Mean_sum','Var_sum','m/std']
    """
    
    dict_spans = {}
    dict_traces = {} ## traceID, Graph, lat
    span_stats = {} ## global span stats
    span_stats_e2e = {} ## e2e latencies per span
    span_stats_summed = {} ### sum repeating spans
    end_to_end_lats = []
    end_to_end_lats_dict ={} ## traceId, lat
    span_max = {}
    span_min = {}
    span_range = {}

    i = 0
    corrupted_traces =0
    print("traces_to_df_asplos trace len now: " , len(all_traces_data))
    for trace in all_traces_data:
        
        G = nx.DiGraph()
        G_spannames = nx.DiGraph()
        traceID = trace["traceID"]
        span_ids = []
     
        for span in trace["spans"]:
            is_server = False
            ## e2e latency
            span_ids.append(span["spanID"])

            
            if is_train_ticket:
                # ****** get http url and append it to span name - unique
                for tag in span["tags"]:
                    if tag["key"] == "http.url":
                        url = tag["value"]
                    if tag["key"] == "span.kind" and tag["value"] == "server":
                        is_server = True

                urlLastPart = url.split("/")[-1]
                while (  any(ele.isupper() for ele in urlLastPart) or any(ele.isdigit() for ele in urlLastPart)):
                    url = url[0:url.rindex(urlLastPart)-1]
    #                 print("url; " + url)
                    urlLastPart = url.split("/")[-1]

                if not is_server:
                    key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] + ":" + url
                else:
                    key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
            ## if deathstar or uber traces -> keynow does not include url
            else:
                key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
#                 key_now = span["operationName"] 
                
            
            if span["references"]:
                parent = span["references"][0]["spanID"]
            else:
                parent = ""

            ## create node from span 
            node = Node(key_now,parents=parent, iden=span["spanID"], latency=span["duration"], start=span["startTime"], end=span["startTime"]+span["duration"])
            dict_spans[span["spanID"]] = span
            G.add_node(span["spanID"], node=node)

        ## construct edges
        for item in span_ids:
            span = dict_spans[item]
            if span["references"]:
                G.add_edge(span["references"][0]["spanID"],span["spanID"])

        dict_traces[traceID] = G.copy()        
        ## get spans for analysis
        local_span_stats = {}
        local_span_max = {}
        local_span_min = {}
        local_span_count = {}
        e2e_lat = 0
        
        if nx.number_of_nodes(G) != nx.number_of_edges(G) + 1:
            print("mertiko problematic trace",traceID)
            dict_traces.pop(traceID)
            corrupted_traces += 1
            continue
        
        for x in G.nodes():
            if not G.nodes[x]:
                print("Problem",traceID)
                continue
                
            span_now = G.nodes[x]['node'].name   
#             print("\n---Main span now: ", span_now, " , duration", G.nodes[x]['node'].latency)
            
            if G.in_degree(x) == 0: ## root
                end_to_end_lats.append(G.nodes[x]['node'].latency)
                end_to_end_lats_dict[traceID] = G.nodes[x]['node'].latency
                e2e_lat = G.nodes[x]['node'].latency

            if G.out_degree(x)==0: ###leaves
                 ## sum local observations
                local_span_stats[span_now] = local_span_stats.get(span_now,0) + G.nodes[x]['node'].latency
                local_span_count[span_now] = local_span_count.get(span_now,0) + 1
                
                local_span_max[span_now] = G.nodes[x]['node'].latency if G.nodes[x]['node'].latency > local_span_max.get(span_now,0) else local_span_max.get(span_now,0)
                
                local_span_min[span_now] = G.nodes[x]['node'].latency if G.nodes[x]['node'].latency < local_span_min.get(span_now,0) else local_span_min.get(span_now,0)

            else: ## intermediate spans
                span_child =nx.dfs_successors(G, source=x, depth_limit=1)
                
                child_lat = 0
                ## check concurrency
                child_dict = {}
                for key, values in span_child.items():
                    if len(values) > 1:
                        for val in values:
                            child_dict[G.nodes[val]['node'].id + "_start"] = G.nodes[val]['node'].start
                            child_dict[G.nodes[val]['node'].id + "_end"] = G.nodes[val]['node'].end
                    else:
                        child_lat = child_lat + G.nodes[values[0]]['node'].latency
#                         print("-=-= Tek child: ", G.nodes[values[0]]['node'].name, " duration: ", child_lat)
                    
                    if child_dict:
                        spans_processes = []
                        most_start = 0
                        ## sort child dict by values
                        child_dict = {k: v for k, v in sorted(child_dict.items(), key=lambda item: item[1])}
                        for key,value in child_dict.items():
                            ## if first span then set most start time
                            if len(spans_processes) == 0:
                                most_start = value
                            ## if process started add it to the list
                            if "_start" in key:
                                spans_processes.append(key)
                            else:
                                ## remove the start tracepoint
                                spans_processes.remove(key.split("_end")[0]+"_start")
                                ## if we do not have any elements in the list, then set most end
                                if len(spans_processes) == 0: 
                                    child_lat = child_lat + (value - most_start)
#                                     print("*** Check child : ", key, " , duration: ", (value - most_start), ' ,total dur: ' , child_lat)
                                    most_start = 0
                                    
                
                local_span_stats[span_now] = local_span_stats.get(span_now,0) + G.nodes[x]['node'].latency - child_lat
                local_span_count[span_now] = local_span_count.get(span_now,0) + 1
#                 print("Parent now: ", span_now, " new duration: ", local_span_stats[span_now])
        
        ### sum repeating spans and add to span_stats
        for item in local_span_stats:
            if item not in span_stats:
                span_stats[item] = []
                span_stats_e2e[item] = []
                
            span_stats[item].append(local_span_stats[item]/local_span_count[item])
            span_stats_e2e[item].append(e2e_lat) ## e2e latency per span

            if item not in span_stats_summed:
                span_stats_summed[item] = []
            span_stats_summed[item].append(local_span_stats[item])

        i = i + 1 
   # print("+Parsed traces", i)

    ### e2e correlation and eta
#     span_single = {}
#     for item in span_stats:
#         if item not in span_single:
#             span_single[item] = []
#         span_single[item].extend(span_stats[item])

#     # span_single
#     span_single["e2e"] = end_to_end_lats
#     print(span_single)
#     single_ladies = pd.DataFrame.from_dict(span_single)
#     # single_ladies
#     corr_matrix = single_ladies.corr()

## new e2e correlation for different sizes
#     corr_matrix_span ={}
#     for item in span_stats:
#         corr, _ = pearsonr(span_stats[item], span_stats_e2e[item])
#         corr_matrix_span[item]= corr
        
        
        
    ### final dataframe populated below
    datalar = []
    pd.set_option("precision", 2)

    for item in span_stats:

        datalar.append([
        item, 
        len(span_stats[item]),
        np.mean(span_stats[item]), 
        np.var(span_stats[item]),
        np.std(span_stats[item]),
        np.max(span_stats[item]),
            
        ## summed variance, mean, std for repeating spans
        np.mean(span_stats_summed[item]), 
        np.var(span_stats_summed[item]), 
        np.std(span_stats_summed[item]), 
        np.max(span_stats_summed[item]),
            
#         np.mean(span_stats_summed[item])/ np.std(span_stats_summed[item]), ### log2 == mean/std
                        
        np.percentile(span_stats_summed[item], 50),
        np.percentile(span_stats_summed[item], 99),
        
#         corr_matrix.loc['e2e'][item],
#             corr_matrix_span[item],
#             corr_matrix_span[item]*(np.percentile(span_stats_summed[item], 99)/np.percentile(span_stats_summed[item], 50))
            
                        
                        
                       ])

    df_traces = pd.DataFrame(datalar, columns = ['Name','Count', 'Mean','Var','Std','Max', 'Mean_sum','Var_sum','Std_sum', 'Max_sum', '50_sum', '99_sum']) #,'R','R*99/50']) # ,'m/std'
#                                                 'Max','Min','Range'])

    df_traces['Var_sum_cum'] = df_traces['Var_sum']/(df_traces['Var_sum'].sum()/100)
#     df_traces['R*99/50_cum'] = df_traces['R*99/50']/(df_traces['R*99/50'].sum()/100)
#     df_traces['50_sum_cum'] = df_traces['50_sum']/(df_traces['50_sum'].sum()/100)
#     return df_traces
    return df_traces, end_to_end_lats_dict, span_stats, dict_traces,end_to_end_lats,span_stats_summed



### experimental method to calculate all stats for utility -- LOG2
def traces_to_df_asplos_experimental_log2(all_traces_data, is_train_ticket = True):
    """
    Method for mapping traces to astraea data structure. 
    Input: trace["data"] and boolean flag to indicate application. Boolean is for identifying unique spans (url is used in tt).
    Output: DataFrame w/ columns = ['Name','Count', 'Mean','Var','Mean_sum','Var_sum','m/std']
    """
    
    dict_spans = {}
    dict_traces = {} ## traceID, Graph, lat
    span_stats = {} ## global span stats
    span_stats_e2e = {} ## e2e latencies per span
    span_stats_summed = {} ### sum repeating spans
    end_to_end_lats = []
    end_to_end_lats_dict ={} ## traceId, lat
    span_max = {}
    span_min = {}
    span_range = {}
    


    i = 0
    corrupted_traces =0
    print("traces_to_df_asplos trace len now: " , len(all_traces_data))
    for trace in all_traces_data:
        
        local_span_stats = {}
        local_span_count = {}
        
        traceID = trace["traceID"]
#         print("Now working on : " , traceID, " len: ", len(trace["spans"]))
        span_ids = []
     
        for span in trace["spans"]:
            is_server = False
            ## e2e latency
            span_ids.append(span["spanID"])
            
            if is_train_ticket:
                # ****** get http url and append it to span name - unique
                for tag in span["tags"]:
                    if tag["key"] == "http.url":
                        url = tag["value"]
                    if tag["key"] == "span.kind" and tag["value"] == "server":
                        is_server = True

                urlLastPart = url.split("/")[-1]
                while (  any(ele.isupper() for ele in urlLastPart) or any(ele.isdigit() for ele in urlLastPart)):
                    url = url[0:url.rindex(urlLastPart)-1]
    #                 print("url; " + url)
                    urlLastPart = url.split("/")[-1]

                if not is_server:
                    key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] + ":" + url
                else:
                    key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
            ## if deathstar or uber traces -> keynow does not include url
            else:
                #key_now = trace["processes"][span["processID"]]["serviceName"] + ":" + span["operationName"] 
                key_now = span["operationName"] 
                
            
            if span["references"]:
                parent = span["references"][0]["spanID"]
            else:
                parent = ""

            ## create node from span 
            node = Node(key_now,parents=parent, iden=span["spanID"], latency=span["duration"], start=span["startTime"], end=span["startTime"]+span["duration"])
            
            ## we do not need graph in log2 -- so just get duration
            local_span_stats[key_now] = local_span_stats.get(key_now,0) + span["duration"]
            local_span_count[key_now] = local_span_count.get(key_now,0) + 1
            
            span_max[key_now] = span["duration"] if span["duration"] > span_max.get(key_now,0) else span_max.get(key_now,0)
                
            

        ## now calculate mean for single trace w/ multiple spans
        for item in local_span_stats:
            if item not in span_stats:
                span_stats[item] = []
                span_stats_e2e[item] = []
                
            span_stats[item].append(local_span_stats[item]/local_span_count[item])
        
    ### final dataframe populated below
    datalar = []
    pd.set_option("precision", 2)
    
#     print(local_span_count)
#     print(span_stats)
#     print(span_max)

    for item in span_stats:     
        datalar.append([
                        item, 
                        local_span_count[item] if item in local_span_count else 1,
                        np.mean(span_stats[item]), 
                        span_max[item]
                       ])

    df_traces = pd.DataFrame(datalar, columns = ['Name','Count', 'Mean','Max']) # ,'m/std'
#                                                 'Max','Min','Range'])
    df_traces['Mean_cum'] = df_traces['Mean']/(df_traces['Mean'].sum()/100)
    df_traces['Max_cum'] = df_traces['Max']/(df_traces['Max'].sum()/100)

    return df_traces
