import numpy as np
import pandas as pd
import itertools

import re, math, sys
from collections import defaultdict

def sepDistrib(data):
    for idx, line in enumerate(data):
        if line == "":
            end_header = idx
            break

    for idx, line in enumerate(data):
        if line == "\\init":
            start_init = idx+1
        elif line == "\\transition":
            end_init = idx-3
            start_transition = idx+1
        elif line == "\\emission":
            end_transition = idx-2
            start_emission = idx+1

    init = data[start_init:end_init]
    transition = data[start_transition:end_transition]
    emission = data[start_emission:]
    return init, transition, emission

def toProbDict(transition, emission):
    transition_dict = defaultdict(dict)
    for line in transition:
        match = re.match("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t##", line)
        try:
            t1t2 = match.group(1)
            t2t3 = match.group(2)
            prob = float(match.group(3))
            lgprob = float(match.group(4))
            if prob > 1 or prob < 0:
                print("warning: the prob is not in [0,1] range: $line", line)
                continue
            try:
                transition_dict[t1t2][t2t3] = (prob, lgprob)
            except:
                transition_dict[t1t2] = {}
                transition_dict[t1t2][t2t3] = (prob, lgprob)
        except:
            print(f"Cannot parse {line}")

    emission_dict = defaultdict(dict)
    vocab = set()
    for line in emission:
        match = re.match("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t##", line)
        try:
            pos = match.group(1)
            word = match.group(2)
            prob = float(match.group(3))
            lgprob = float(match.group(4))
        except:
            match = re.match("([^\t]+)\t([^\t]+)\t([^\t]+)", line)
            pos = match.group(1)
            word = match.group(2)
            prob = float(match.group(3))
            lgprob = math.log10(prob)
        if prob > 1 or prob < 0:
            print("warning: the prob is not in [0,1] range: $line", line)
            continue
        vocab.add(word)
        emission_dict[pos][word] = (prob, lgprob)
            
    return transition_dict, emission_dict, vocab

def viterbiFindBestPath(N, T, line, transition_dict, emission_dict, vocab, beam_width):
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))
    
    # Init
    for i, state in enumerate(transition_dict):
        if line[0] in vocab:
            word = line[0]
        else:
            word = "<unk>"
        if word in emission_dict[state]:
            emiss_prob = emission_dict[state][word][0]
        else:
            emiss_prob = 0
        trans_prob = 1
        viterbi[i][0] =  trans_prob * emiss_prob
        backpointer[i][0] = 0
    sorted_idx = np.argsort(viterbi[:,0]).tolist()
    sorted_idx.reverse()
    beststates_idx = sorted_idx[:beam_width]

    # Recursion
    for i, word in enumerate(line[1:]):
        if word in vocab:
            pass
        else:
            word = "<unk>"
        for j, state in enumerate(transition_dict):
            if word in emission_dict[state]:
                emiss_prob = emission_dict[state][word][0]
            else:
                emiss_prob = 0
            previous_list = []
            for k in beststates_idx:
                previous_state = states[k]
                viterbi_val = viterbi[k,i]
                if state in transition_dict[previous_state]:
                    trans_prob = transition_dict[previous_state][state][0]
                else:
                    trans_prob = 0
                previous_list.append(viterbi_val*trans_prob)
            np_previous_list = np.asarray(previous_list)
            argmax_idx = np_previous_list.argmax()
            argmax = beststates_idx[argmax_idx]
            viterbi[j,i+1] = max(previous_list) * emiss_prob
            backpointer[j,i+1] = argmax
        sorted_idx = np.argsort(viterbi[:,i+1]).tolist()
        sorted_idx.reverse()
        beststates_idx = sorted_idx[:beam_width]

    # Terminate
    bestpathprob = viterbi.max(axis=0)[-1]
    bestpathpointer = viterbi[:,-1].argmax(axis=0)
    pointer = bestpathpointer
    bestpath = []
    for i in range(len(line)):
        bestpath.append(pointer)
        pointer = int(backpointer[pointer][len(line)-1-i])
    bestpath.reverse()
    return bestpathprob, bestpath

def path2seq(bestpath, states):
    bestseq = []
    for i in bestpath:
        bestseq.append(states[i])
    return bestseq


hmm = sys.argv[1]
inp = sys.argv[2]
out = sys.argv[3]

with open(hmm) as f:
    data = f.readlines()
    data = [re.sub("\s+", "\t", line.replace("\n", "")) for line in data]

init, transition, emission = sepDistrib(data)
transition_dict, emission_dict, vocab= toProbDict(transition, emission)
states = list(transition_dict.keys())
beam_width = int(math.floor(len(states)*0.01))
N = len(states)

with open(out, 'w') as f:
    with open(inp, 'r') as test:
        for obs in test:
            obs = obs.split()
            line = ['<s>']
            line.extend(obs)
            T = len(line)
            bestpathprob, bestpath = viterbiFindBestPath(N, T, line, transition_dict, emission_dict, vocab, beam_width)
            bestseq = " ".join(path2seq(bestpath, states))
            lgprob = math.log10(bestpathprob)
            f.write(f"{' '.join(line)} => {bestseq} {lgprob}")
            f.write("\n")