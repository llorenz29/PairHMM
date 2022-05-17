from ctypes import sizeof
from multiprocessing.spawn import old_main_modules
from re import U
from tracemalloc import start
import numpy as np
import random
import pandas as pd
random.seed(1)





def viterbi_func(x1, x2, states, transitions, emissions):
    #n = len(x1) + len(x2)
    s = len(states)
    n1 = len(x1)
    n2 = len(x2)
    df_match = pd.DataFrame(data = np.zeros((n1+1,n2+1)), columns = list(range(0,n2+1)), index = list(range(0,n1+1)))
    df_deletions = pd.DataFrame(data = np.zeros((n1+1,n2+1)), columns = list(range(0,n2+1)), index = list(range(0,n1+1)))
    df_insertions = pd.DataFrame(data = np.zeros((n1+1,n2+1)), columns = list(range(0,n2+1)), index = list(range(0,n1+1)))
    #build dynamic programming df with probabilities, and build df to keep track of the states used
    blank_moves= np.full((n1+1,n2+1), '', dtype=object)
    insertion_moves = pd.DataFrame(data = blank_moves, columns = list(range(0,n2+1)), index = list(range(0,n1+1)))
    deletion_moves = pd.DataFrame(data = blank_moves, columns = list(range(0,n2+1)), index = list(range(0,n1+1)))
    match_moves = pd.DataFrame(data = blank_moves, columns = list(range(0,n2+1)), index = list(range(0,n1+1)))
    #base case - 0,0
    start_prob = 0.33
    df_match[0][0] = start_prob
    df_insertions[0][0] = start_prob
    df_deletions[0][0] = start_prob
    #first column, first row
    for i in range(1,n1+1):
        df_match[0][i] = -1
        df_insertions[0][i] = df_insertions[0][i-1] * transitions['I']['I'] * emissions[x1[i-1]]['-']
        df_deletions[0][i] = -1
    for j in range(1,n2+1):
        df_match[j][0] = -1
        df_insertions[j][0] = -1
        df_deletions[j][0] = df_deletions[j-1][0] * transitions['D']['D'] * emissions[x2[j-1]]['-'] 

    for i in range(1,len(x1)+1):
        for j in range(1,len(x2)+1):
        #state match
            options = (transitions['M']['M'] * df_match[j-1][i-1], transitions['I']['M'] * df_insertions[j-1][i-1], transitions['D']['M'] * df_deletions[j-1][i-1])
            maximum = max(options)
            old_state_index = list(options).index(maximum)
            #old_state = states[old_state_index]
            previous_moves_list = (match_moves[j-1][i-1], insertion_moves[j-1][i-1], deletion_moves[j-1][i-1])
            match_moves[j][i] = previous_moves_list[old_state_index] + 'M'
            df_match[j][i] = maximum * emissions[x1[i-1]][x2[j-1]]

            #state insertion
            options = (transitions['M']['I'] * df_match[j][i-1], transitions['I']['I'] * df_insertions[j][i-1])
            maximum = max(options)
            old_state_index = list(options).index(maximum)
            #old_state = states[old_state_index]
            previous_moves_list = (match_moves[j][i-1], insertion_moves[j][i-1])
            insertion_moves[j][i] = previous_moves_list[old_state_index] + 'I'
            df_insertions[j][i] = maximum * emissions[x1[i-1]]['-']


            #state deletion
            options = (transitions['M']['D'] * df_match[j-1][i], transitions['D']['D'] * df_deletions[j-1][i])
            maximum = max(options)
            old_state_index = list(options).index(maximum)
            previous_moves_list = (match_moves[j-1][i], deletion_moves[j-1][i])
            deletion_moves[j][i] = previous_moves_list[old_state_index] + 'D'
            df_deletions[j][i] = maximum * emissions['-'][x2[j-1]]
    return df_match, df_insertions, df_deletions, match_moves, insertion_moves, deletion_moves
    




if __name__ == "__main__":
    with open('final_bioinfo.txt') as f:
        lines = f.readlines()
    x1 = lines[0][:-1]
    x2 = lines[1][:-1]
    alphabet = lines[3].split()
    states = lines[5].split()
    #build transition df
    transition_m = []
    for i in range(len(states)):
        transition_line = lines[8+i].split()[1:]
        line_float = list(map(float, transition_line))
        transition_m.append(line_float)
    transition_df = pd.DataFrame(transition_m, index = states, columns = states)
    print('transition')
    print(transition_df)
    #build emission df
    emission_m = []
    start_row = 8 + len(alphabet)
    for i in range(len(alphabet)):
        emission_line = lines[start_row+i].split()[1:]
        line_float = list(map(float, emission_line))
        emission_m.append(line_float)
    emission_df = pd.DataFrame(emission_m, index = alphabet, columns = alphabet)
    print('emission')
    print(emission_df)


    df_match, df_insert, df_delete, match_moves, insert_moves, delete_moves = viterbi_func(x1, x2, states, transition_df, emission_df)
    n1 = len(x1)
    n2 = len(x2)
    options_probs = (df_match[n2][n1], df_insert[n2][n1], df_delete[n2][n1])
    maximum = max(options_probs)
    index = list(options_probs).index(maximum)
    options_moves = (match_moves[n2][n1], insert_moves[n2][n1], delete_moves[n2][n1])
    best_path = options_moves[index]
    print(best_path)

    #Develop Alignment
    x1_copy = x1[:]
    x2_copy = x2[:]
    line1 = ''
    line2 = ''
    for letter in best_path:
        if(letter == 'M'):
            line1 += x1_copy[0]
            line2 += x2_copy[0]
            if(len(x1_copy)>1):
                x1_copy = x1_copy[1:]
            if(len(x2_copy)>1):
                x2_copy = x2_copy[1:]
        elif(letter == 'I'):
            line1 += x1_copy[0]
            line2 += '-'
            if(len(x1_copy)>1):
                x1_copy = x1_copy[1:]
        elif(letter == 'D'):
            line1 += '-'
            line2 += x2_copy[0]
            if(len(x2_copy)>1):
                x2_copy = x2_copy[1:]

    #Develop new Transition matrix
    new_transition_df = pd.DataFrame(data = np.full((3,3), 5, dtype = float), index = states, columns = states)
    for i in range(0, len(best_path)-1):
        start_state = best_path[i]
        end_state = best_path[i+1]
        new_transition_df[start_state][end_state] = new_transition_df[start_state][end_state] + 1
    sums = new_transition_df.sum(axis=1) 
    new_transition_df = new_transition_df / sums


    #Develop new emissions matrix
    new_emission_df = pd.DataFrame(data = np.full((len(alphabet), len(alphabet)), 5, dtype = float), index = alphabet, columns = alphabet)
    for i in range(len(line1)):
        letter1 = line1[i]
        letter2 = line2[i]
        new_emission_df[letter1][letter2] = new_emission_df[letter1][letter2] + 1
        new_emission_df[letter2][letter1] = new_emission_df[letter2][letter1] + 1
    print(new_emission_df)
    sums = new_emission_df.sum(axis=1)
    new_emission_df = new_emission_df / sums
    print(new_emission_df)
    print(line1)
    print(line2)



#Rerun with new matrices
    df_match, df_insert, df_delete, match_moves, insert_moves, delete_moves = viterbi_func(x1, x2, states, new_transition_df, new_emission_df)
    n1 = len(x1)
    n2 = len(x2)
    options_probs = (df_match[n2][n1], df_insert[n2][n1], df_delete[n2][n1])
    maximum = max(options_probs)
    index = list(options_probs).index(maximum)
    options_moves = (match_moves[n2][n1], insert_moves[n2][n1], delete_moves[n2][n1])
    best_path = options_moves[index]
    print(best_path)

    #Develop Alignment
    x1_copy = x1[:]
    x2_copy = x2[:]
    line1 = ''
    line2 = ''
    for letter in best_path:
        if(letter == 'M'):
            line1 += x1_copy[0]
            line2 += x2_copy[0]
            if(len(x1_copy)>1):
                x1_copy = x1_copy[1:]
            if(len(x2_copy)>1):
                x2_copy = x2_copy[1:]
        elif(letter == 'I'):
            line1 += x1_copy[0]
            line2 += '-'
            if(len(x1_copy)>1):
                x1_copy = x1_copy[1:]
        elif(letter == 'D'):
            line1 += '-'
            line2 += x2_copy[0]
            if(len(x2_copy)>1):
                x2_copy = x2_copy[1:]

    print(line1)
    print(line2)

    print(new_transition_df)
    print(new_emission_df)



