import numpy as np
import random, copy, re, time
import heapq as hq
from nltk.util import ngrams
from utils.preparation import ALPHABET, FREQUENCY, FREQUENCY_KEY, replaceText
from utils.game import autoGame

ALPHABET_SIZE = len(ALPHABET)

FREQUENCY_DISTRIBUTION = []
for i, c in enumerate(FREQUENCY_KEY):
    FREQUENCY_DISTRIBUTION.extend([i] * int(10000 * FREQUENCY[c]))

# reference: http://norvig.com/mayzner.html
# digram: row-column
DIGRAM_FREQUENCY = np.array([
    #  e      t      a      o      i      n      s      r      h      l      d      c      u      m      f      p      g      w      y      b      v      k      x      j      q      z
    [0.378, 0.413, 0.688, 0.073, 0.183, 1.454, 1.339, 2.048, 0.026, 0.530, 1.168, 0.477, 0.031, 0.374, 0.163, 0.172, 0.120, 0.117, 0.144, 0.027, 0.255, 0.016, 0.214, 0.005, 0.057, 0.005],
    [1.205, 0.171, 0.530, 1.041, 1.343, 0.010, 0.337, 0.426, 3.556, 0.098, 0.001, 0.026, 0.255, 0.026, 0.006, 0.004, 0.002, 0.082, 0.227, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.004],
    [0.012, 1.487, 0.003, 0.005, 0.316, 1.985, 0.871, 1.075, 0.014, 1.087, 0.368, 0.448, 0.119, 0.285, 0.074, 0.203, 0.205, 0.060, 0.217, 0.230, 0.205, 0.105, 0.019, 0.012, 0.002, 0.012],
    [0.039, 0.442, 0.057, 0.210, 0.088, 1.758, 0.290, 1.277, 0.021, 0.365, 0.195, 0.166, 0.870, 0.546, 1.175, 0.224, 0.094, 0.330, 0.036, 0.097, 0.178, 0.064, 0.019, 0.007, 0.001, 0.003],
    [0.385, 1.123, 0.286, 0.835, 0.023, 2.433, 1.128, 0.315, 0.002, 0.432, 0.296, 0.699, 0.017, 0.318, 0.203, 0.089, 0.255, 0.001, 0.000, 0.099, 0.288, 0.043, 0.022, 0.001, 0.011, 0.064],
    [0.692, 1.041, 0.347, 0.465, 0.339, 0.073, 0.509, 0.009, 0.011, 0.064, 1.352, 0.416, 0.079, 0.028, 0.067, 0.006, 0.953, 0.006, 0.098, 0.004, 0.052, 0.052, 0.003, 0.011, 0.006, 0.004],
    [0.932, 1.053, 0.218, 0.398, 0.550, 0.009, 0.405, 0.006, 0.315, 0.056, 0.005, 0.155, 0.311, 0.065, 0.017, 0.191, 0.002, 0.024, 0.057, 0.008, 0.001, 0.039, 0.000, 0.000, 0.007, 0.000],
    [1.854, 0.362, 0.686, 0.727, 0.728, 0.160, 0.397, 0.121, 0.015, 0.086, 0.189, 0.121, 0.128, 0.175, 0.032, 0.042, 0.100, 0.013, 0.248, 0.027, 0.069, 0.097, 0.001, 0.001, 0.001, 0.001],
    [3.075, 0.130, 0.926, 0.485, 0.763, 0.026, 0.015, 0.084, 0.001, 0.013, 0.003, 0.001, 0.074, 0.013, 0.002, 0.001, 0.000, 0.005, 0.050, 0.004, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.829, 0.124, 0.528, 0.387, 0.624, 0.006, 0.142, 0.010, 0.002, 0.577, 0.253, 0.012, 0.135, 0.023, 0.053, 0.019, 0.006, 0.013, 0.425, 0.007, 0.035, 0.020, 0.000, 0.000, 0.000, 0.000],
    [0.765, 0.003, 0.151, 0.188, 0.493, 0.008, 0.126, 0.085, 0.005, 0.032, 0.043, 0.003, 0.148, 0.018, 0.003, 0.002, 0.031, 0.008, 0.050, 0.003, 0.019, 0.000, 0.000, 0.005, 0.001, 0.000],
    [0.651, 0.461, 0.538, 0.794, 0.281, 0.001, 0.023, 0.149, 0.598, 0.149, 0.002, 0.083, 0.163, 0.003, 0.001, 0.001, 0.001, 0.000, 0.042, 0.001, 0.000, 0.118, 0.000, 0.000, 0.005, 0.001],
    [0.147, 0.405, 0.136, 0.011, 0.101, 0.394, 0.454, 0.543, 0.001, 0.346, 0.091, 0.188, 0.001, 0.138, 0.019, 0.136, 0.128, 0.000, 0.005, 0.089, 0.003, 0.005, 0.004, 0.001, 0.000, 0.002],
    [0.793, 0.001, 0.565, 0.337, 0.318, 0.009, 0.093, 0.003, 0.001, 0.005, 0.001, 0.004, 0.115, 0.096, 0.004, 0.239, 0.001, 0.001, 0.062, 0.090, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.237, 0.082, 0.164, 0.488, 0.285, 0.000, 0.006, 0.213, 0.000, 0.065, 0.000, 0.001, 0.096, 0.001, 0.146, 0.000, 0.001, 0.000, 0.009, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.478, 0.106, 0.324, 0.361, 0.123, 0.001, 0.055, 0.474, 0.094, 0.263, 0.001, 0.001, 0.105, 0.016, 0.001, 0.137, 0.000, 0.001, 0.012, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000],
    [0.385, 0.015, 0.148, 0.132, 0.152, 0.066, 0.051, 0.197, 0.228, 0.061, 0.003, 0.000, 0.086, 0.010, 0.001, 0.000, 0.025, 0.001, 0.026, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.361, 0.007, 0.385, 0.222, 0.374, 0.079, 0.035, 0.031, 0.379, 0.015, 0.004, 0.001, 0.001, 0.001, 0.002, 0.001, 0.000, 0.000, 0.002, 0.001, 0.000, 0.001, 0.000, 0.000, 0.000, 0.000],
    [0.093, 0.017, 0.016, 0.150, 0.029, 0.013, 0.097, 0.008, 0.001, 0.015, 0.007, 0.014, 0.001, 0.024, 0.001, 0.025, 0.003, 0.003, 0.000, 0.004, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002],
    [0.576, 0.017, 0.146, 0.195, 0.107, 0.002, 0.046, 0.112, 0.001, 0.233, 0.002, 0.002, 0.185, 0.003, 0.000, 0.001, 0.000, 0.000, 0.176, 0.011, 0.004, 0.000, 0.000, 0.023, 0.000, 0.000],
    [0.825, 0.000, 0.140, 0.071, 0.270, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.005, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.214, 0.001, 0.017, 0.006, 0.098, 0.051, 0.048, 0.003, 0.003, 0.011, 0.001, 0.000, 0.003, 0.002, 0.002, 0.001, 0.003, 0.002, 0.006, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.022, 0.047, 0.030, 0.003, 0.039, 0.000, 0.000, 0.000, 0.004, 0.001, 0.000, 0.026, 0.005, 0.000, 0.002, 0.067, 0.000, 0.000, 0.003, 0.000, 0.002, 0.000, 0.003, 0.000, 0.000, 0.000],
    [0.052, 0.000, 0.026, 0.054, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.059, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.148, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.050, 0.000, 0.025, 0.007, 0.012, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003],
])

def calculateFrequency(text):
    '''
    Calculate a frequency of each character
    Return a list which contains each frequency from A to Z
    '''
    freq = [0 for _ in range(len(ALPHABET))]
    n = 0

    for c in text.upper():
        idx = ord(c) - 65
        if idx < 0 or idx > 25:
            continue
        freq[idx] += 1
        n += 1
    
    return [(cnt / n) * 100 for cnt in freq]

def initialKey(values):
    '''
    Create an initial key
    Return a list with the descending order by frequency
    '''
    freq = [[ALPHABET[idx], value] for idx, value in enumerate(values)]
    freq.sort(key=lambda x: x[1], reverse=True)

    return [c[0] for c in freq]

def noBlankNgram(text, n):
    '''
    Return a ngram list
        Each ngram don't have a blank
    '''
    text = re.sub(r"[^A-Z]", " ", text.upper())
    parts = ngrams(text, n)
    result = []

    for part in parts:
        part = "".join(part)
        if part.isalpha():
            result.append(part)

    return result

def initialMatrix(text, key):
    '''
    Create an initial matrix
    Return the matrix
    '''
    matrix = np.zeros((ALPHABET_SIZE, ALPHABET_SIZE))
    digrams = noBlankNgram(text, 2)

    for digram in digrams:
        i1, i2 = key.index(digram[0]), key.index(digram[1])
        matrix[i1, i2] += 1
    
    # transfer to percentage
    for i in range(ALPHABET_SIZE):
        for j in range(ALPHABET_SIZE):
            matrix[i, j] = matrix[i, j] / len(digrams) * 100

    return matrix

def calculateScore(matrix):
    '''
    Return a float value
    '''
    return abs(matrix - DIGRAM_FREQUENCY).sum()

def swapMatrix(matrix, i1, i2):
    '''
    Swap rows and columns of matrix
    '''
    matrix[[i1, i2]] = matrix[[i2, i1]]
    matrix[:, [i1, i2]] = matrix[:, [i2, i1]]

def findKeys(matrix, key, method=0):
    '''
    Find candidate for key
        Can change method by parameter
        If you don't pass it, default method is activated
        Default (0) is Jakobsen's version
    Return best score and list for key
    '''
    best_score = calculateScore(matrix)
    cnt_limit = 3000
    cnt = 0

    # jakobsen
    if method == 0:
        while cnt < cnt_limit:
            for i in range(1, ALPHABET_SIZE):
                for j in range(ALPHABET_SIZE - i):
                    i1, i2 = j, j+i

                    swapMatrix(matrix, i1, i2)
                    score = calculateScore(matrix)

                    if score < best_score:
                        best_score = score
                        key[i1], key[i2] = key[i2], key[i1]
                    else:
                        swapMatrix(matrix, i1, i2)

                    cnt += 1                

                    if cnt >= cnt_limit:
                        break

                if cnt >= cnt_limit:
                    break
                
    else:
        while cnt < cnt_limit:
            i1, i2 = random.sample(FREQUENCY_DISTRIBUTION, 2)

            swapMatrix(matrix, i1, i2)
            score = calculateScore(matrix)

            if score < best_score:
                best_score = score
                key[i1], key[i2] = key[i2], key[i1]
            else:
                swapMatrix(matrix, i1, i2)

            cnt += 1

    return best_score, key

def printKey(key):
    '''
    Input a dictionary (cipher:[plain])
    Print key as an alphabetic order
    '''
    pairs = []
    unused = set(ALPHABET)

    for c in key:
        p = key[c][0]
        pairs.append([p, c])
        unused.remove(p)
    
    for c in unused:
        pairs.append([c, c])

    pairs.sort(key=lambda x: x[0])

    print("Key:", "".join([pair[1] for pair in pairs]))

def calculateAccuracy(guess, origin):
    length = len(guess)
    cnt = 0

    for i in range(length):
        if guess[i] == origin[i]:
            cnt += 1

    return (cnt / length) * 100

def makeKey():
    '''
    Generate a random key
    '''
    temp = list(ALPHABET)
    random.shuffle(temp)
    return "".join(temp)

def encrypt(text, key):
    '''
    Generate a ciphertext based on key
    '''
    ciphertext = ""

    for c in text:
        if c.isalpha():
            if c.isupper():
                ciphertext += key[ALPHABET.index(c)]
            else:
                ciphertext += key[ALPHABET.index(c.upper())].lower()
        else:
            ciphertext += c

    return ciphertext

def calculateKeyAccuracy(key, not_check, answer):
    cnt = 0
    for c in key:
        if c in not_check:
            continue
        p = key[c][0]
        if c == answer[ALPHABET.index(p)]:
            cnt += 1
    
    return cnt

def main(file):
    plaintext = open(file).read()
    answer_key = makeKey()

    ciphertext = encrypt(plaintext, answer_key)  
    frequency = calculateFrequency(ciphertext)

    init_key = initialKey(frequency)
    init_matrix = initialMatrix(ciphertext, init_key)

    # Phase 1
    results = []
    
    start = time.time()
    for i in range(10):
        result = findKeys(np.copy(init_matrix), copy.deepcopy(init_key), i)
        results.append(result)
    end = time.time()
    phase1_time = end - start

    candidates = []

    for result in results:
        score, key = result
        hq.heappush(candidates, (score, key))
    
    # Phase 1 result
    _, jakobsen = results[0]
    jakobsen = ''.join(jakobsen)
    _, currentBest = candidates[0]
    currentBest = ''.join(currentBest)
    
    jakobsen_only = calculateAccuracy(replaceText(ciphertext, jakobsen), plaintext)
    phase1_best = calculateAccuracy(replaceText(ciphertext, currentBest), plaintext)

    # Phase 2
    start = time.time()
    guess_key, not_check = autoGame(ciphertext, candidates)
    end = time.time()
    phase2_time = end - start

    replaced = replaceText(ciphertext, guess_key)
    accuracy = calculateAccuracy(replaced, plaintext)
    key_accuracy = calculateKeyAccuracy(guess_key, not_check, answer_key)

    f = open("result.txt", 'a')
    f.write(f"{accuracy} {key_accuracy}/(26-{len(not_check)}) {jakobsen_only}/{phase1_best} {phase1_time:.5f} {phase2_time:.5f}\n")
    f.close()

