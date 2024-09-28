import urllib.request
import heapq as hq
import json, math
from utils.preparation import ALPHABET

# reference: https://www.datamuse.com/api/

def patternMatching(w1, w2):

    def pattern(w):
        used = dict()
        result = ""
        idx = 0

        for c in list(w.upper()):
            if not c.isalpha():
                return None
            if c not in used:
                used[c] = ALPHABET[idx]
                idx += 1
            result += used[c]

        return result
    
    return pattern(w1) == pattern(w2)

def searchWord(word):
    url = f'https://api.datamuse.com/words?sp={word}'
    response = urllib.request.urlopen(url)
    message = response.read().decode('utf8')

    return json.loads(message)

def chooseSimilarWord(options):
    '''
    Find a similar word among options
    Return the most similar word if available
    '''
    candidates = dict()

    for option in options:
        data = searchWord(option)
        if not len(data):
            continue

        for i in range(min(len(data), 5)):
            word = data[i].get('word')
            if not word.isalpha():
                continue

            if len(word) != len(option):
                continue

            if not patternMatching(word, option):
                continue
            
            score = data[i].get('score')
            if word in candidates:
                candidates[word][0] += 1
            else:
                candidates[word] = [1, score]

    if not candidates:
        return 0, options[0]
    
    ranking = []
    for word in candidates:
        count, score = candidates[word]
        weight = count * (1 + math.log10(score)) * len(word)
        hq.heappush(ranking, (-weight, word))

    return hq.heappop(ranking)

def guessWord(keyword, origin):
    '''
    Guess a word for the keyword which including question marks
    Return the word if available
    '''
    data = searchWord(keyword)

    for i in range(len(data)):
        word = data[i].get('word')
        if patternMatching(word, origin):
            return word
        
    return None
