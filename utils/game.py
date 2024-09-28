import re, copy, queue
import heapq as hq
from utils.preparation import ALPHABET, replaceText
from utils.api import chooseSimilarWord, guessWord

def validKey(key):
    '''
    Input a dictionary (cipher:plain)
    Return a boolean
    '''
    used = set()

    for c in key:
        temp = list(key[c])
        if len(temp) > 1:
            return False
        if temp[0] in used:
            return False
        used.add(temp[0])
    return True

def wordDistance(w1, w2):
    dist = 0
    w1, w2 = w1.upper(), w2.upper()
    for i in range(len(w1)):
        if w1[i] != w2[i]:
            dist += 1
    return dist

def autoGame(ciphertext, candidates):
    '''
    Input a ciphertext and key candidates
    Return a guess key and not used alphabet set
    '''

    # origin words from ciphertext
    # ciphertext ---> origin
    origin_words = re.sub(r"[^A-Z]", " ", ciphertext.upper()).split()
    
    # save options based on candidates
    # replaced tokens ---> options
    options = dict()
    for i in range(len(origin_words)):
        options[i] = set()

    first = None
    for _ in range(len(candidates)):
        _, key = hq.heappop(candidates)
        words = re.sub(r"[^A-Z]", " ", replaceText(ciphertext, key).upper()).split()
        if not first:
            first = words
        for idx, word in enumerate(words):
            options[idx].add(word)

    # make a guess key
    guess_key = dict()

    # save the best word option for each token
    # using heap structure:
    #   (calculated score, suggested word, token from first option, origin)
    best_options = []

    for i in range(len(origin_words)):

        score, word = chooseSimilarWord(list(options[i]))
        origin = origin_words[i].upper()
        hq.heappush(best_options, (score, word, first[i], origin))

        # update key
        word = word.upper()
        for idx, c in enumerate(origin):
            if not c.isalpha():
                continue
            if c in guess_key:
                guess_key[c].add(word[idx])
            else:
                guess_key[c] = set([word[idx]])

    # change set to list
    for c in guess_key:
        guess_key[c] = list(guess_key[c])           
            
    # make a valid key
    # if the previous step creates a valid key,
    #   this step is skipped

    # not used key
    not_check = set()

    secondary = queue.Queue()

    if not validKey(guess_key):        
        temp_key = dict()
        used = set()

        while best_options:
            _, word, option, origin = hq.heappop(best_options)

            # already existed
            insert = False
            for c in origin:
                if c not in temp_key:
                    insert = True
                    break
            if not insert:
                continue

            # too short phrase
            if len(word) < 4:
                secondary.put((word, option, origin))
                continue
            
            # similiar to the first option
            if wordDistance(word, option) < 2:
                collision = False
                word = word.upper()
                for idx, c in enumerate(origin):
                    if c in temp_key:
                        if temp_key[c] != word[idx]:
                            collision = True
                            break
                    elif word[idx] in used:
                        collision = True
                        break

                if not collision:
                    for idx, c in enumerate(origin):
                        if c in temp_key:
                            continue
                        temp_key[c] = word[idx]
                        used.add(word[idx])
                    continue

            # make another option with question marks
            temp = ""
            count = 0
            for c in origin:
                if c in temp_key:
                    temp += temp_key[c]
                else:
                    temp += "?"
                    count += 1
            
            guess = guessWord(temp, origin)
            if guess == None or count >= 2:
                secondary.put((word, option, origin))
            else:
                guess = guess.upper()
                for idx, c in enumerate(origin):
                    if c in temp_key or guess[idx] in used:
                        continue
                    temp_key[c] = guess[idx]
                    used.add(guess[idx])

        while not secondary.empty():

            word, option, origin = secondary.get()

            temp = ""
            for c in origin:
                if c in temp_key:
                    temp += temp_key[c]
                else:
                    if len(guess_key[c]) == 1 and guess_key[c][0] not in used:
                        temp += guess_key[c][0]
                    else:
                        temp += "?"

            guess = guessWord(temp, origin)
            if guess == None:
                guess = option
     
            guess = guess.upper()
            for idx, c in enumerate(origin):
                if c in temp_key or guess[idx] in used:
                    continue
                temp_key[c] = guess[idx]
                used.add(guess[idx])

        # fill in guess key with not checked ones
        not_used = list(set(ALPHABET) - used)
        i = 0
        for c in ALPHABET:
            if c in temp_key:
                continue
            temp_key[c] = not_used[i]
            i += 1
            not_check.add(c)

        # change set to list
        for c in temp_key:
            temp_key[c] = list(temp_key[c])

        guess_key = copy.deepcopy(temp_key)

    return guess_key, not_check
