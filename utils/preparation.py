import unidecode

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# reference: http://norvig.com/mayzner.html
FREQUENCY = {
    "E": 0.1249,
    "T": 0.0928,
    "A": 0.0804,
    "O": 0.0764,
    "I": 0.0757,
    "N": 0.0723,
    "S": 0.0651,
    "R": 0.0628,
    "H": 0.0505,
    "L": 0.0407,
    "D": 0.0382,
    "C": 0.0334,
    "U": 0.0273,
    "M": 0.0251,
    "F": 0.024,
    "P": 0.0214,
    "G": 0.0187,
    "W": 0.0168,
    "Y": 0.0166,
    "B": 0.0148,
    "V": 0.0105,
    "K": 0.0054,
    "X": 0.0023,
    "J": 0.0016,
    "Q": 0.0012,
    "Z": 0.0009,
}

FREQUENCY_KEY = "".join(FREQUENCY.keys())

def createFiles(n, size):

    print(f"{n} files: Each file has a sentence {size}-long")

    f = open("alice.txt", 'r')

    lines = f.readlines() # 140935
    texts = []
    for line in lines:
        line = line.strip()
        if len(line):
            unaccented = unidecode.unidecode(line) # remove accented words
            texts.extend(unaccented.split())

    f.close()

    idx = 0
    for i in range(n):

        f = open(f"plains/{i}.txt", 'w')

        data = ""
        while len(data) < size:
            data += texts[idx]
            data += " "
            idx += 1

        f.write(data[:size])
        f.close()

def replaceText(text, key):
    '''
    Replace text with key (dictionary (cipher:[plain]) or string)
    Return a string
    '''
    guess = ""

    if isinstance(key, dict):
        for c in text:
            if c.isalpha():
                if c.isupper():
                    guess += key[c][0]
                else:
                    guess += key[c.upper()][0].lower()
            else:
                guess += c
    
    if isinstance(key, str) or isinstance(key, list):
        for c in text:
            if c.isalpha():
                if c.isupper():
                    guess += FREQUENCY_KEY[key.index(c)]
                else:
                    guess += FREQUENCY_KEY[key.index(c.upper())].lower()
            else:
                guess += c

    return guess