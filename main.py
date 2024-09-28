from utils.controller import main
from utils.preparation import createFiles

if __name__ == "__main__":
    n = 50
    size = 100 # number of characters

    # create divided text files in plains folder
    createFiles(n, size)

    f = open("result.txt", 'w')
    f.close()

    for i in range(n):
        main(f"plains/{i}.txt")