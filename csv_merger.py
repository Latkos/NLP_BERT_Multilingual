import shutil
import glob


#import csv files from folder

path = r"C:/Users/micha/Desktop/NLP_BERT_Multilingual/data/full_merging/"
allFiles = glob.glob(path + "*test*.tsv")
allFiles.sort()  # glob lacks reliable ordering, so impose your own if output order matters
print(allFiles)
with open('test.tsv', 'wb') as outfile:
    for i, fname in enumerate(allFiles):
        with open(fname, 'rb') as infile:
            if i != 0:
                infile.readline()  # Throw away header on all but first file
            # Block copy rest of file from input to output without parsing
            shutil.copyfileobj(infile, outfile)
            print(fname + " has been imported.")