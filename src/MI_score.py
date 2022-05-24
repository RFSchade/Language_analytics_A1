#================================================================#
#=============> Calculate Mutual Information Score <=============#
#================================================================#

#=====> Import modules
# System tools
import os
import argparse

# Data tools
import re
import math
import pandas as pd

# Spacy 
import spacy
# Loading language model into pipeline
nlp = spacy.load("en_core_web_sm")

#=====> Define functions
# > Load data 
def load_txt(filename):
    # Print info 
    print("[INFO] Loading data...")
    
    # Loading a text from data folder - get a new text - this one is too long
    filepath = os.path.join("in", filename)
    with open(filepath, "r") as f:
        txt = f.read()
    
    return txt

# > Normalize text 
def normalize(txt):
    # Remove punctuation
    no_punct = re.sub("[^\w\s]", '', txt)
    # Remove numbers
    no_numbers = re.sub("\d", '', no_punct)
    # Remove newline
    no_newline = re.sub("\s+", ' ', no_numbers) 
    
    return no_newline

# > Find context words 
def find_context(doc, keyword, window):
    # Print info 
    print("[INFO] Finding context words...")
    
    # Define empthy list
    context_words = []
    
    # Iterate through tokens in doc
    for token in doc: 
        if token.lower_ == keyword:
            # Context words before keyword
            for before_word in doc[token.i-(window):token.i]:
                context_words.append((before_word.i, before_word.lower_))
            # Context words after keyword    
            for after_word in doc[token.i+1:token.i+window+1]:
                context_words.append((after_word.i, after_word.lower_))
        else:
            pass
        
    return context_words

# > Calculate MI
def get_MI(doc, window, keyword, context_words):
    # Print info 
    print("[INFO] Calculating MI...")
    
    # Create list of context words
    context_word_list = [sublist[1] for sublist in context_words]
    
    # Define empthy list
    word_list = []
    # Create list of words in corpus to count word frequency
    for token in doc: 
        word_list.append(token.lower_)

    # Define variables in the MI formula that does not need a for-loop
    A = word_list.count(keyword)
    size_corpus = len(word_list)
    span = window*2

    # Defining empty list for output
    collocate_info = []

    # Get mutual informaition scores
    for word in context_word_list:
        # Define variables in the MI formula that need a for-loop
        B = word_list.count(word)
        AB = context_word_list.count(word)
        # Calculate MI score
        MI = math.log10( (AB*size_corpus) / (A * B * span) ) / math.log10(2)
        # Save informatio to list 
        collocate_info.append((word, B, AB, MI))

    return collocate_info

# > Save data 
def save_data(collocate_info, filename):
    # Print info 
    print("[INFO] Saving data...")
    
    # Defineing the name of the dataset
    dataname = filename.split(".")[0]
    
    # create a dataframe
    collocate_df = pd.DataFrame(collocate_info, columns=["collocate_term", "app_in_text", "app_in_context", "MI"])
    # Saving CSV
    outpath = os.path.join("output", f"{dataname}_df.csv")
    collocate_df.to_csv(outpath, index=False)
    
# > Parse arguments
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-d", "--data",
                    required=True, 
                    help="Name of txt data - maximum length is 1000000")
    ap.add_argument("-k", "--keyword",
                    required=True, 
                    help="Search term to find collocations for - should be lowercase")
    ap.add_argument("-w", "--window",
                    required=True, 
                    type=int,
                    help="Size of the window the script will find collocates within - +/- the keyword")
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args


#=====> Define main()
def main():
    # Get arguments
    args = parse_args()
    
    # Load data 
    txt = load_txt(args["data"])
    # Normalize
    normalized_txt = normalize(txt)
    # Create NLP document 
    doc = nlp(normalized_txt)
    # Find context words 
    context_words = find_context(doc, args["keyword"], args["window"])
    # Calculate MI
    collocate_info = get_MI(doc, args["window"], args["keyword"], context_words)
    # Save data 
    save_data(collocate_info, args["data"])
    
    # Print info 
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()

