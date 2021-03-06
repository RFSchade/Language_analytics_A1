# Language Analytics Assignment 1
## Assignment description
This repository contains my solution to assignment 1 for the Language Analytics course at Aarhus university. The goal of this assignment is to write a program that can take a user-defined search term, window size, and text and find all the context words which appear ± the window size from the search term in that text, as well as calculate the mutual information (MI) score for each context word.    
The script was made to run on data sourced from the 100 English Novels corpus found here: https://github.com/computationalstylistics/100_english_novels

## Methods
The script loads a user-defined text and removes punctuation, digits, and newlines. It then finds all instances of a user-defined search term in the text, as well as the context words within the user-defined window. The script then calculates the MI score for each context word (MI as defined in the formula found on the website for the [British National Corpus](https://www.english-corpora.org/mutualInformation.asp))

## Repository structure
in: Folder for input data    
notebooks: Folder for experimental code    
output: Folder for the output generated by the script – at present it contains 1 file:    
- Doyle_Hound_1902_df.csv
    - Generated by running MI_score.py – a dataframe containing of 4 columns and 51 rows containing information about the context words of the search term. The column “collocation_term” contains a context word, “app_in_text” contains the nr. of times it appears in the text, “app_in_context” contains the nr. of times it appears in context with the user-defined search term, and “MI” contains its mutual information score.     

src: Folder for python scripts    
- \_\_init__.py
- MI_score.py

github_link.txt: link to github repository    
requirements.txt: txt file containing the modules required to run the code    

## Usage
Modules listed in requirements.txt should be installed before the script is run.    
The format of the input data should be a .txt file containing the text to be analyzed. The maximum length of this data is 1.000.000 characters.     
__MI_score.py__    
To identify and score collocation terms in a text, run MI_score.py from the Language_analytics_A1 repository folder. The script has three arguments:    
- _-d or --data: Name of txt data - maximum length is 1000000 characters – this argument is required_
- _-k or --keyword: Search term to find collocations for – the term should be in all lowercase – this argument is required_
- _-w or --window: Size of the window the script will find collocates within - +/- the keyword – this argument is required_

Example of code running the script from the terminal:
```
python src/MI_score.py -d text_file.txt -k keyword -w 10
```

## Discussion of results
I ran the script on the Doyle_Hound_1902.txt from the 100 English Novels corpus, using the keyword “murder” and a window size of 10.     
The words with the highest MI scores in this case are words like “accessory”, “coldblooded”, and “deliberate” (row 51, 9 and 10 of Doyle_Hound_1902_df.csv respectively), which seems to make intuitive sense, but also the word “answered” (row 4) which has a less obvious connection to the keyword (perhaps it might owe its high MI score to sparse usage in general?), and “mur” (row 11), which seems to be an artifact of some sort (perhaps the product of a phonetically written stammer?). 
