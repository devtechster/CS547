# homework 1
# goal: tokenize, index, boolean query
# exports: 
# student - a populated and instantiated ir4320.Student object
# Index - a class which encapsulates the necessary logic for indexing and searching a corpus of text documents
# ########################################
# first, create a student object
# ########################################

import glob
import os
import cs547
import PorterStemmer

MY_NAME = "Adhiraj Nitin Budukh"
MY_ANUM  = 901004534
MY_EMAIL = "abudukh@wpi.edu"

# the COLLABORATORS list contains tuples of 2 items, the name of the helper
# and their contribution to your homework
COLLABORATORS = [('Di You','Helped me understand Python and its functionalities as well as the guided in the code implementation')]

# Set the I_AGREE_HONOR_CODE to True if you agree with the following statement
# "I do not lie, cheat or steal, or tolerate those who do."
I_AGREE_HONOR_CODE = True

# this defines the student object
student = cs547.Student(
    MY_NAME,
    MY_ANUM,
    MY_EMAIL,
    COLLABORATORS,
    I_AGREE_HONOR_CODE
    )


# our index class definition will hold all logic necessary to create and search
# an index created from a directory of text files 
class Index(object):
    def __init__(self):
        # _inverted_index contains terms as keys, with the values as a list of
        # document indexes containing that term
        self._inverted_index = {}
        # _documents contains file names of documents
        self._documents = []
        # example:
        #   given the following documents:
        #     doc1 = "the dog ran"
        #     doc2 = "the cat slept"
        #   _documents = ['doc1', 'doc2']
        #   _inverted_index = {
        #      'the': [0,1],
        #      'dog': [0],
        #      'ran': [0],
        #      'cat': [1],
        #      'slept': [1]
        #      }

#---------------------------------------------------------------------------------------------------------------------------------------------------#
    # index_dir( base_path )
    # purpose: crawl through a nested directory of text files and generate and inverted index of the contents
    # preconditions: none
    # returns: num of documents indexed
    # hint: glob.glob()
    # parameters:base_path - a string containing a relative or direct path to a directory of text files to be indexed.

    def index_dir(self, base_path):
        num_files_indexed = 0
            
        for filename in glob.glob(base_path + '*.txt'):
            with open(filename, encoding="utf8") as f:
                document_name = filename[(filename.rfind('/') + 1) : ]
                self._documents.append(document_name)
                cur_index = len(self._documents) - 1

                text = f.read()
                tokens = self.tokenize(text)
                stemmed_tokens = self.stemming(tokens)
                stemmed_tokens = list(set(stemmed_tokens))

            for single_token in stemmed_tokens:
                if single_token not in self._inverted_index:
                    self._inverted_index[single_token] = [cur_index]
                else:
                    self._inverted_index[single_token].append(cur_index)
            f.close()

            num_files_indexed +=1
        return num_files_indexed

#---------------------------------------------------------------------------------------------------------------------------------------------------#
    # tokenize( text )
    # purpose: convert a string of terms into a list of tokens.        
    # convert the string of terms in text to lower case and replace each character in text, 
    # which is not an English alphabet (a-z) and a numerical digit (0-9), with whitespace.
    # preconditions: none
    # returns: list of tokens contained within the text
    # parameters:text - a string of terms.
    def tokenize(self, text):
        tokens = []
        # replace char into ' ' if isalum() return false
        # then convert into lower case and split
        tokens = ("".join(ch if ch.isalnum() else ' ' for ch in text).lower().split())
        return tokens
#---------------------------------------------------------------------------------------------------------------------------------------------------#
    # purpose: convert a string of terms into a list of tokens.        
    # convert a list of tokens to a list of stemmed tokens,     
    # preconditions: tokenize a string of terms
    # returns: list of stemmed tokens
    # parameters:tokens - a list of tokens.

    def stemming(self, tokens):
        stemmed_tokens = []
        stemmer = PorterStemmer.PorterStemmer()           
        for single_token in tokens:
            stemmed_tokens.append(stemmer.stem(single_token,0,len(single_token)-1))
        #print('----',stemmed_tokens,'\n')           
        return stemmed_tokens

#---------------------------------------------------------------------------------------------------------------------------------------------------#
    # boolean_search( text )
    # purpose: searches for the terms in "text" in our corpus using logical OR or logical AND. 
    # If "text" contains only single term, search it from the inverted index. If "text" contains three terms including "or" or "and", 
    # do OR or AND search depending on the second term ("or" or "and") in the "text".  
    # preconditions: _inverted_index and _documents have been populated from the corpus.
    # returns: list of document names containing relevant search results.
    # parameters: text - a string of terms.

    def boolean_search(self, text):
        results = []
        # PUT YOUR CODE HERE
        individual_terms = self.tokenize(text)
        individual_terms = self.stemming(individual_terms)

        if len(individual_terms) == 1:
            #single term search
            individual_term = individual_terms[0]
            docIndexes = []
            if individual_term in self._inverted_index:
                docIndexes = self._inverted_index[individual_term]
            for index in docIndexes:
                results.append(self._documents[index])
        elif len(individual_terms) == 3 and (individual_terms[1] == 'or' or individual_terms[1] == 'and'):
            docIndexes_final = []
             # OR or AND search
            left_term, operator, right_term = individual_terms[0], individual_terms[1], individual_terms[2]
            left_results = []
            right_results = []
            if left_term in self._inverted_index:
                left_results = self._inverted_index[left_term]
            if right_term in self._inverted_index:
                right_results = self._inverted_index[right_term]

            operator = individual_terms[1]
            if operator.upper() == 'OR':
            #union
                docIndexes_final = left_results + right_results
                docIndexes_final = list(set(docIndexes_final))
                docIndexes_final = sorted(docIndexes_final)
            else:
            #intersection
                docIndexes_final = list(set(left_results).intersection(right_results))

            for index in docIndexes_final:
                results.append(self._documents[index])

        return results


# now, we'll define our main function which actually starts the indexer and
# does a few queries
def main(args):
    print(student)
    index = Index()
    print("***STARTING INDEXER***")
    num_files = index.index_dir("C:/Users/adhir/Documents/CS547/Assignment1/data/")
    print("indexed %d files" % num_files)
    for term in ('football', 'mike', 'sherman', 'mike OR sherman', 'mike AND sherman'):
        results = index.boolean_search(term)
        print("searching: %s -- results: %s" % (term, ", ".join(results)))
    print("-------------------------")

# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)