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

# ########################################
# now, write some code
# ########################################

# our index class definition will hold all logic necessary to create and search
# an index created from a directory of text files 
class Index(object):
    def _init_(self):
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

    # index_dir( base_path )
    # purpose: crawl through a nested directory of text files and generate an
    #   inverted index of the contents
    # preconditions: none
    # returns: num of documents indexed
    # hint: glob.glob()
    # parameters:
    #   base_path - a string containing a relative or direct path to a
    #     directory of text files to be indexed

    def index_dir(self, base_path):
        num_files_indexed = 0
        # PUT YOUR CODE HERE

        import glob
        import os
        for path in glob.glob(base_path + "/*"):
            self._documents.append(os.path.basename(path))

            with open(base_path + os.path.basename(path),'r',encoding = "utf-8") as file:

                for line in file:
                    tks= self.tokenize(line)
                    stks = self.stemming(tks)

                    for word in stks:

                        if (word in self._inverted_index.keys()):
                            if num_files_indexed not in self._inverted_index[word]:
                                self._inverted_index[word].append(num_files_indexed)
                        else:
                            self._inverted_index[word] = [num_files_indexed]

                num_files_indexed +=1

        return num_files_indexed

    # tokenize( text )
    # purpose: convert a string of terms into a list of tokens.        
    # convert the string of terms in text to lower case and replace each character in text, 
    # which is not an English alphabet (a-z) and a numerical digit (0-9), with whitespace.
    # preconditions: none
    # returns: list of tokens contained within the text
    # parameters:
    #   text - a string of terms
    def tokenize(self, text):
        tokens = []
        # PUT YOUR CODE HERE
        import re
        text = text.lower()
        text = re.sub("[^0-9a-zA-Z]+", " ", text)
        tokens = text.split(" ")
        return tokens

    # purpose: convert a string of terms into a list of tokens.        
    # convert a list of tokens to a list of stemmed tokens,     
    # preconditions: tokenize a string of terms
    # returns: list of stemmed tokens
    # parameters:
    #   tokens - a list of tokens
    def stemming(self, tokens):
        stemmed_tokens = []
        # PUT YOUR CODE HERE
        for token in tokens:
            stemmed_token = PorterStemmer.PorterStemmer().stem(token,0,len(token)-1)
            stemmed_tokens.append(stemmed_token)
        return stemmed_tokens
    
    # boolean_search( text )
    # purpose: searches for the terms in "text" in our corpus using logical OR or logical AND. 
    # If "text" contains only single term, search it from the inverted index. If "text" contains three terms including "or" or "and", 
    # do OR or AND search depending on the second term ("or" or "and") in the "text".  
    # preconditions: _inverted_index and _documents have been populated from
    #   the corpus.
    # returns: list of document names containing relevant search results
    # parameters:
    #   text - a string of terms
    def boolean_search(self, text):
        results = []
        # PUT YOUR CODE HERE
        tks = self.tokenize(text)
        stks = self.stemming(tks)

        r1= []
        r2= []

        word = stks.pop(0)
        for item in self._inverted_index[word]:
            r1.append(self._documents[item])

        if len(stks) ==0:
            results = r1

        else:
            operator = stks.pop(0)
            word = stks.pop(0)
            for item in self._inverted_index[word]:
                r2.append(self._documents[item])

            if operator == 'or':
                results = sorted(list(set(r1+r2)))

            else: 
                results = sorted(list(set(r1).intersection(r2)))

        return results


# now, we'll define our main function which actually starts the indexer and
# does a few queries
def main(args):
    print(student)
    index = Index()
    print("starting indexer")
    num_files = index.index_dir('C:\Users\adhir\Documents\CS547\Assignment1\data')
    print("indexed %d files" % num_files)
    for term in ('football', 'mike', 'sherman', 'mike OR sherman', 'mike AND sherman'):
        results = index.boolean_search(term)
        print("searching: %s -- results: %s" % (term, ", ".join(results)))

# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)