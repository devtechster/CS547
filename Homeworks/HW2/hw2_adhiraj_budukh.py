# homework 2
# goal: wildcard query
# exports: 
#   student - a populated and instantiated cs547.Student object
#   BetterIndex - a class which encapsulates the necessary logic for
#     indexing and searching a corpus of text documents

# we'll want this in our index
import binarytree
from functools import reduce


# ########################################
# first, create a student object
# ########################################

import cs547
MY_NAME = "Adhiraj N Budukh"
MY_ANUM  = 901004534 # put your UID here
MY_EMAIL = "abudukh@wpi.edu"

# the COLLABORATORS list contains tuples of 2 items, the name of the helper
# and their contribution to your homework
COLLABORATORS = [ 
    ('Di You', 'helped me learn Python'),
    ]

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


# crawl_tree( node, term )
# purpose: crawl a binary tree looking for permuterm matches
# hint: node.key.startswith(term)
# preconditions: none
# returns: list of documents contained matching the term
# parameters: a node within a binary tree and a term as a string
def crawl_tree(node, term):
    if not node: return set()
    if ('*' in term and node.key.startswith(term[:-1])) or term == node.key:
        x = node.data
    else: x = set()
    return x.union(crawl_tree(node.left, term)).union(crawl_tree(node.right, term))
    #if term <= node.key[:len(term)]:
    #    x = x.union(crawl_tree(node.left, term))
    #else:
    #    x = x.union(crawl_tree(node.right, term))
    #return x  

# our index class definition will hold all logic necessary to create and search
# an index created from a directory of text files 
#
# NOTE - if you would like to subclass your original Index class from homework
# 1, feel free, but it's not required.  The grading criteria will be to call
# the index_dir(...), wildcard_search_or(...), and wildcard_search_and(...)
# functions and to examine their output.  The index_dir(...) function will also
# be examined to ensure you are building the permuterm index appropriately.
#
class BetterIndex(object):
    WILDCARD = '*'
    def __init__(self):
        # _inverted_index contains terms as keys, with the values as a list of
        # document indexes containing that term
        self._bt = binarytree.binary_tree()
        # _documents contains file names of documents
        self._documents = []

    # _permute( term )
    # purpose: generate and return a list of permutations for the given term
    # preconditions: none
    # returns: list of permutations for the given term
    # parameters: a single term as a string 
    def _permute(self, term):
        x = term + "$"
        return [x[i:] + x[:i] for i in range(len(x))]

    # _rotate( term )
    # purpose: rotate a wildcard term to generate a search token
    # preconditions: none
    # returns: string containing a search token
    # parameters: a single term as a string
    def _rotate(self, term):
        x = term + "$"
        if self.WILDCARD not in term: return x
        n = x.index(self.WILDCARD) + 1
        return (x[n:] + x[:n])

    # index_dir( base_path )
    # purpose: crawl through a directory of text files and generate a
    #   permuterm index of the contents
    # preconditions: none
    # returns: num of documents indexed
    # hint: glob.glob()
    # parameters:
    #   base_path - a string containing a relative or direct path to a
    #     directory of text files to be indexed
    def index_dir(self, base_path):
        num_files_indexed = 0
        from glob import glob
        for fn in glob("%s/*" % base_path):
            num_files_indexed += 1
            for line in open(fn,encoding="utf8"):
                if fn not in self._documents:
                    self._documents.append(fn)
                doc_idx = self._documents.index(fn)
                for t in self.tokenize(line):
                    for term in self._permute(t):
                        if term not in self._bt:
                            self._bt[term] = set()
                        if doc_idx not in self._bt[term]:
                            self._bt[term].add(doc_idx)
        return num_files_indexed


    # tokenize( text )
    # purpose: convert a string of terms into a list of terms 
    # preconditions: none
    # returns: list of terms contained within the text
    # parameters:
    #   text - a string of terms
    #   is_search - boolean which determines whether we strip out the wildcard
    #     character or not
    def tokenize(self, text, is_search=False):
        import re
        if is_search:
            # don't strip out our wildcard character from query terms
            clean_string = re.sub('[^a-z0-9 *]', ' ', text.lower())
        else:
            clean_string = re.sub('[^a-z0-9 ]', ' ', text.lower())
        tokens = clean_string.split()
        return tokens


    # wildcard_search_or( text )
    # purpose: searches for the terms in "text" in our index
    # preconditions: _bt and _documents have been populated from
    #   the corpus.
    # returns: list of document names containing relevant search results
    # parameters:
    #   text - a string of terms
    def wildcard_search_or(self, text):
        terms = self.tokenize(text,True)        
        index_of_documents = []
        
        for single_term in terms:
            permuterms = self._rotate(single_term)
            index_of_documents = list(set(index_of_documents + list(crawl_tree(self._bt.root,permuterms))))
            
        output_documents=[self._documents[i] for i in index_of_documents]
        #if not found anything
        if len(output_documents)==0:
            output_documents.append('N/A')
        return output_documents    
    
    # wildcard_search_and( text )
    # purpose: searches for the terms in "text" in our corpus
    # preconditions: _bt and _documents have been populated from
    #   the corpus.
    # returns: list of file names containing relevant search results
    # parameters:
    #   text - a string of terms

    def wildcard_search_and(self, text):
        # FILL OUT CODE HERE
        terms = self.tokenize(text,True)
        index_of_documents = []
        
        for iterator,single_term in enumerate(terms):
            permuterms = self._rotate(single_term)
            if iterator == 0:
                index_of_documents = crawl_tree(self._bt.root,permuterms)
            index_of_documents = list(crawl_tree(self._bt.root,permuterms) & set(index_of_documents))
        
        output_documents=[self._documents[iterator] for iterator in index_of_documents]
        #if not found anything
        if len(output_documents) == 0:
            output_documents.append('N/A')
        return output_documents    

# now, we'll define our main function which actually starts the indexer and
# does a few queries
def main(args):
    print(student)
    index = BetterIndex()
    print("starting indexer")
    num_files = index.index_dir('data/')
    #print (index._bt.formattree())
    print("indexed %d files" % num_files)
    
    for term in ('hel*o', 'aggies', 'agg*', 'mike sherm*', 'dot cat'):
        results = index.wildcard_search_or(term)
        print("OR  searching: %s -- results: %s" % (term, ", ".join(results)))
        results = index.wildcard_search_and(term)
        print("AND searching: %s -- results: %s" % (term, ", ".join(results)))


# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)

