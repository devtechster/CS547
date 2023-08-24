# homework 4
# goal: ranked retrieval, PageRank, crawling
# exports:
#   student - a populated and instantiated cs547.Student object
#   PageRankIndex - a class which encapsulates the necessary logic for
#     indexing and searching a corpus of text documents and providing a
#     ranked result set

# ########################################
# first, create a student object
# ########################################

import cs547
MY_NAME = "Adhiraj Budukh"
MY_ANUM  = 901004534 # put your UID here
MY_EMAIL = "abudukh@wpi.edu"

# the COLLABORATORS list contains tuples of 2 items, the name of the helper
# and their contribution to your homework
COLLABORATORS = [ ('Di You', 'helped me learn Python')]

# Set the I_AGREE_HONOR_CODE to True if you agree with the following statement
# "An Aggie does not lie, cheat or steal, or tolerate those who do."
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

import bs4 as BeautifulSoup  # you will want this for parsing html documents
import urllib.request as request 
from  urllib.parse import urljoin
import re
import numpy as np

# our index class definition will hold all logic necessary to create and search
# an index created from a web directory
#
# NOTE - if you would like to subclass your original Index class from homework
# 1 or 2, feel free, but it's not required.  The grading criteria will be to
# call the index_url(...) and ranked_search(...) functions and to examine their
# output.  The index_url(...) function will also be examined to ensure you are
# building the index sanely.

class PageRankIndex(object):
    def __init__(self):
        # you'll want to create something here to hold your index, and other
        # necessary data members
        self._webgraph_root_node = {}
        self._inverted_index={}
        self._url=[]
        self._pageRank = []
        
    def evaluate_pageRank(self):
        teleportation_factor = 0.1
        n = len(self._url)
        t = 1/n
        teleport_matrix = np.full((n,n),t)
        transition = np.zeros(teleport_matrix.shape)
        first_vector = np.full((1,n),t)
        for link in self._webgraph_root_node:
            i = self._url.index(link)
            for sublink in self._webgraph_root_node[link]:
                j = self._url.index(sublink)
                transition[i,j]=1
            transition[i,:] = transition[i,:]/np.sum(transition[i,:])
        P = teleportation_factor*teleport_matrix + (1-teleportation_factor)*transition
        relative_tolerance = 1e-08
        i=0
        while(1):
            second_vector = np.matmul(first_vector,P)
            i+=1
            if np.allclose(first_vector, second_vector, relative_tolerance):
                break
            else:
                first_vector = second_vector.copy()
        self._pageRank = first_vector.copy()


#--------------------------------------------------------------------------------------------------------------------------------#
    # index_url( url )
    # purpose: crawl through a web directory of html files and generate an
    #   index of the contents
    # preconditions: none
    # returns: num of documents indexed
    # hint: use BeautifulSoup and urllib
    # parameters:
    #   url - a string containing a url to begin indexing at
    def index_url(self, url):
        # ADD CODE HERE
        url1_page = request.urlopen(url)
        url1_parser = BeautifulSoup.BeautifulSoup(url1_page, 'html.parser')
        tokens = []
        for href_link in url1_parser.find_all('a'):
            link = urljoin(url,href_link.get('href'))
            if link not in self._webgraph_root_node:
                self._url.append(link)
                self._webgraph_root_node[link]=[]
                url2_page = request.urlopen(link)

                url2_parser = BeautifulSoup.BeautifulSoup(url2_page, 'html.parser')
                for href_sublink in url2_parser.find_all('a'):
                    sublink = urljoin(url,href_sublink.get('href'))
                    if sublink not in self._webgraph_root_node[link]:
                        self._webgraph_root_node[link].append(sublink)

                text = url2_parser.get_text()
                doc_Tokens = self.tokenize(text)
                tokens.append(doc_Tokens)
                for word in doc_Tokens:
                    if word not in self._inverted_index:
                        self._inverted_index[word] = []
                    if link not in self._inverted_index[word]:
                        self._inverted_index[word].append(link)
        self.evaluate_pageRank()
        return len(self._url)
#--------------------------------------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------------------------------------#
    # tokenize( text )
    # purpose: convert a string of terms into a list of terms 
    # preconditions: none
    # returns: list of terms contained within the text
    # parameters:
    #   text - a string of terms
    def tokenize(self, text):
        # ADD CODE HERE
        input_tokens = []
        text = text.lower()
        text = re.sub('[^0-9a-zA-Z]', ' ', text)
        input_tokens = text.split()
        return input_tokens
#--------------------------------------------------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------------------------------------------------#
    # ranked_search( text )
    # purpose: searches for the terms in "text" in our index and returns
    #   AND results for highest 10 ranked results
    # preconditions: .index_url(...) has been called on our corpus
    # returns: list of tuples of (url,PageRank) containing relevant
    #   search results
    # parameters:
    #   text - a string of query terms
    def ranked_search(self, text):
        # ADD CODE HERE
        input_tokens = self.tokenize(text)
        containing_pages = []
        links_list=[]

        for i,t in enumerate(input_tokens):
            if t not in self._inverted_index:
                return ['NA']
            else:
                if i==0:
                    links_list = self._inverted_index[t]
                links_list = list(set(self._inverted_index[t]) & set(links_list))
        if len(links_list)==0:
            return ['NA']
        for l in links_list:
            index = self._url.index(l)
            rank = self._pageRank[0][index]
            containing_pages.append((l,rank))
        containing_pages.sort(key = lambda x: x[1],reverse=True)
        if len(containing_pages)>10:
            containing_pages = containing_pages[:10]
        return containing_pages
#--------------------------------------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------------------------------------#
# now, we'll define our main function which actually starts the indexer and
# does a few queries
def main(args):
    print(student)
    index = PageRankIndex()
    url = 'http://web.cs.wpi.edu/~kmlee/cs547/new10/index.html'
    num_files = index.index_url(url)
    search_queries = (
       'palatial', 'college ', 'palatial college', 'college supermarket', 'famous aggie supermarket'
        )
    for q in search_queries:
        results = index.ranked_search(q)
        print("searching: %s -- results: %s" % (q, results))


# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)
#--------------------------------------------------------------------------------------------------------------------------------#

