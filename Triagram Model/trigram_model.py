import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

"""
Name Surname: Taner Giray Sonmez
UNI:tgs2126
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """   
    ngram_list=[]
    if n-1==0:
        sequence.insert(0,'START')
        sequence.append('STOP') 
    else:    
        for _ in range(n-1):
            sequence.insert(0,'START')
        sequence.append('STOP')    
    for idx in range(len(sequence)):
        ngram_tuple_list=[]
        for i in range(idx,idx+n):
            ngram_tuple_list.append(sequence[i])
        ngram_tuple_list=tuple(ngram_tuple_list)    
        ngram_list.append(ngram_tuple_list)    
        if ngram_list[-1][0] == 'STOP' or ngram_list[-1][-1] == 'STOP':
            break
    return ngram_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = defaultdict(int)  
        self.bigramcounts = defaultdict(int)   
        self.trigramcounts = defaultdict(int)  
        self.totalWords = 0 

        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)       
            for u in unigrams:
                self.unigramcounts[u] += 1
                self.totalWords += 1       
            for b in bigrams:
                self.bigramcounts[b] += 1              
            for t in trigrams:
                self.trigramcounts[t] += 1


    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        bigram = (trigram[0], trigram[1])
        if trigram in self.trigramcounts:
            trigramCount = self.trigramcounts[trigram]
        else:
            trigramCount = 0
        if bigram in self.bigramcounts:
            bigramCount = self.bigramcounts[bigram]
        else:
            bigramCount = 0
        if bigramCount > 0:
            result = trigramCount / bigramCount
        else:
            result = 1 / len(self.lexicon)

        return result

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        unigram = (bigram[0],)
        if bigram in self.bigramcounts:
            bigramCount = self.bigramcounts[bigram]
        else:
            bigramCount = 0
        if unigram in self.unigramcounts:
            unigramCount = self.unigramcounts[unigram]
        else:
            unigramCount = 0          
        if unigramCount > 0:
            result = bigramCount / unigramCount
        else:
            result = 0

        return result
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it. 
         
        if unigram in self.unigramcounts:
            unigramCount = self.unigramcounts[unigram]
        else:
            unigramCount = 0        
        if self.totalWords > 0:
            result = unigramCount / self.totalWords
        else:
            result = 0
        return result

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        #return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigramProb = self.raw_trigram_probability(trigram)
        bigramProb = self.raw_bigram_probability(trigram[1:])
        unigramProb = self.raw_unigram_probability(trigram[2:])
        
        return (lambda1 * trigramProb) + (lambda2 * bigramProb) + (lambda3 * unigramProb)

        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        log = 0.0     
        trigrams = get_ngrams(sentence, 3) 
        for i in trigrams:
            prob = self.smoothed_trigram_probability(i)
            if prob !=0:
                log += math.log2(prob)       
        return log


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        words = 0
        logprob = 0.0       
        for sentence in corpus:
            logprob += self.sentence_logprob(sentence)   
            words += len(sentence)
        l = logprob / words
        return 2 ** (-1*l)

def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))      
            if pp1 < pp2:
                correct += 1
            total += 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp2 < pp1:
                correct += 1
            total += 1

        acc= correct/total    
        return acc

if __name__ == "__main__":

    model = TrigramModel("hw1_data/brown_train.txt") 
    #model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)

    dev_corpus = corpus_reader("hw1_data/brown_test.txt", model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Perplexity test ",pp)

    dev_corpus = corpus_reader("hw1_data/brown_train.txt", model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("Perplexity train ",pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment("hw1_data/ets_toefl_data/train_high.txt", "hw1_data/ets_toefl_data/train_low.txt", "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print(acc)

    print(model.trigramcounts[('START','START','the')])
    print(model.bigramcounts[('START','the')])
    print(model.unigramcounts[('the',)])


