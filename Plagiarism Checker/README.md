
# Plagarism-Checker Python

## You can try app from this link:
  https://share.streamlit.io/fatihcdeniz/plagarism-checker/main/plagiarism.py

## Problem Definition:

Plagiarism is using some other person's ideas and information without acknowledging that person in the source. One of the ways to avoid plagiarism is paraphrasing, creating the same information with your own words, it is possible not to paraphrase properly, and this could create problems. This uses different distance measures and text embedding techniques to calculate the similarity between multiple texts.

## What Algorithms and Techniques are used:

This program can use five different similarity distance:
- Cosine Similarity
- Euclidean Distance
- Jaccard Index
- Longest Common Substring
- Hamming Distance 

And six different Text Embedding Techniques:
- Count Vectorizer
- TF-IDF
- Word2Vec
- Roberta
- Bert
- Universal Sentence Encoder

### Similarity Distance Techniques:

- Longest Common Substring: Longest Common Substring is a character-based similarity measure. Given two it simply finds the longest string which is the substring of the given two strings. If string1 is "Correlation" and string2 is "Correspondence" their value for Longest Common Substring would be "Corr".

- Hamming Distance: Hamming Distance is a character-based similarity measure. It measures the minimum number of changes needed to make for changing one string into the other. For instance, if our first string is "Cargo" and the second string is "Carry", we need to make three changes to "Carry" -> "Cargy" -> "Cargo" to make it "Cargo". 

- Jaccard Index: Jaccard Index measure how different two sets are. It is defined as the size of internsection of two sets divided by the size of the union.
  
  ![image](https://user-images.githubusercontent.com/96383593/163215808-828b2197-0c4b-453a-87a3-57c85bbe8d14.png)
  
  Mathematically, it can be written as:
  
  ![image](https://user-images.githubusercontent.com/96383593/163215735-397db1e5-d8a7-4de1-8e0e-f007bd231cae.png)
  
- Euclidean Distance: Euclidean distance is a token-based similarity distance. It calculates distance between two points that are in the Euclidean space.
  
  ![image](https://user-images.githubusercontent.com/96383593/163216652-537b3b87-fb37-4c2f-b6cc-e1b126a32f7d.png)

- Cosine Similarity: Cosine similarity measures the cosine angle between two vectors. It determines whether these two vectors are pointing in the roughly same direction. It is calculated by getting the dot product of two vectors and dividing it by the product of their lengths. The values are always within the range of [ -1 , 1 ]. Two vectors with the same orientation have a value of 1, two orthogonal vectors have a value of 0 and two opposite vectors have a value of -1.

  ![image](https://user-images.githubusercontent.com/96383593/163218631-9ac448a4-044a-45cc-93fa-d6cf90e8386f.png)

### Text Embedding Techniques:
- Count Vectorizer: Count Vectorizer applies tokenization(giving and id to each word) and occurence counting. Count vectorizer tokenize each word in the sentence and create a sparse matrix based on the vocabulary, where 1 indicates that words is in that sentence and 0 indicates word is not on that sentence.

  ```python
  headlines = [
  'Investors unfazed by correction as crypto funds see $154 million inflows',
  'Bitcoin, Ethereum prices continue descent, but crypto funds see inflows']
  ```
  ![](https://github.com/FatihcDeniz/Plagarism-Checker/blob/main/Screen%20Shot%202022-04-13%20at%2020.07.46.png)

  These sentences turned into these sparse matrixes where 1 indicate word is in the sentence and 0 indicates word not is in the sentence.

- TF-IDF(Term Frequency Inverse Document Frequency): TF-IDF is a measure of originality of a word by comparing the number of times a word appears in a document with the number of documents in the corpus that contain the word. In this way we can understand how important is a word in a document.

  ![image](https://user-images.githubusercontent.com/96383593/163244155-bd5c6c58-8d9f-4867-9d6f-ee67a70d3ac0.png)

  Term Frequency(TF) is relative frequency of term *t* in the document *d*, the number of time that term *t*       occurs in document *d*. It is calculated by number of times a word *t* appears in document divided by total       number of words in the document *d*.
  
  ![image](https://user-images.githubusercontent.com/96383593/163246035-d84ae806-c121-46d1-8008-d66c2e1dc48b.png)
  
  Numerator represent raw count of a word *t* in the document and denominator represent total number of words in   the document *d*.
  
  Inverse Document Frequency(IDF) is a measure for identifying how much information the word provides.  

  ![image](https://user-images.githubusercontent.com/96383593/163246265-d927605d-a2d9-4f13-a61b-62a125429b91.png)

  N is the total number of documents in the corpus and n is number of document where the word *t* appears.
  
- Word2Vec: Word2vec is a group of algorithms that can be used to produce word embedings. Word2vec can utilize two model architecture to represent words as a vector:
  
  Continuous Bag of Words(CBOW): This architecture predicts the current word based on surrounded context words.     Context means a few words before and after the current(middle) word.
  
  Skip-Gram: This model uses current word to predict surrounding context words.
  
  Continuous Bag of Words uses context words to predict current word, on the other hand Skip-Gram uses current     word to predict context words.
  
  ![](https://www.researchgate.net/profile/Daniel-Braun-6/publication/326588219/figure/fig1/AS:652185784295425@1532504616288/Continuous-Bag-of-words-CBOW-CB-and-Skip-gram-SG-training-model-illustrations.png)


- Bidirectional Encoder Representations from Transformers(BERT): BERT uses Transformers, in this way it can process any given word in realation to all other words in a sentence rather than processing them one at a time. By looking surrounding words, BERT can understand full context of the word, in contrast to other methods such as Word2vec which only map single word to a vector.


- RoBERTa: RoBERTa is optimizer version of BERT. To improve the training procedure, RoBERTa removes the Next Sentence Prediction (NSP) task from BERT’s pre-training and introduces dynamic masking so that the masked token changes during the training epochs. In the BERT, masking is performed on the data preparation time. On the other hand, in RoBERTa, the masking is done during training.


- Universal Sentence Encoder(USE): Universal Sentence Encoder uses a ont-to-many multi-tasking learning framework to learn a universal sentence embedding by switching between these tasks. We use these multiple tasks and based on the mistakes in makes on those we update sentence embedding. The 6 tasks, skip-thoughts prediction of the next/previous sentence, neural machine translation, constituency parsing, and natural language inference, share the same sentence embedding.

  ![](https://amitness.com/images/use-overall-pipeline.png)



#### References:
- https://www.researchgate.net/figure/Continuous-Bag-of-words-CBOW-CB-and-Skip-gram-SG-training-model-illustrations_fig1_326588219
- https://amitness.com/2020/06/universal-sentence-encoder/
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- https://en.wikipedia.org/wiki/Jaccard_index
- https://en.wikipedia.org/wiki/Euclidean_distance
- https://www.tensorflow.org/tutorials/text/word2vec
- https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
- https://en.wikipedia.org/wiki/Cosine_similarity
- https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model
- https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8
