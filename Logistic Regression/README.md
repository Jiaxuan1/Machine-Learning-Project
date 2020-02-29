
This is a binary logistic classifier to conduct a sentiment analysis for review text data.

feature.py: turn the review data into a sparse matrix with two different kinds of feature model. 
            1. if the word exists, the value of it will be 1
            2. if the word exists and the frequency is less than 4, the value of it will be 1.
            
lr.py: using the sparse matrix to conduct SGD to optimize the parameter and predict the positive/negative label
