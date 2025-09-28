import gensim.downloader as api
gt = api.load("glove-twitter-50")

# gives the similarity score for 2 words
gt.similarity(w1="great",w2="amazing")

# top 5 matching words to great
gt.most_similar("great",topn=5)

gt.most_similar("queen")

#outputs queen
gt.most_similar(positive=["king","woman"], negative=["man"])

# odd out amsterdam
gt.doesnt_match(["delhi","mumbai","amsterdam","pune","bhopal"])