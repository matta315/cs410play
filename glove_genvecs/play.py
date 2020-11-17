from pymagnitude import Magnitude, FeaturizerMagnitude, MagnitudeUtils

word2vec = Magnitude("./vectors.magnitude")
pos_vectors = FeaturizerMagnitude(100, namespace="PartsOfSpeech")
dependency_vectors = FeaturizerMagnitude(100, namespace = "SyntaxDependencies")

print(len(word2vec))
print(word2vec.dim)

print("dog" in word2vec)
print("cat" in word2vec)
print("me" in word2vec)

#for key, vec in vts:
#    print(key, "=", vec)

#key = "Battle"
#print(key, "=", vts.query(key))

#sentence = "leadership matters"
#print(sentence, '=', word2vec.query(' '.split(sentence)))
#sentence = "matters leadership"
#print(sentence, '=', word2vec.query(' '.split(sentence)))

#print(word2vec.distance("Democrats", ["Trump", "leadership", "chairman"]))
#print(word2vec.similarity("Democrats", ["Trump", "leadership", "chairman"]))

#qq = word2vec.query("dog loves")
sent1 = "I always walk my dog"
#qq1 = word2vec.query(sent1)
qq1 = word2vec.most_similar(sent1, topn=5)
print(sum([f[1] for f in qq1]))
#print(word2vec.distance(sent1, qq1))
#temp = word2vec.most_similar(qq1, topn=6)
#print(sum([f[1] for f in temp]))

sent2 = "Trump lost the election"
#qq2 = word2vec.query(sent2)
qq2 = word2vec.most_similar(sent2, topn=5)
print(sum([f[1] for f in qq2]))
#print(word2vec.distance(sent2, qq2))
#temp = word2vec.most_similar(qq2, topn=6)
#print(sum([f[1] for f in temp]))
#mostsim = word2vec.most_similar(qq, topn=1)
#print('mostsim =', mostsim)
#print(word2vec.similarity("I ate a dog", "I am walking a cat"))


"""
# concatenate word2vec with pos and dependencies
vts = Magnitude(word2vec, pos_vectors, dependency_vectors)
# array of size 5 x (300 + 4 + 4) or 5 x 308
res = vts.query([
    ("I", "PRP", "nsubj"),
    ("saw", "VBD", "ROOT"),
    ("a", "DT", "det"),
    ("cat", "NN", "dobj"),
    (".",  ".", "punct")
  ])
print('res =', res)
"""
