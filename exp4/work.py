import gensim
word_embedding=gensim.models.KeyedVectors.load_word2vec_format("dataset/WikiWord/wiki_word2vec_50.bin",binary=True)
print(word_embedding.vocabulary)