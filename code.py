from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
from sklearn.utils.extmath import randomized_svd

# Other imports
from collections import Counter
import re

# type: ignore 

def get_relevant_words(words, stop_words):
    return([word for word in words if re.match("[a-zA-Z#]", word) and word not in stop_words])

def top_k_unigrams(tweets, stop_words, k):
    unigrams = Counter()
    cleaned_tweets = [get_relevant_words(tweet.split(), stop_words) for tweet in tweets]
    
    for tweet in cleaned_tweets:
        unigrams.update(Counter(tweet))
    
    if(k>0):
        return dict(unigrams.most_common(k))
    return dict(unigrams.most_common())

def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
    d = Counter()
    for tweet in tweets:
        all_words = tweet.split()
        valid_words = list(set(get_relevant_words(all_words, stop_words)).intersection(frequent_unigrams))
        for idx, word1 in enumerate(all_words):
            surrounding_words = [all_words[i] for i in range(max(0,idx-context_size), min(idx+context_size+1, len(all_words)))]
            surrounding_words.remove(word1) # Even if this removes the wrong occurrence, the result will be identical
            for word2 in surrounding_words:
                if(word2 not in valid_words):
                    continue
                d.update({(word1,word2):1})
    return d

# Rounding issues exist
def pmi(word1, word2, unigram_counter, context_counter):
    s = sum(unigram_counter.values())
    freq_xy = context_counter.get((word1,word2), 1)
    freq_x = unigram_counter.get(word1, 1)
    freq_y = unigram_counter.get(word2, 1)
    return( np.log((freq_xy/s) / ((freq_x/s) * (freq_y/s))) / np.log(2))
    
def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    d = dict([])
    for word2 in frequent_unigrams:
        if((word1, word2) not in context_counter):
            d.update({word2:0.0})
        else:
            d.update({word2:pmi(word1, word2, unigram_counter, context_counter)})
    return(d)

def get_top_k_dimensions(word1_vector, k):
    c = Counter(word1_vector)
    return dict(c.most_common(k))

def get_cosine_similarity(word1_vector, word2_vector):
    dot_product = 0
    sumOfSquares1 = 0
    sumOfSquares2 = 0
    for key in word1_vector.keys():
        dot_product += (word1_vector[key]*word2_vector[key])
        sumOfSquares1 += word1_vector[key] ** 2
        sumOfSquares2 += word2_vector[key] ** 2
    return( dot_product / ( np.sqrt(sumOfSquares1) * np.sqrt(sumOfSquares2) ) )

def get_most_similar(word2vec, word, k):
    if word2vec.__contains__(word):
        return(word2vec.similar_by_key(word, k))
    return []

def word_analogy(word2vec, word1, word2, word3):
    return(word2vec.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0])

def cos_sim(A, B):
    return( np.dot(A,B) / (np.sqrt(A.dot(A)) * np.sqrt(B.dot(B))) )

def get_cos_sim_different_models(word, model1, model2, cos_sim_function):
    return(cos_sim_function(model1.wv[word], model2.wv[word]))

def get_average_cos_sim(word, neighbors, model, cos_sim_function):
    cos_sims = []
    v1 = model.wv[word]
    for neighbor in neighbors:
        if neighbor in model.wv:
            cos_sims.append(cos_sim_function(v1, model.wv[neighbor]))
    return(sum(cos_sims)/len(cos_sims))

def process_document(document, stopwords):
    lowercase_doc = [word.lower() for word in document]
    culled_doc = [word for word in lowercase_doc if word not in stopwords]
    processed_doc = [word for word in culled_doc if word.isalnum()]
    return(processed_doc)

def create_tfidf_matrix(documents, stopwords):
    processed_docs = [process_document(document, stopwords) for document in documents]
    vocab = []
    for doc in processed_docs:
        for word in doc:
            if word not in vocab:
                vocab.append(word)
    vocab.sort()

    num_docs = len(processed_docs)
    num_words = len(vocab)
    nt_array = np.zeros(num_words)
    tf_matrix = np.zeros((num_docs, num_words))

    # Calculate tf and nt values
    for i, doc in enumerate(processed_docs):
        for j, word in enumerate(vocab):
            tf_matrix[i][j] = doc.count(word)
            if tf_matrix[i][j] != 0:
                nt_array[j] += 1

    # Calculate tf_idf values
    for i in range(num_docs):
        for j in range(num_words):
            tf_matrix[i][j] *= (np.log10(num_docs/(nt_array[j]+1))+1)

    return(tf_matrix, vocab)

def get_idf_values(documents, stopwords):
    processed_docs = [process_document(document, stopwords) for document in documents]
    vocab = []
    for doc in processed_docs:
        for word in doc:
            if word not in vocab:
                vocab.append(word)
    vocab.sort()

    num_docs = len(processed_docs)
    num_words = len(vocab)
    nt_array = np.zeros(num_words)
    idf_dict = dict([])

    # Calculate nt values
    for i, doc in enumerate(processed_docs):
        for j, word in enumerate(vocab):
            if word in doc:
                nt_array[j]+=1

    # Calculate idf values
    for i, word in enumerate(vocab):
        idf_dict.update({word:(np.log10(num_docs/(nt_array[i]+1))+1)})

    return(idf_dict)

def calculate_sparsity(tfidf_matrix):
    zeros = 0
    for i in range(tfidf_matrix.shape[0]):
        for j in range(tfidf_matrix.shape[1]):
            if not tfidf_matrix[i][j]:
                zeros+=1
    return(zeros / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))

def extract_salient_words(VT, vocabulary, k):
    salient_words = dict([])
    for i in range (0, VT.shape[0]):
        indices = np.argpartition(VT[i], -k)[-k:]
        topk = dict([(vocabulary[index], VT[i][index]) for index in indices])
        sortedk = dict(sorted(topk.items(), key=lambda item: item[1]))
        salient_words.update({i:list(sortedk.keys())})
    return(salient_words)

def get_similar_documents(U, Sigma, VT, doc_index, k):
    doc_vectors = np.ndarray(U.shape)
    for i, row in enumerate(U):
        doc_vectors[i] = [row[index] * Sigma[index] for index in range(len(row))]

    cos_sims = np.array([cos_sim(doc_vectors[doc_index], doc_vector) for doc_vector in doc_vectors])
    cos_sims[doc_index] = -1
    indices = np.argpartition(cos_sims, -k)[-k:]
    topk = dict([(index, cos_sims[index]) for index in indices])
    topkdict = dict(sorted(topk.items(), key=lambda item: item[1]))
    sortedk = list(topkdict.keys())
    sortedk.reverse()
    return(sortedk)

def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):
    qv = np.zeros(len(vocabulary))
    words = list(set(query).intersection(vocabulary))
    for word in words:
        qv[vocabulary.index(word)] = query.count(word) / idf_values[word]

    qv_embedding = np.zeros(len(VT))
    V = VT.transpose()
    qv_embedding = qv.dot(V)
    
    cos_sims = np.array([cos_sim(qv_embedding, doc_vector * Sigma) for doc_vector in U])
    indices = np.argpartition(cos_sims, -k)[-k:]
    topk = dict([(index, cos_sims[index]) for index in indices])
    topkdict = dict(sorted(topk.items(), key=lambda item: item[1]))
    sortedk = list(topkdict.keys())
    sortedk.reverse()
    return(sortedk)

if __name__ == '__main__':
    
    tweets = []
    with open('data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt', encoding="utf-8") as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open('data/stop_words.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]


    """Building Vector Space model using PMI"""

    print(top_k_unigrams(tweets, stop_words, 10))
    # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    
    ### THIS PART IS JUST TO PROVIDE A REFERENCE OUTPUT
    sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
    print(sample_output.most_common(10))
    """
    [(('the', 'pandemic'), 19811),
    (('a', 'pandemic'), 16615),
    (('a', 'mask'), 14353),
    (('a', 'wear'), 11017),
    (('wear', 'mask'), 10628),
    (('mask', 'wear'), 10628),
    (('do', 'n’t'), 10237),
    (('during', 'pandemic'), 8127),
    (('the', 'covid'), 7630),
    (('to', 'go'), 7527)]
    """
    ### END OF REFERENCE OUTPUT
    
    context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)

    word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    print(get_top_k_dimensions(word_vector, 10))
    # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.2341567704935342

    word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.05127326904936171

    word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.7052644362543867

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.6144272810573133

    word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.37083874436657593

    word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    print(get_cosine_similarity(word1_vector, word2_vector))
    # 0.34568665086152817


    """Exploring Word2Vec"""

    EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    print(similar_words)
    # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # Word analogy - Tokyo is to Japan as Paris is to what?
    print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # ('France', 0.7889978885650635)


    """Word2Vec for Meaning Change"""

    # Comparing 40-60 year olds in the 1910s and 40-60 year olds in the 2000s
    model_t1 = Word2Vec.load('data/1910s_50yos.model')
    model_t2 = Word2Vec.load('data/2000s_50yos.model')

    # Cosine similarity function for vector inputs
    vector_1 = np.array([1,2,3,4])
    vector_2 = np.array([3,5,4,2])
    cos_similarity = cos_sim(vector_1, vector_2)
    print(cos_similarity)
    # 0.8198915917499229

    # Similarity between embeddings of the same word from different times
    major_cos_similarity = get_cos_sim_different_models("major", model_t1, model_t2, cos_sim)
    print(major_cos_similarity)
    # 0.19302374124526978

    # Average cosine similarity to neighborhood of words
    neighbors_old = ['brigadier', 'colonel', 'lieutenant', 'brevet', 'outrank']
    neighbors_new = ['significant', 'key', 'big', 'biggest', 'huge']
    print(get_average_cos_sim("major", neighbors_old, model_t1, cos_sim))
    # 0.6957747220993042
    print(get_average_cos_sim("major", neighbors_new, model_t1, cos_sim))
    # 0.27042335271835327
    print(get_average_cos_sim("major", neighbors_old, model_t2, cos_sim))
    # 0.2626224756240845
    print(get_average_cos_sim("major", neighbors_new, model_t2, cos_sim))
    # 0.6279034614562988

    ### The takeaway -- When comparing word embeddings from 40-60 year olds in the 1910s and 2000s,
    ###                 (i) cosine similarity to the neighborhood of words related to military ranks goes down;
    ###                 (ii) cosine similarity to the neighborhood of words related to significance goes up.


    """Latent Semantic Analysis"""

    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    # tfidf_matrix = np.load("tfidf_matrix.npy")
    # vocabulary = list(np.load("vocabulary.npy"))
    # idf_values = dict(np.load("idf_values.npy", allow_pickle=True).item())

    # np.save("tfidf_matrix.npy", tfidf_matrix)
    # np.save("vocabulary.npy", vocabulary)
    # np.save("idf_values.npy", idf_values)

    print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    print(relevant_doc_indices)
    # for i in range(2):
    #     print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
