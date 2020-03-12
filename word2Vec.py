from nltk.corpus import brown
from nltk.corpus import conll2000
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
import collections
from gensim.models import Word2Vec
import multiprocessing

# Getting some sentences to train
sentences = brown.sents()
EMBEDDING_DIM = 300

# Convert words to vector
w2v = Word2Vec(sentences, size=EMBEDDING_DIM, window=5, min_count=5, negative=15, iter=10,
               workers=multiprocessing.cpu_count())
word_vectors = w2v.wv

# Getting some random train and test data
train_words = conll2000.tagged_words("train.txt")
test_words = conll2000.tagged_words("test.txt")


# Getting tag words
def get_tag_vocabulary(tagged_words):
    tag_id_map = {}
    for item in tagged_words:
        tag = item[1]
        tag_id_map.setdefault(tag, len(tag_id_map))
    return tag_id_map


# Mapping words and tags to theie ids
word2id = {k: v.index for k, v in word_vectors.vocab.items()}
tag2id = get_tag_vocabulary(train_words)


# Add new word in the matrix
def add_new_word(new_word, new_vector, new_index, matrix, word_id_map):
    matrix = np.insert(matrix, [new_index], [new_vector], axis=0)
    word_id_map = {word: (index + 1) if index >= new_index else index
                   for word, index in word_id_map.items()}
    word_id_map[new_word] = new_index
    return matrix, word_id_map


# Handling words not in the vocabulary
UNK_INDEX = 0
UNK_TOKEN = "UNK"
embedding_matrix = word_vectors.vectors
unk_vector = embedding_matrix.mean(0)
embedding_matrix, word2id = add_new_word(UNK_TOKEN, unk_vector, UNK_INDEX, embedding_matrix, word2id)

HIDDEN_SIZE = 50
BATCH_SIZE = 128

# Mapping ids to words and tags to retrieve
id2word = sorted(word2id, key=word2id.get)
id2tag = sorted(tag2id, key=tag2id.get)

# Handling end of sequence in case of window
EOS_INDEX = 1
EOS_TOKEN = "EOS"
eos_vector = np.random.standard_normal(EMBEDDING_DIM)
embedding_matrix, word2id = add_new_word(EOS_TOKEN, eos_vector, EOS_INDEX, embedding_matrix, word2id)

# How many words to take from both side
CONTEXT_SIZE = 2


# Generating the train and test lists with the context
def get_window_int_data(tagged_words, word_id_map, tag_id_map):
    x, y = [], []
    unk_count = 0
    span = 2 * CONTEXT_SIZE + 1
    buffer = collections.deque(maxlen=span)
    padding = [(EOS_TOKEN, None)] * CONTEXT_SIZE
    buffer += padding + tagged_words[:CONTEXT_SIZE]

    for item in (tagged_words[CONTEXT_SIZE:] + padding):
        buffer.append(item)
        window_ids = np.array([word_id_map.get(word) if (word in word_id_map) else UNK_INDEX
                               for (word, _) in buffer])
        x.append(window_ids)
        middle_word, middle_tag = buffer[CONTEXT_SIZE]
        y.append(tag_id_map.get(middle_tag))
        if middle_word not in word_id_map:
            unk_count += 1
    print("Data created, percentage of unknown words: %.3f" % (unk_count / len(tagged_words)))
    return np.array(x), np.array(y)


# Model with context sensitivity
def define_context_sensitive_model(matrix, class_count):
    vocab_length = len(matrix)
    total_span = 2 * CONTEXT_SIZE + 1

    model = Sequential()
    model.add(
        Embedding(input_dim=vocab_length, output_dim=EMBEDDING_DIM, weights=[matrix],
                  input_length=total_span))
    model.add(Flatten())
    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation("tanh"))
    model.add(Dense(class_count))
    model.add(Activation("softmax"))

    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Evaluating the model with test data
def evaluate_model(model, id_word_map, x_test, y_test):
    _, acc = model.evaluate(x_test, y_test)
    print("%.2f" % acc)
    y_pred = model.predict_classes(x_test)
    error_counter = collections.Counter()

    for i in range(len(x_test)):
        correct_tag_id = np.argmax(y_test[i])
        if y_pred[i] != correct_tag_id:
            if isinstance(x_test[i], np.ndarray):
                word = id_word_map[x_test[i][CONTEXT_SIZE]]
            else:
                word = id_word_map[x_test[i]]
            error_counter[word] += 1
    print("Most common errors: \n", error_counter.most_common(10))


# Generating train and test data sets
X_train, Y_train = get_window_int_data(train_words, word2id, tag2id)
X_test, Y_test = get_window_int_data(test_words, word2id, tag2id)
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

# Generating, training and evaluating the model
cs_pos_model = define_context_sensitive_model(embedding_matrix, len(tag2id))
cs_pos_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=1, verbose=1)
evaluate_model(cs_pos_model, id2word, X_test, Y_test)
