import jieba
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and tokenize the dataset
def load_data(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')
            texts.append(' '.join(jieba.lcut(text)))  # Tokenize Chinese text with jieba
            labels.append(int(label))
    return texts, labels

train_texts, train_labels = load_data('E:\in.txt')
dev_texts, dev_labels = load_data('E:\dev.txt')
test_texts, test_labels = load_data('E:\est.txt')

class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.num_words = 0

    def fit_on_texts(self, texts):
        for text in texts:
            for word in jieba.cut(text):
                if word not in self.word_index:
                    self.word_index[word] = self.num_words
                    self.index_word[self.num_words] = word
                    self.num_words += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in jieba.cut(text):
                if word in self.word_index:
                    sequence.append(self.word_index[word])
            sequences.append(sequence)
        return sequences
# Build the vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
dev_sequences = tokenizer.texts_to_sequences(dev_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to ensure equal length
maxlen = 100
train_data = pad_sequences(train_sequences, maxlen=maxlen)
dev_data = pad_sequences(dev_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

train_labels = np.array(train_labels)
dev_labels = np.array(dev_labels)
test_labels = np.array(test_labels)


def create_cnn_model(vocab_size, embedding_dim, maxlen):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=maxlen),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Assuming 4 classes
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

embedding_dim = 100
model = create_cnn_model(vocab_size, embedding_dim, maxlen)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_data, train_labels,
                    epochs=20,
                    batch_size=128,
                    validation_data=(dev_data, dev_labels),
                    callbacks=[early_stopping])
# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test Accuracy: {test_acc * 100:.2f}%')