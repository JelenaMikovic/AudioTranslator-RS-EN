import time

import tensorflow as tf
import pandas as pd
import numpy as np
import librosa
import tensorflow.python.keras.models
from keras import Sequential
from keras.src.layers import Masking, Embedding
from keras.src.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Load your TSV files


# Load data
train_df = pd.read_csv('data/data_sr_rs_train.tsv', delimiter='\t', header=None,
                       names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])

dev_df = pd.read_csv('data/data_sr_rs_dev.tsv', delimiter='\t', header=None,
                     names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess audio data
def load_and_preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

# Apply the preprocessing to all audio files
train_df['MFCC'] = train_df['Audio_Path'].apply(lambda x: load_and_preprocess_audio('data/train/' + x))
dev_df['MFCC'] = dev_df['Audio_Path'].apply(lambda x: load_and_preprocess_audio('data/dev/' + x))

# Tokenize target text
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(train_df['Text'])
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_df['Text'])
dev_sequences = tokenizer.texts_to_sequences(dev_df['Text'])

# Pad sequences to have consistent length
max_len = max(max(map(len, train_sequences)), max(map(len, dev_sequences)))
X_train = pad_sequences(train_sequences, maxlen=max_len, padding='post')
X_dev = pad_sequences(dev_sequences, maxlen=max_len, padding='post')

# Prepare target sequences
y_train = pad_sequences(train_sequences, maxlen=max_len, padding='post')
y_dev = pad_sequences(dev_sequences, maxlen=max_len, padding='post')

# Reshape MFCC data
mfcc_padded = pad_sequences(train_df['MFCC'], padding='post', dtype='float32', maxlen=max_len, truncating='post')
dev_mfcc_padded = pad_sequences(dev_df['MFCC'], padding='post', dtype='float32', maxlen=max_len, truncating='post')

# Expand dimensions to match the input shape expected by the model
mfcc_padded = np.expand_dims(mfcc_padded, axis=-1)
dev_mfcc_padded = np.expand_dims(dev_mfcc_padded, axis=-1)


model = Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=256, input_length=max_len))
model.add(Masking(mask_value=0.0))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))  # Add another LSTM layer
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("mfcc_padded[0]:", mfcc_padded[0])
print("Shape of mfcc_padded:", mfcc_padded.shape)

print("X_train[0]:", X_train[0])
print("Shape of X_train:", X_train.shape)
model.fit(mfcc_padded, np.expand_dims(X_train, -1), epochs=5, batch_size=32,
          validation_data=(dev_mfcc_padded, np.expand_dims(X_dev, -1)))

model.evaluate(dev_mfcc_padded, np.expand_dims(X_dev, -1))

model.save('asr_model.h5')

model = tensorflow.keras.models.load_model('asr_model.h5')
# Testing
test_df = pd.read_csv('data/data_sr_rs_test.tsv', delimiter='\t', header=None,
                      names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])


test_df['MFCC'] = test_df['Audio_Path'].apply(lambda x: load_and_preprocess_audio('data/test/' + x))

test_mfcc_padded = pad_sequences(test_df['MFCC'], padding='post', dtype='float32', maxlen=max_len, truncating='post')
test_mfcc_padded = np.expand_dims(test_mfcc_padded, axis=-1)

predictions = model.predict(test_mfcc_padded)

# Decode predictions back to text
decoded_predictions = [tokenizer.sequences_to_texts(np.argmax(pred, axis=1).reshape(1, -1))[0] for pred in predictions]
from transformers import M2M100ForConditionalGeneration, NllbTokenizer

model = M2M100ForConditionalGeneration.from_pretrained('./nllb-200-distilled-1.3B')
tokenizer = NllbTokenizer.from_pretrained('./nllb-200-distilled-1.3B', source_spm='./sentencepiece.bpe.model')

# Display predictions
for i in range(len(test_df)):
    print(f"ID: {test_df['ID'][i]}, Predicted: {decoded_predictions[i]}, True Text: {test_df['Text'][i]}")

    # Translate from Serbian to English
    input_ids = tokenizer.encode(decoded_predictions[i], return_tensors='pt')

    # Generate translation
    output_ids = model.generate(input_ids, max_length=50, num_beams=5, length_penalty=1.0, no_repeat_ngram_size=2)
    english_translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("English Translation:", english_translation)

# Convert indices to words using the tokenizer

