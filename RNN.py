import time
import re

import tensorflow as tf
import pandas as pd
import numpy as np
import librosa
import tensorflow.python.keras.models
from jiwer import wer
from keras import Sequential, preprocessing
from keras.src.layers import Masking, Embedding, Bidirectional, LayerNormalization, Dropout, Permute
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from tensorflow.python.keras import models
from tensorflow.python.keras.models import load_model
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, LayerNormalization, Permute
from tqdm import tqdm

# Load your TSV files


# Load data
train_df = pd.read_csv('data/data_sr_rs_train.tsv', delimiter='\t', header=None,
                       names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])
train_df = train_df[:3]
dev_df = pd.read_csv('data/data_sr_rs_dev.tsv', delimiter='\t', header=None,
                     names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])
dev_dv = dev_df[:2]
train=train_df[['Audio_Path','Text']]
val = dev_df[["Audio_Path","Text"]]

np.random.seed(42)
tf.random.set_seed(42)

preprocessed_texts = [ re.sub(r'[^0-9\s\u0430-\u0448\u0410-\u0428\u0458]', '', text.lower()) for text in train_df['Raw_Transcript']]

tokenizer = Tokenizer(char_level=True)

tokenizer.fit_on_texts(preprocessed_texts)
word_index = tokenizer.word_index

characters = [x for x in word_index.keys()]

char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
# Print all tokens
for char in word_index.keys():
    print(f"{char}")

frame_length = 200
frame_step = 80
fft_length = 256


@tf.function
def process_audio_file_tensor(wav_file):

    # Process audio using librosa
    audio = tf.py_function(process_audio_file_with_librosa, [wav_file], Tout=tf.float32)

    # Convert NumPy array to TensorFlow tensor
    return audio

def process_audio_file_with_librosa(wav_file):
    # 1. Read the wav file using librosa
    audio, sr = librosa.load(wav_file.numpy().decode('utf-8'), sr=None)

    # 2. Change type to float32
    audio = audio.astype(np.float32)

    # 3. Get the spectrogram using librosa
    spectrogram = librosa.stft(audio, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)

    # 4. We only need the magnitude
    spectrogram = np.abs(spectrogram)

    # 5. Square root compression
    spectrogram = np.sqrt(spectrogram)

    # 6. Normalization
    means = np.mean(spectrogram, axis=1, keepdims=True)
    stddevs = np.std(spectrogram, axis=1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram

def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    # Decode the wav file using librosa within a tf.py_function
    audio = tf.py_function(process_audio_file_tensor, [wav_file], tf.float32)
    audio.set_shape([None, None])
    audio = tf.cast(audio, tf.float32)




    ###########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)

    # 9. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")

    # 10. Map the characters in label to numbers
    label = char_to_num(label)

    # 11. Return a dict as our model is expecting two inputs
    return audio, label

# Define the training dataset
batch_size = 32

# Define the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((list("data/train/" + train["Audio_Path"]), list(train["Text"])))
train_dataset = (
    train_dataset.map(encode_single_sample,num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((list("data/dev/" + val["Audio_Path"]), list(val["Text"])))
val_dataset = (
    val_dataset.map(encode_single_sample,num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    print(input_length,label_length)
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = tf.keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=128,
)
model.summary(line_length=110)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# Define the number of epochs.
epochs = 1
# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
)

predictions = []
targets = []
i=0
for batch in tqdm(val_dataset):
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    print(batch_predictions)
    i+=1
    if i>5:
        break
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)
wer_score = wer(targets, predictions)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)
for i in np.random.randint(0, len(predictions), 5):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 100)


from transformers import M2M100ForConditionalGeneration, NllbTokenizer

model = M2M100ForConditionalGeneration.from_pretrained('./nllb-200-distilled-1.3B')
tokenizer = NllbTokenizer.from_pretrained('./nllb-200-distilled-1.3B', source_spm='./sentencepiece.bpe.model')

# Display predictions

# Translate from Serbian to English
input_ids = tokenizer.encode("Asdasdds", return_tensors='pt')

# Generate translation
output_ids = model.generate(input_ids, max_length=50, num_beams=5, length_penalty=1.0, no_repeat_ngram_size=2)
english_translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("English Translation:", english_translation)

# Convert indices to words using the tokenizer

