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
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, LayerNormalization, Permute
from tqdm import tqdm

# Load your TSV files
allowed_characters = set("абвгдђежзијклљмнњопрстћуфхцчџш ")

# Function to filter and clean text
def clean_text(text):
    if re.search('[\u0400-\u04FF]', text):
        cleaned_text = ''.join(char for char in text if char in allowed_characters)
        return cleaned_text
    else:
        return None

# Load data
train_df = pd.read_csv('data/data_sr_rs_train.tsv', delimiter='\t', header=None,
                       names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])

train_df = train_df[~train_df['Text'].str.contains('\d', na=False)]
train_df = train_df[~train_df['Raw_Transcript'].str.contains('\d', na=False)]
train_df = train_df[~train_df['Tokens'].str.contains('\d', na=False)]

train_df['Processed_Text'] = train_df['Text'].apply(lambda x: clean_text(x.lower()))

dev_df = pd.read_csv('data/data_sr_rs_dev.tsv', delimiter='\t', header=None,
                     names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])
test_df = pd.read_csv('data/data_sr_rs_test.tsv', delimiter='\t', header=None,
                     names=['ID', 'Audio_Path', 'Text', 'Raw_Transcript', 'Tokens', 'Audio_Length', 'Gender'])

test_df = test_df[35:45]
dev_df = dev_df[~dev_df['Text'].str.contains('\d', na=False)]
dev_df = dev_df[~dev_df['Raw_Transcript'].str.contains('\d', na=False)]
dev_df = dev_df[~dev_df['Tokens'].str.contains('\d', na=False)]

dev_df['Processed_Text'] = dev_df['Text'].apply(lambda x: clean_text(x.lower()))

test_df = test_df[~test_df['Text'].str.contains('\d', na=False)]
test_df = test_df[~test_df['Raw_Transcript'].str.contains('\d', na=False)]
test_df = test_df[~test_df['Tokens'].str.contains('\d', na=False)]

test_df['Processed_Text'] = test_df['Text'].apply(lambda x: clean_text(x.lower()))

train=train_df[['Audio_Path','Processed_Text']]
val = dev_df[["Audio_Path","Processed_Text"]]
test = test_df[["Audio_Path","Processed_Text"]]


np.random.seed(42)
tf.random.set_seed(42)


characters = [char for char in "абвгдђежзијклљмнњопрстћуфхцчџш "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="",invert=True
)
print(char_to_num.get_vocabulary())
print(len(char_to_num.get_vocabulary()))
# Print all tokens
for char in characters:
    print(f"{char}")

# An integer scalar Tensor. The window length in samples.
frame_length = 200
# An integer scalar Tensor. The number of samples to step.
frame_step = 80
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 256



def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    #label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label

# Define the training dataset
batch_size = 32



print(train["Processed_Text"])
train_dataset = tf.data.Dataset.from_tensor_slices((list("data/train/" + train["Audio_Path"]), list([char for char in train["Processed_Text"]])))
train_dataset = (
    train_dataset.map(encode_single_sample,num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((list("data/dev/" + val["Audio_Path"]), list([char for char in val["Processed_Text"]])))
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

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
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
    model = tf.keras.Model(input_spectrogram, output, name="ASR")
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
# model = build_model(
#     input_dim=fft_length // 2 + 1,
#     output_dim=char_to_num.vocabulary_size(),
#     rnn_units=128,
# )
# model.summary(line_length=110)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)

# Define the number of epochs.
epochs = 50
# Callback function to check transcription on the val set.
#validation_callback = CallbackEval(val_dataset)
# Train the model
#history = model.fit(
#    train_dataset,
#    validation_data=val_dataset,
#    epochs=epochs,
#    callbacks=[validation_callback],
#)

#model.save('/content/drive/MyDrive/asr_model', save_format="h5")
#predictions = []
#targets = []
#i=0
#for batch in tqdm(train_dataset):
#     X, y = batch
#     batch_predictions = model.predict(X)
#     batch_predictions = decode_batch_predictions(batch_predictions)
#     print(batch_predictions)
#     i+=1
#     if i>5:
#         break
#     predictions.extend(batch_predictions)
#     for label in y:
#         label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#         targets.append(label)
# print(targets)
# print(predictions)
# wer_score = wer(targets, predictions)
# print("-" * 100)
# print(f"Word Error Rate: {wer_score:.4f}")
# print("-" * 100)
# for i in np.random.randint(0, len(predictions), 5):
#     print(f"Target    : {targets[i]}")
#     print(f"Prediction: {predictions[i]}")
#     print("-" * 100)

from transformers import M2M100ForConditionalGeneration, NllbTokenizer, M2M100Tokenizer
from transformers import VitsModel, AutoTokenizer
import torch
import scipy

model = M2M100ForConditionalGeneration.from_pretrained("./m2m-finetuned-418M")
tokenizer = M2M100Tokenizer.from_pretrained("./m2m-finetuned-418M")

tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

# Assuming test_df is your DataFrame
test = test_df[["Audio_Path", "Processed_Text"]]

# Iterate through each row
for index, row in test.iterrows():
    input_text = row["Processed_Text"]

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    tokenizer.src_lang = "sr"
    encoded_hi = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
    output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    print("Input:")
    print(input_text)
    print("-"*100)
    print("Output:")
    print(output_text)

    inputs = tts_tokenizer(output_text, return_tensors="pt")

    with torch.no_grad():
        output = tts_model(**inputs).waveform
        scipy.io.wavfile.write("output_audio/"+str(index)+".wav", rate=tts_model.config.sampling_rate, data=output.float().numpy().T)
