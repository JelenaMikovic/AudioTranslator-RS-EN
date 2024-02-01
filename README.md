# AudioTranslator-RS-EN

- Problem Definition:
In our globalized society, there's a growing demand for efficient audio translation from Serbian to other languages. Effective solutions in audio translation facilitate authentic international communication and access to cultural content in the Serbian language.

- Dataset:
Audio Recognition: Google Fleurs Dataset

- Methodology:
We filter and process audio and textual materials in the Serbian language from the dataset. Specifically, we developed a model for translating audio recordings into textual format and training on relevant data. The generated text is then translated into the target language (in this case English). Finally, the translated text is transformed into audio using a pre-trained and optimized text-to-speech model.

- Evaluation:
Performance metrics like Word Error Rate (WER) will be employed to evaluate the models.

- Models:
For Translation: Facebook NLLB-200 Distilled 1.3B
For Text-to-Speech: Facebook MMS-TTS-ENG
For Audio-to-Text: This project employs a Recurrent Neural Network (RNN) model with bidirectional Long Short-Term Memory (LSTM) units, utilizing the Connectionist Temporal Classification (CTC) loss in TensorFlow. 

- Summary:
The focus is on accurate audio transcription, leveraging advanced deep-learning techniques. The models are trained on a diverse dataset, showcasing their versatility for various natural language processing and speech recognition applications. This project aims to provide a robust solution for audio translation, leveraging state-of-the-art models and methodologies. Contributions from the team and the use of cutting-edge technologies make this repository a valuable resource for those interested in audio translation and natural language processing.
