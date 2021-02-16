# File contatins Class for Mutli-task learning model 
# File: mtlm.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

from utils.utils import Utils
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import Model






class MTLM():


    # ------------------------------------------------------------ Constructor ------------------------------------------------------------
    
    def __init__(self, word_weight_column="weights", base_model_type="CNN", activation="relu", kr_rate=0.001, score_loss="mse", binary_loss="binary_crossentropy", multiclass_loss="sparse_categorical_crossentropy", cpkt="trial"):
        self.word_weights = word_weights = utils.get_dict("/content/gdrive/My Drive/WASSA-2021-Shared-Task/empathy_word_weights.csv",
                                                          key_column="words",
                                                          value_column=word_weight_column)
        self.iri_features = ["iri_perspective_taking", "iri_personal_distress", "iri_fantasy", "iri_empathatic_concern"]
        self.personality_features = ["personality_conscientiousness", "personality_openess", "personality_extraversion", "personality_agreeableness", "personality_stability"]
        self.liwc_features = []
        self.empath_features = []
        
        self.kr_rate = kr_rate
        # Set the model activation:
        if activation == "leaky_relu":
            self.activation = LeakyReLU()
            self.kr_initializer = tf.keras.initializers.HeUniform()
        elif activation == "paramaterized_leaky_relu":
            self.activation = PReLU() 
            self.kr_initializer = tf.keras.initializers.HeUniform()          
        elif activation == "relu":
            self.activation = "relu"
            self.kr_initializer = tf.keras.initializers.HeUniform()
        else:
            self.activation = activation
            self.kr_initializer  = tf.keras.initializers.GlorotUniform()

        # Set the regression loss:
        self.score_metric = "mean_squared_error"
        if score_loss == "huber":
            delta = 2.0
            self.score_loss = losses.Huber(delta=delta)
        elif score_loss == "log_cosh":
            self.score_loss = "log_cosh"
        elif score_loss == "mean_squared_logarithmic_error":
            self.score_loss = "mean_squared_logarithmic_error"
        elif score_loss == "mae":
            self.score_loss = "mae"
        else:
            self.score_loss = "mse"

        # Set the binary classification loss:
        if binary_loss == "hinge":
            self.binary_loss = "hinge"
            self.binary_activation = "tanh"
        elif binary_loss == "squared_hinge":
            self.binary_loss = "squared_hinge"
            self.binary_activation = "tanh"
        else:
            self.binary_loss = "binary_crossentropy"
            self.binary_activation = "sigmoid"

        # Set the multi-class calssification loss:
        if multiclass_loss == "kld":
            self.multiclass_loss = "kl_divergence"
        else:
            self.multiclass_loss = "sparse_categorical_crossentropy"

        self.base_model_type = base_model_type
        self.bert_models = ["BERT", "DistilBERT", "RoBERTa", "custom"]
        if self.base_model_type in self.bert_models:
            self.base_model = BertModel(self.activation, self.kr_initializer, self.kr_rate, self.base_model_type, output_hidden_states=False)
        elif self.base_model_type == "CNN":
            self.base_model = CNN(self.activation, self.kr_initializer, self.kr_rate)
        elif self.base_model_type == "BiLSTM":
            self.base_model = BiLSTM(self.activation, self.kr_initializer, self.kr_rate)

        self.gender_encoder = LabelEncoder()
        self.education_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()
        self.emotion_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()

        self.gender_onehot_encoder = OneHotEncoder(sparse=False)
        self.education_onehot_encoder = OneHotEncoder(sparse=False)
        self.race_onehot_encoder = OneHotEncoder(sparse=False)
        self.emotion_onehot_encoder = OneHotEncoder(sparse=False)
        self.age_onehot_encoder = OneHotEncoder(sparse=False)
        
        self.polarity_subjectivity_scaler = MinMaxScaler(feature_range=(1, 7))
        self.emotion_lexicon_scaler = MinMaxScaler(feature_range=(1, 7))
        self.vad_lexicon_scaler = MinMaxScaler(feature_range=(1, 7))
        self.iri_scaler = MinMaxScaler(feature_range=(1, 7))
        self.personality_scaler = MinMaxScaler(feature_range=(1, 7))

        # self.polarity_subjectivity_scaler = StandardScaler()
        # self.emotion_lexicon_scaler = StandardScaler()
        # self.vad_lexicon_scaler = StandardScaler()
        # self.iri_scaler = StandardScaler()
        # self.personality_scaler = StandardScaler()

        # ModelCheckPoint Callback:
        if score_loss == "huber":
            cpkt = cpkt + "-kr-{}-{}-{}-{}".format(self.kr_rate, self.activation, score_loss, delta)
        else:
            cpkt = cpkt + "-kr-{}-{}-{}".format(self.kr_rate, self.activation, score_loss)

        cpkt = cpkt + "-epoch-{epoch:02d}-val-loss-{val_score_output_loss:02f}.h5"
        checkpoint_filepath = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/model-weights/"+ cpkt
        self.model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                                    save_weights_only=True,
                                                    monitor='val_score_output_loss',
                                                    mode='auto',
                                                    save_freq = 'epoch',
                                                    save_best_only=True)

        # Reduce Learning Rate on Plateau Callback:
        self.reduce_lr_callback = ReduceLROnPlateau(monitor='val_score_output_loss', 
                                                    mode='auto',
                                                    factor=0.2, 
                                                    patience=10, 
                                                    min_lr=0.0005, 
                                                    verbose=1)
        # Early Stopping
        self.early_stopping = EarlyStopping(monitor='val_score_output_loss', 
                                            patience=40,
                                            verbose=1)
        print("\nActivation: ", self.activation)
        print("Kernel Initializer: ", self.kr_initializer)
        print("Kernel Regularizing Rate: ", self.kr_rate)
        print("\n")





    # ------------------------------------------------------------ Function to prepare input for respective models ------------------------------------------------------------
    
    def prepare_input(self, utils_obj, df, maxlen=200, padding_type='post', truncating_type='post', mode="train"):
        # polarity_subjectivity_score = [TextBlob(text).sentiment for text in df.essay.values.tolist()]

        essay = [pre.clean_text(text, remove_stopwords=False, lemmatize=False) for text in df.essay.values.tolist()]
        #lemmatized_essay = [pre.clean_text(text, remove_stopwords=False, lemmatize=False) for text in df.essay.values.tolist()]
        #essay_weight_score = utils.get_essay_empathy_distress_scores(lemmatized_essay, self.word_weights)

        # emotion_lexicon_score = utils.get_essay_emotion_vad_scores(essay, df.WC.values.tolist(), mode="emotion")
        # vad_lexicon_score = utils.get_essay_emotion_vad_scores(essay, df.WC.values.tolist(), mode="vad")
        
        gender = np.reshape(df.gender.values.tolist(), (len(df), 1))
        education = np.reshape(df.education.values.tolist(), (len(df), 1))
        race = np.reshape(df.race.values.tolist(), (len(df), 1))
        age = np.reshape(df.age.apply(lambda x: utils.categorize_age(x)).values.tolist(), (len(df), 1))

        iri_input = df[self.iri_features].values.tolist()
        personality_input = df[self.personality_features].values.tolist()

        if mode == "train":
            gender = self.gender_encoder.fit_transform(gender)
            gender = np.reshape(gender, (len(gender), 1))
            gender = self.gender_onehot_encoder.fit_transform(gender)
        
            education = self.education_encoder.fit_transform(education)
            education = np.reshape(education, (len(education), 1))
            education = self.education_onehot_encoder.fit_transform(education)

            race = self.race_encoder.fit_transform(race)
            race = np.reshape(race, (len(race), 1))
            race = self.race_onehot_encoder.fit_transform(race)

            age = self.age_encoder.fit_transform(age)
            age = np.reshape(age, (len(age), 1))
            age = self.age_onehot_encoder.fit_transform(age)
        #     polarity_subjectivity_score = self.polarity_subjectivity_scaler.fit_transform(polarity_subjectivity_score)
        #     emotion_lexicon_score = self.emotion_lexicon_scaler.fit_transform(emotion_lexicon_score)
        #     vad_lexicon_score = self.vad_lexicon_scaler.fit_transform(vad_lexicon_score)
        #     iri_input = self.iri_scaler.fit_transform(iri_input)
        #     personality_input = self.personality_scaler.fit_transform(personality_input)

        elif mode == "dev" or "test":
            gender = self.gender_encoder.transform(gender)
            gender = np.reshape(gender, (len(gender), 1))
            gender = self.gender_onehot_encoder.transform(gender)
        
            education = self.education_encoder.transform(education)
            education = np.reshape(education, (len(education), 1))
            education = self.education_onehot_encoder.transform(education)

            race = self.race_encoder.transform(race)
            race = np.reshape(race, (len(race), 1))
            race = self.race_onehot_encoder.transform(race)

            age = self.age_encoder.transform(age)
            age = np.reshape(age, (len(age), 1))
            age = self.age_onehot_encoder.transform(age)
        #     polarity_subjectivity_score = self.polarity_subjectivity_scaler.transform(polarity_subjectivity_score)
        #     emotion_lexicon_score = self.emotion_lexicon_scaler.transform(emotion_lexicon_score)
        #     vad_lexicon_score = self.vad_lexicon_scaler.transform(vad_lexicon_score)
        #     iri_input = self.iri_scaler.transform(iri_input)
        #     personality_input = self.personality_scaler.transform(personality_input)
        
        # polarity_subjectivity_score = np.reshape(polarity_subjectivity_score, (len(polarity_subjectivity_score), len(polarity_subjectivity_score[0])))

        iri_input = np.reshape(iri_input, (len(iri_input), len(iri_input[0])))
        personality_input = np.reshape(personality_input, (len(personality_input), len(personality_input[0])))
        
        if self.base_model_type in self.bert_models:
            return [self.base_model.prepare_input(essay, maxlen), 
                    gender,
                    education,
                    race,
                    age,
                    iri_input, 
                    personality_input]
        else:
            return [self.base_model.prepare_input(utils_obj, essay, maxlen, padding_type, truncating_type, mode), 
                    gender,
                    education,
                    race,
                    age,
                    iri_input, 
                    personality_input]





    # ------------------------------------------------------------ Funciton to prepare model outputs ------------------------------------------------------------
    
    def prepare_output(self,utils,  df, task="empathy", mode="train"):
        emotion = np.reshape(df.gold_emotion.values.tolist(), (len(df), 1))   

        if mode == "train":
            emotion = self.emotion_encoder.fit_transform(emotion)

        elif mode == "dev" or "test":
            emotion = self.emotion_encoder.transform(emotion)

        if task == "empathy":
            score = np.reshape(df.gold_empathy.values.tolist(), (len(df), 1))
            bin = np.reshape(df.gold_empathy_bin.values.tolist(), (len(df), 1))
            return [bin, emotion, score]
        if task == "distress":
            score = np.reshape(df.gold_distress.values.tolist(), (len(df), 1))
            bin = np.reshape(df.gold_distress_bin.values.tolist(), (len(df), 1))
            return[bin, emotion, score]





    # ------------------------------------------------------------ Function to build the model ------------------------------------------------------------
    
    def build(self, embedding_matrix, input_length=100):
        if self.base_model_type in self.bert_models:
            input_ids = Input(shape=(input_length,), name="input_ids")
            attention_mask = Input(shape=(input_length,), name="attention_mask")
            base_output = self.base_model.build(input_length)([input_ids, attention_mask])
        else:
            input = Input(shape=(input_length,), name="base_model_input")
            base_output = self.base_model.build(input_length, embedding_matrix)(input)

        # Predict Bin
        # essay_weight_score_input = Input(shape=(1,))
        # essay_weight_score_dense = Dense(4, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(essay_weight_score_input)

        x1 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)
        bin = Dense(1, activation="sigmoid", name='bin_output')(x1)

        # Predict Emotion
        # emotion_lexicon_input = Input(shape=(8,), name="emotion_lexicon_input")
        # emoiton_lexicon_dense = Dense(5, activation=self.activation, kernel_initializer=self.kr_initializer)(emotion_lexicon_input)

        # vad_lexicon_input = Input(shape=(3,), name="vad_lexicon_input")
        # vad_lexicon_dense = Dense(2, activation=self.activation, kernel_initializer=self.kr_initializer)(vad_lexicon_input)

        # polarity_subjectivity_input = Input(shape=(2,), name="polarity_subjectivity_input")
        # polarity_subjectivity_dense = Dense(2, activation=self.activation, kernel_initializer=self.kr_initializer)(polarity_subjectivity_input)

        # emotion_dense = Concatenate(axis=1)([emotion_lexicon_input, vad_lexicon_input, polarity_subjectivity_input])
        # emotion_dense = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(emotion_dense)

        x2 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)
        # x2 = Concatenate(axis=1)([x2, polarity_subjectivity_dense])
        emotion = Dense(7, activation='softmax', name='emotion_output')(x2)

        # Predict Gender
        # x3 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)
        # gender = Dense(3, activation='softmax', name='gender_output')(x3)

        # Predict Education
        # x4 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)
        # education = Dense(6, activation='softmax', name='education_output')(x4)
  
        # Predict Race
        # x5 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)
        # race = Dense(6, activation='softmax', name='race_output')(x5)

        # Categorical data inputs:

        gender_input = Input(shape=(3,), name='gender_input')
        gender_embedding = Embedding(input_dim=3, output_dim=3, trainable=True)(gender_input)
        gender_embedding = Flatten()(gender_embedding)

        education_input = Input(shape=(6,), name='education_input')
        education_embedding = Embedding(input_dim=6, output_dim=3, trainable=True)(education_input)
        education_embedding = Flatten()(education_embedding)

        race_input = Input(shape=(6,), name='race_input')
        race_embedding = Embedding(input_dim=6, output_dim=3, trainable=True)(race_input)
        race_embedding = Flatten()(race_embedding)

        age_input = Input(shape=(4,), name='age_input')
        age_embedding = Embedding(input_dim=4, output_dim=3, trainable=True)(age_input)
        age_embedding = Flatten()(age_embedding)

        # gender_input = Input(shape=(1,), name='gender_input')
        # gender_embedding = Embedding(input_dim=3, output_dim=3, trainable=True)(gender_input)
        # gender_embedding = Reshape(target_shape=(3,))(gender_embedding)

        # education_input = Input(shape=(1,), name='education_input')
        # education_embedding = Embedding(input_dim=6, output_dim=3, trainable=True)(education_input)
        # education_embedding = Reshape(target_shape=(3,))(education_embedding)

        # race_input = Input(shape=(1,), name='race_input')
        # race_embedding = Embedding(input_dim=6, output_dim=3, trainable=True)(race_input)
        # race_embedding = Reshape(target_shape=(3,))(race_embedding)

        # age_input = Input(shape=(1,), name='age_input')
        # age_embedding = Embedding(input_dim=4, output_dim=3, trainable=True)(age_input)
        # age_embedding = Reshape(target_shape=(3,))(age_embedding)

        categorical_concat = Concatenate(axis=1)([gender_embedding, education_embedding, race_embedding, age_embedding])
        categorical_dense = Dense(32, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(categorical_concat)
        categorical_dense = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(categorical_dense)

        # IRI Input
        iri_input = Input(shape=(4,), name="iri_input")
        x6 = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer)(iri_input)
        
        # Big 5 Personality Input
        personality_input = Input(shape=(5,), name="personality_input")
        x7 = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer)(personality_input)

        x = Concatenate(axis=1)([x6, x7])
        x = Dense(32, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x)
        
        # Predict Score
        x = Concatenate(axis=1)([x1, x2, categorical_dense, x])
        x = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer)(x)
        score = Dense(1, name='score_output')(x)
        
        if self.base_model_type in self.bert_models:
            self.model = Model(inputs=[input_ids, 
                                       attention_mask,
                                       gender_input,
                                       education_input,
                                       race_input,
                                       age_input,
                                       iri_input,
                                       personality_input], 
                               outputs=[bin, emotion, score])
        else:
            self.model = Model(inputs=[input, iri_input, personality_input], 
                               outputs=[bin, emotion, score])
        self.model.compile(optimizer=Adam(lr=0.001), 
                           loss={"bin_output":self.binary_loss,                                                           
                                 "emotion_output":self.multiclass_loss,
                                 "score_output":self.score_loss},
                           metrics={"score_output":self.score_metric})
        self.model.summary()





    # ------------------------------------------------------------ Function to plot model architecture ------------------------------------------------------------
        
    def plot_model_arch(self):
        return plot_model(self.model, show_shapes=True)





    # ------------------------------------------------------------ Function to train the model ------------------------------------------------------------
    
    def train(self, x_train, y_train, x_val, y_val, epochs=200, batch_size=32):
        history = self.model.fit(x_train,
                                 y_train, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 verbose=1, 
                                 validation_data = (x_val, y_val),
                                 callbacks=[self.model_checkpoint_callback, self.reduce_lr_callback, self.early_stopping])
        return history





    # ------------------------------------------------------------ Function to predict model output ------------------------------------------------------------
    
    def prediction(self, val_essay, model_path=""):
        model_path = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/model-weights/"+model_path
        self.model.load_weights(model_path)
        pred = self.model.predict(val_essay)
        return pred[-1]





    # ------------------------------------------------------------ Function to calculate the Pearson's correlation ------------------------------------------------------------
    
    def compute_correlation(self, y_true, y_pred):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        return pearsonr(y_true, y_pred)
        
        
        


    # ------------------------------------------------------------ Function to calculate the Pearson's correlation ------------------------------------------------------------
    
    def compute_mse(self, y_true, y_pred):  
        return np.average(losses.mean_squared_error(y_true, y_pred))      





    # ------------------------------------------------------------ Function to plot model loss ------------------------------------------------------------
    
    def plot_curves(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.show() 

        
        
        
        
        
        
