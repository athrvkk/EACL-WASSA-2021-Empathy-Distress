# File contatins RoBERTa model with entity embeddings and IRI and Personality Inputs
# File: RoBERTa_entity_embeddings_iri_personality.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

from utils.utils import Utils
from bert_model import BertModel
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







class RoBERTa_multi_input():

    # ------------------------------------------------------------ Constructor ------------------------------------------------------------
    
    def __init__(self, task="empathy", activation="relu", kr_rate=0.001, score_loss="mse", binary_loss="binary_crossentropy", multiclass_loss="sparse_categorical_crossentropy", cpkt="trial"):
        
        self.task = task
        self.kr_rate = kr_rate
        self.iri_features = ["iri_perspective_taking", "iri_personal_distress", "iri_fantasy", "iri_empathatic_concern"]
        self.personality_features = ["personality_conscientiousness", "personality_openess", "personality_extraversion", "personality_agreeableness", "personality_stability"]
        
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

        self.gender_encoder = LabelEncoder()
        self.education_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()

        self.gender_onehot_encoder = OneHotEncoder(sparse=False)
        self.education_onehot_encoder = OneHotEncoder(sparse=False)
        self.race_onehot_encoder = OneHotEncoder(sparse=False)
        self.age_onehot_encoder = OneHotEncoder(sparse=False)

        self.iri_scaler = StandardScaler()
        self.personality_scaler = StandardScaler()

        # ModelCheckPoint Callback:
        if score_loss == "huber":
            cpkt = cpkt + "-kr-{}-{}-{}-{}".format(self.kr_rate, self.activation, score_loss, delta)
        else:
            cpkt = cpkt + "-kr-{}-{}-{}".format(self.kr_rate, self.activation, score_loss)

        cpkt = cpkt + "-epoch-{epoch:02d}-val-loss-{val_loss:02f}.h5"
        self.model_checkpoint_callback = ModelCheckpoint(filepath=cpkt,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_freq = 'epoch',
                                                    save_best_only=True)

        # Reduce Learning Rate on Plateau Callback:
        self.reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', 
                                                    mode='auto',
                                                    factor=0.2, 
                                                    patience=10, 
                                                    min_lr=0.0005, 
                                                    verbose=1)
        # Early Stopping
        self.early_stopping = EarlyStopping(monitor='val_loss', 
                                            patience=20,
                                            verbose=1)
        print("\nActivation: ", self.activation)
        print("Kernel Initializer: ", self.kr_initializer)
        print("Kernel Regularizing Rate: ", self.kr_rate)
        print("\n")





    # ------------------------------------------------------------ Function to prepare input for respective models ------------------------------------------------------------
    
    def prepare_input(self, pre, utils_obj, df, maxlen=200, padding_type='post', truncating_type='post', mode="train"):
        essay = [pre.clean_text(text, remove_stopwords=False, lemmatize=False) for text in df.essay.values.tolist()]

        gender = np.reshape(df.gender.values.tolist(), (len(df), 1))
        education = np.reshape(df.education.values.tolist(), (len(df), 1))
        race = np.reshape(df.race.values.tolist(), (len(df), 1))
        age = np.reshape(df.age.apply(lambda x: utils_obj.categorize_age(x)).values.tolist(), (len(df), 1))

        iri_input = df[self.iri_features].values
        personality_input = df[self.personality_features].values

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

            iri_input = self.iri_scaler.fit_transform(iri_input)
            personality_input = self.personality_scaler.fit_transform(personality_input)

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

            iri_input = self.iri_scaler.transform(iri_input)
            personality_input = self.personality_scaler.transform(personality_input)

        iri_input = np.reshape(iri_input, (len(iri_input), len(iri_input[0])))
        personality_input = np.reshape(personality_input, (len(personality_input), len(personality_input[0])))
        

        return [self.base_model.prepare_input(essay, maxlen), 
                gender,
                education,
                race,
                age,
                iri_input, 
                personality_input]





    # ------------------------------------------------------------ Funciton to prepare model outputs ------------------------------------------------------------
    
    def prepare_output(self,utils,  df, mode="train"):
        if self.task == "empathy":
            score = np.reshape(df.gold_empathy.values.tolist(), (len(df), 1))
            return score
        if self.task == "distress":
            score = np.reshape(df.gold_distress.values.tolist(), (len(df), 1))
            return score





    # ------------------------------------------------------------ Function to build the model ------------------------------------------------------------
    
    def build(self, embedding_matrix, input_length=100):
        input_ids = Input(shape=(input_length,), name="input_ids")
        attention_mask = Input(shape=(input_length,), name="attention_mask")
        base_output = self.base_model.build(input_length)([input_ids, attention_mask])
       
        # Plain RoBERTa representations
        x1 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)

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

        x2 = Concatenate(axis=1)([gender_embedding, education_embedding, race_embedding, age_embedding])
        x2 = Dense(32, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x2)
        x2 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x2)

        # IRI Input
        iri_input = Input(shape=(4,), name="iri_input")
        iri_dense = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer)(iri_input)
        
        # Big 5 Personality Input
        personality_input = Input(shape=(5,), name="personality_input")
        personality_dense = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer)(personality_input)

        x3 = Concatenate(axis=1)([iri_dense, personality_dense])
        x3 = Dense(32, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x3)
        
        # Predict Score
        x = Concatenate(axis=1)([x1, x2, x3])
        x = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer)(x)
        score = Dense(1, name='score_output')(x)
        
        self.model = Model(inputs=[input_ids, 
                                   attention_mask,
                                   gender_input,
                                   education_input,
                                   race_input,
                                   age_input,
                                   iri_input,
                                   personality_input], 
                           outputs=score)
       
        self.model.compile(optimizer=Adam(lr=0.001), 
                           loss={"score_output":self.score_loss},
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
        self.model.load_weights(model_path)
        pred = self.model.predict(val_essay)
        return pred





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

        

        
