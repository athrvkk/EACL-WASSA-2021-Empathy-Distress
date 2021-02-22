# File contatins simple RoBERTa with IRI and Personality Inputs
# File: RoBERTa_given_multi_input.py
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







class RoBERTa_iri_personality():


    # ------------------------------------------------------------ Constructor ------------------------------------------------------------
    
    def __init__(self, word_weight_column="weights", base_model_type="CNN", activation="relu", kr_rate=0.001, score_loss="mse", binary_loss="binary_crossentropy", multiclass_loss="sparse_categorical_crossentropy", cpkt="trial"):
        self.iri_features = ["iri_perspective_taking", "iri_personal_distress", "iri_fantasy", "iri_empathatic_concern"]
        self.personality_features = ["personality_conscientiousness", "personality_openess", "personality_extraversion", "personality_agreeableness", "personality_stability"]
    
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

        self.iri_scaler = StandardScaler()
        self.personality_scaler = StandardScaler()

        # ModelCheckPoint Callback:
        if score_loss == "huber":
            cpkt = cpkt + "-kr-{}-{}-{}-{}".format(self.kr_rate, self.activation, score_loss, delta)
        else:
            cpkt = cpkt + "-kr-{}-{}-{}".format(self.kr_rate, self.activation, score_loss)

        cpkt = cpkt + "-epoch-{epoch:02d}-val-loss-{val_loss:02f}.h5"
        checkpoint_filepath = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/model-weights/"+ cpkt
        self.model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
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
                                            patience=40,
                                            verbose=1)
        print("\nActivation: ", self.activation)
        print("Kernel Initializer: ", self.kr_initializer)
        print("Kernel Regularizing Rate: ", self.kr_rate)
        print("\n")





    # ------------------------------------------------------------ Function to prepare input for respective models ------------------------------------------------------------
    
    def prepare_input(self, utils_obj, df, maxlen=200, padding_type='post', truncating_type='post', mode="train"):
        essay = [pre.clean_text(text, remove_stopwords=False, lemmatize=False) for text in df.essay.values.tolist()]

        iri_score = df[self.iri_features].values
        personality_score = df[self.personality_features].values
        
        if self.base_model_type in self.bert_models:
            return [self.base_model.prepare_input(essay, maxlen),
                    iri_score, 
                    personality_score]
        else:
            return [self.base_model.prepare_input(utils_obj, essay, maxlen, padding_type, truncating_type, mode), 
                    iri_score, 
                    personality_score]





    # ------------------------------------------------------------ Funciton to prepare model outputs ------------------------------------------------------------
    
    def prepare_output(self,utils,  df, task="empathy", mode="train"):

        if task == "empathy":
            print("\nIn empathy\n")
            score = np.reshape(df.gold_empathy.values.tolist(), (len(df), 1))
            return score

        if task == "distress":
            print("\nIn distress\n")
            score = np.reshape(df.gold_distress.values.tolist(), (len(df), 1))
            return score





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
        x1 = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(base_output)

        # IRI Input
        iri_input = Input(shape=(len(self.iri_features),), name="iri_input")
        iri_dense = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer)(iri_input)
        
        # Big 5 Personality Input
        personality_input = Input(shape=(len(self.personality_features),), name="personality_input")
        personality_dense = Dense(8, activation=self.activation, kernel_initializer=self.kr_initializer)(personality_input)

        x2 = concatenate([iri_dense, personality_dense])
        x2 = Dense(32, activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x2)

        # Predict Score
        x = Concatenate(axis=1)([x1, x2])
        x = Dense(16, activation=self.activation, kernel_initializer=self.kr_initializer)(x)
        score = Dense(1, name='score_output')(x)
        
        if self.base_model_type in self.bert_models:
            self.model = Model(inputs=[input_ids, 
                                       attention_mask,
                                       iri_input,
                                       personality_input],
                               outputs=score)
        else:
            self.model = Model(inputs=[input, iri_input, personality_input], 
                               outputs=[bin, emotion, score])
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
        model_path = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/best-models/"+model_path
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

        
