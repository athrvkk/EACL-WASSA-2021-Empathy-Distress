# file contatins Class for Mutli-task learning model 
# File: mtlm.py
# Author: Atharva Kulkarni


from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Conv1D, LSTM, Bidirectional, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, GlobalMaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from cnn import CNN





class MTLM():


    def __init__(self, base_model_type="CNN", cpkt="trial"):
        self.base_model_type = base_model_type
        if self.base_model_type == "CNN":
            self.base_model  =CNN()
        # ModelCheckPoint Callback:
        checkpoint_filepath = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/model-weights/"+ cpkt + "-epoch-{epoch:02d}-val-loss-{val_loss:02f}.h5"
        self.model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_freq = 'epoch',
                                                    save_best_only=False)

        # Reduce Learning Rate on Plateau Callback:
        self.reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', 
                                                    mode='auto',
                                                    factor=0.2, 
                                                    patience=10, 
                                                    min_lr=0.0005, 
                                                    verbose=2)
        # Early Stopping
        self.early_stopping = EarlyStopping(monitor='val_loss', 
                                            patience=20)




    def build(self, embedding_matrix, input_length=100):
        input = Input(shape=(input_length,))
        x = self.base_model.build(input_length, embedding_matrix)(input)

        x = Dense(32, activation="relu")(x)
        bin = Dense(1, activation='sigmoid', name='bin_output')(x)

        x = Dense(8, activation="relu")(x)
        score = Dense(1, name='score_output')(x)
        
        self.model = Model(inputs=input, outputs=[score, bin])
        self.model.compile(optimizer=Adam(lr=0.001), loss={"score_output":"mse", "bin_output":"binary_crossentropy"})
        self.model.summary()




    def plot_model_arch(self):
        return plot_model(self.model, show_shapes=True)




    def train(self, x_train, score_train, bin_train, x_val, score_val, bin_val, epochs=200, batch_size=32):
        history = self.model.fit(x_train,
                                 [score_train, bin_train], 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 verbose=1, 
                                 validation_data = (x_val, [score_val, bin_val]),
                                 callbacks=[self.model_checkpoint_callback, self.reduce_lr_callback, self.early_stopping])
        return history




    def plot_curves(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.show() 




    def prediction(self, val_essay, model_path=""):
        model_path = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/model-weights/"+model_path
        self.model.load_weights(model_path)
        pred_score, pred_bin = self.model.predict(val_essay)
        return pred_score




    def correlation(self, y_true, y_pred):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        return pearsonr(y_true, y_pred)[0]
        
        
        
        
