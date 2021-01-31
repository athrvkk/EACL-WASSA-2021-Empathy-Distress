# File contatins Class BertModel
# File: bert_model.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

class BertModel():
    """ Implementation of BERT for regression """


    def __init__(self, bert_model="BERT", output_hidden_states=True, activation="relu", cpkt=""):
        """ Constructor to initialize Deep learning models
        @param model (str): the model to be used (DNN, CNN, BiLSTM, BERT, DistilBERT, RoBERTa).
        @param output_hidden_states (bool): Whether to output the hidden states of BERT.
        """
        if activation == "leaky_relu":
            self.activation = LeakyReLU()
        elif activation == "paramaterized_leaky_relu":
            self.activation = PReLU()           
        else:
            self.activation = "relu"

        if bert_model == "BERT":
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=output_hidden_states)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            self.bert = TFBertModel.from_pretrained("bert-base-uncased", config=config)

        elif bert_model == "DistilBERT":
            config = BertConfig.from_pretrained("distilbert-base-uncased", output_hidden_states=output_hidden_states)
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
            self.bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        elif bert_model == "RoBERTa":
            config = BertConfig.from_pretrained("roberta-base", output_hidden_states=output_hidden_states)
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
            self.bert = TFRobertaModel.from_pretrained("roberta-base", config=config)
        
        # ModelCheckPoint Callback:
        checkpoint_filepath = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/model-weights/"+ cpkt + "-epoch-{epoch:02d}-val-loss-{val_loss:02f}.h5"
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
                                                    verbose=2)
        # Early Stopping
        self.early_stopping = EarlyStopping(monitor='val_loss', 
                                            patience=20)





    def prepare_bert_input(self, corpus, max_len=200):
        """ Function to prepare input data for BERT.
        @param corpus (list): Dataset to be processed.
        @param max_len (int): maximum length of input post. Texts longer that max_len will be truncated, while shorter than max_len will be padded.
        """
        input_ids = []
        attention_masks = []
        
        for record in corpus:
            encoded_text = self.tokenizer.encode_plus(text=record,
                                                      add_special_tokens=True,
                                                      return_attention_mask=True,
                                                      max_length=max_len,
                                                      pad_to_max_length=True,
                                                      truncation=True)
            input_ids.append(encoded_text.get("input_ids"))
            attention_masks.append(encoded_text.get("attention_mask"))

        return np.array(input_ids, dtype="int32"), np.array(attention_masks, dtype="int32")





    def build(self, input_ids_dim, attention_mask_dim, fc1_dim=512, fc2_dim=128, d=0.2, l2_rate=0.001):
        """ Function to define the forward pass of the model.
        @param input (tensor): input to the model.
        """
        input_ids = Input(shape=(input_ids_dim,), name='input_ids', dtype="int32")
        attention_mask = Input(shape=(attention_mask_dim,), name='attention_mask', dtype="int32")
        base_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        base_output = GlobalAveragePooling1D()(base_output)
        base_output = Dense(fc1_dim, activation=self.activation, kernel_regularizer=l2(l2_rate))(base_output)
        base_output = Dropout(d)(base_output)
        base_output = Dense(fc2_dim, activation=self.activation, kernel_regularizer=l2(l2_rate))(base_output)
        base_output = Dropout(d)(base_output)
        
        x = Dense(32, self.activation, kernel_regularizer=l2(l2_rate))(base_output)
        empathy_bin = Dense(1, activation='sigmoid', name='empathy_bin_output')(x)

        x = Dense(32, self.activation, kernel_regularizer=l2(l2_rate))(base_output)
        distress_bin = Dense(1, activation='sigmoid', name='distress_bin_output')(x)

        x = Dropout(d)(x)
        x = Dense(16, self.activation, kernel_regularizer=l2(l2_rate))(x)
        empathy_score = Dense(1, name='empathy_score_output')(x)
        
        self.model = Model(inputs=[input_ids, attention_mask], outputs=[empathy_bin, distress_bin, empathy_score])
        self.model.layers[2].trainable = False
        self.model.compile(optimizer=Adam(lr=0.001), loss={"empathy_bin_output":"binary_crossentropy",                                                           
                                                           "distress_bin_output":"binary_crossentropy",
                                                           "empathy_score_output":"mse"})
        self.model.summary()
        




    def plot_model_arch(self):
        return plot_model(self.model, show_shapes=True)




    def train(self, x_train, y_train, x_val, y_val, epochs=200, batch_size=32):
        history = self.model.fit(x_train,
                                 y_train, 
                                 epochs=epochs, 
                                 batch_size=batch_size, 
                                 verbose=1, 
                                 validation_data = (x_val, y_val),
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
        _,  _, pred_score = self.model.predict(val_essay)
        return pred_score




    def correlation(self, y_true, y_pred):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        return pearsonr(y_true, y_pred)[0]
