import tensorflow.keras as k
import chardet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import sqlite3 as sql

target_predict=[]
def text_checker(text):
    import tensorflow.keras as k
    import chardet
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    num_word=1000
    import numpy as np
    shape=159572

    xl=pd.read_csv('train.csv')
    x_train=xl['comment_text'][0:shape].values.reshape(-1,1)
    y_train=xl[['toxic','severe_toxic','obscene','threat','insult','identity_hate']][0:shape].values

    x_train_mod=[]
    for add in x_train[0:]:
        x_train_mod.append("{}".format(add[0]))


    tokenizer = k.preprocessing.text.Tokenizer(num_words=num_word)

    tokenizer.fit_on_texts(x_train_mod)
    sequences = tokenizer.texts_to_sequences(x_train_mod)
    x_train_new = k.preprocessing.sequence.pad_sequences(sequences, maxlen=num_word)



    ##
    model = k.Sequential()
    model.add(k.layers.Dense(1000, activation='relu',input_shape=(num_word,)))
    model.add(k.layers.Dropout(.3))
    model.add(k.layers.Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam',
                      loss='msle',
                      metrics=['accuracy'])
    model.fit(x_train_new,y_train,epochs=2,validation_steps=30)



    #predict



    text_ = [text]
    tokenizer = k.preprocessing.text.Tokenizer(num_words=num_word)

    tokenizer.fit_on_texts(text_)
    sequences = tokenizer.texts_to_sequences(text_)
    x_predict_new = k.preprocessing.sequence.pad_sequences(sequences, maxlen=num_word)
    predict=model.predict(x_predict_new)
    # CLASSIFY PREDICT
    last_predict=[]
    for class_ in predict[0][0:]:
        if class_>=float(0.100000000):
            last_predict.append(class_)
        else:
            last_predict.append(0)
    final_shape_result=[]
    if last_predict.count(0)==6 or last_predict.count("0")==6:
        max=np.max(predict[0])
        n=0
        for max_ in predict[0][0:]:

            if max_==max:
                 final_shape_result.append(max)

            else:
                final_shape_result.append(0)


            n=n+1
        print(final_shape_result)
        if len(final_shape_result) ==0:
            final_shape_result=[0,0,0,0,0,0]
            print(final_shape_result)
            return final_shape_result
        return final_shape_result
    else:
        if len(final_shape_result) ==0:
            final_shape_result=[0,0,0,0,0,0]
            print(final_shape_result)
            return final_shape_result

        print(last_predict)
        return last_predict

text_checker("i love you")

