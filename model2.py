import os
from xml.etree.ElementTree import ProcessingInstruction
import pandas as pd
import numpy as np
import math
import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping # NEW ADDED

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from numpy import array

import random
from matplotlib.figure import Figure
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import plotly.io as pio
import datetime

time_step = 15
num_epochs = 500 #2000 # 1000
unit = 32 # 32 # hidden neuron?
batchSize = 32 # 8

def preprocessing(maindf):
    global date2
    # # file masuk ke maindf dan di rename kolomnya
    # maindf = pd.read_csv(data_file)

    # DATA SELECTION
    maindf = maindf.rename(columns={'Date': 'date','Kasus Baru':'kasus_baru','Meninggal':'meninggal','Sembuh':'sembuh'})
    # di ubah format datenya
    maindf['date'] = pd.to_datetime(maindf.date)
    # dibuat variabel casef hanya khusus kolom kasus baru
    casef = maindf[['date','kasus_baru']]
    # CONSIDER ONLY CURRENT DATA FOR PREDICTION 
    casef = casef[casef['date'] > '2020-01-01']
    # close_stock = casef.copy()

    # DATA CLEANING
    date2 = casef['date']
    del casef['date']
    
    # DATA TRANSFORMATION
    # NORMALIZING CLOSE PRICE
    scaler=MinMaxScaler(feature_range=(0,1))
    casef=scaler.fit_transform(np.array(casef).reshape(-1,1))

    return casef, scaler

def training_preparation(casef):
    # PREPARE DATA FOR TRAIN AND TEST (perbandingan 8:2)
    training_size=int(len(casef)*0.80)  # trainingnya 80%
    test_size=len(casef)-training_size  # testingnya 
    train_data,test_data=casef[0:training_size,:],casef[training_size:len(casef),:1]
    # PENTING, TIME STEP KAYANYA BISA DI COBA2 LAGI NANTI KALO PREDIKSINYA KURANG BAGUS
    # cek di paling bawah ada fungsi def create_dataset()
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    return X_train, y_train, X_test, y_test, test_data


def build_model(X_train, y_train, X_test, y_test):
    # MODEL BUILDING
    # masih bisa di ganti Dropout, Dense, loss, optimizer, dkk biar makin akurat prediksinya
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(GRU(unit,return_sequences=True,input_shape=(time_step,1)))
    model.add(GRU(unit,return_sequences=True))
    model.add(GRU(unit))
    model.add(Dropout(0.20))
    model.add(Dense(1))
#     model.compile(loss='mean_squared_error',optimizer='adam')

#     history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1) # langsung di build
#     history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1000,batch_size=8,verbose=1) # langsung di build

#     history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=256,verbose=1) # langsung di build

#     model.compile(loss='mean_squared_error',optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    es_callback = EarlyStopping(monitor='val_loss', patience=3)
#     history = model.fit(X_train,y_train, batch_size=8,epochs=num_epochs, validation_data=(X_test,y_test), callbacks=[es_callback], shuffle=False, verbose=1)
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=num_epochs,batch_size=batchSize,verbose=1) # langsung di build

    
    # tercatat berapa loss dan val_loss nya
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # nah ini gatau knp epochs diambil dari len(loss)
    epochs = range(len(loss))
    # plotting loss dan val_loss nya manatau diperlukan
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    # plt.figure()
    # plt.write_image("static/images/predictionLoss.png")   # ternyata write_image utk plotly, bukan matplotlib
    plt.savefig('static/images/predictionLoss.png')


    # plt.show()

    # gayakin model bisa dipisah fungsi (training dan predict)
    return loss, val_loss, epochs, model


def model_evaluation(model, scaler, casef, X_train, X_test, y_train, y_test):
    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    # train_predict.shape, test_predict.shape       # shape utk tau total data nya

    # MODEL EVALUATION => persiapan utk tau RMSE, MSE dan MAE serta R2 score
    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
    # # Evaluation metrices RMSE and MAE
    # print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
    # print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
    # print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
    # print("-------------------------------------------------------------------------------------")
    # print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
    # print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
    # print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))

    # print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
    # print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
    # print("Train data R2 score:", r2_score(original_ytrain, train_predict))
    # print("Test data R2 score:", r2_score(original_ytest, test_predict))
    train_rmse = math.sqrt(mean_squared_error(original_ytrain,train_predict))
    train_mse = mean_squared_error(original_ytrain,train_predict)
    train_mae = mean_absolute_error(original_ytrain,train_predict)
    test_rmse = math.sqrt(mean_squared_error(original_ytest,test_predict))
    test_mse = mean_squared_error(original_ytest,test_predict)
    test_mae = mean_absolute_error(original_ytest,test_predict)
    train_variance = explained_variance_score(original_ytrain, train_predict)
    test_variance = explained_variance_score(original_ytest, test_predict)
    train_r2 = r2_score(original_ytrain, train_predict)
    test_r2 = r2_score(original_ytest, test_predict)

    

    # COMPARISON OF ORIGINAL STOCK PRICE AND PREDICTED CLOSE PRICE
    # shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(casef)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(casef)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(casef)-1, :] = test_predict
    # print("Test predicted data: ", testPredictPlot.shape)

    names = cycle(['Actual case','Train predicted case','Test predicted case'])

    return train_rmse, train_mse, train_mae, test_rmse, test_mse, test_mae, train_variance, test_variance, train_r2, test_r2

def predict(test_data, model, scaler, casef):
    # PREDICTING NEXT 14 DAYS
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 14
    while(i<pred_days):
        
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
                
    # print("Output of predicted next days: ", len(lst_output))

    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    # print(last_days)
    # print(day_pred)
    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(casef[len(casef)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })

    names = cycle(['Last 15 days close price','Predicted next 14 days close price'])

    end_date = date2 + datetime.timedelta(days=14)

    fig = px.line(new_pred_plot,x=end_date[-30:], y=[new_pred_plot['last_original_days_value'],
                                                        new_pred_plot['next_predicted_days_value']],
                labels={'value': 'New Covid Cases','index': 'Timestamp'})
    fig.update_layout(title_text='Compare last 14 days vs next 14 days',
                    plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Covid Cases')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.write_image("static/images/covidPrediction.png")

    # PLOTTING ENTIRE DATA WITH PREDICTION
    lstmdf=casef.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['COVID Case'])

    # PLOTTING HASIL PREDIKSI
    fig = px.line(lstmdf,labels={'value': 'COVID new case','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole COVID cases with prediction (14 days)',
                      plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='COVID')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # fig.show()
    fig.write_image("static/images/covidPrediction2.png")


    # PLOTTING HASIL PREDIKSI DIGABUNG DATA ASLI (ALL)
    # PLOTTING ENTIRE DATA WITH PREDICTION
    # lstmdf=casef.tolist()
    # lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    # lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    # names = cycle(['COVID Case'])

    # fig = px.line(lstmdf,labels={'value': 'COVID new case','index': 'Timestamp'})
    # fig.update_layout(title_text='Plotting whole COVID cases with prediction (14 days)',
    #                 plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='COVID')

    # fig.for_each_trace(lambda t:  t.update(name = next(names)))

    # fig.update_xaxes(showgrid=False)
    # fig.update_yaxes(showgrid=False)
    # fig.show()

    # PLOTTING ENTIRE DATA WITH PREDICTION
    # lstmdf adalah list utk gabungin data asli dan prediksi
    lstmdf=casef.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    lstmdf=scaler.inverse_transform(lstmdf).reshape(1,-1).tolist()[0]

    names = cycle(['COVID Case'])

    # versi date
    dateAll = np.append(date2,end_date[-14:])
    fig = px.line(x=dateAll,y=lstmdf,labels={'value': 'COVID new case','index': 'Timestamp'})
    fig.update_layout(title_text='Plotting whole COVID cases with prediction (14 days)',
                    plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='COVID')

    ## versi time stamp, bukan date
    # fig = px.line(lstmdf,labels={'value': 'COVID new case','index': 'Timestamp'})
    # fig.update_layout(title_text='Plotting whole COVID cases with prediction (14 days)',
    #                 plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='COVID')

    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # fig.show()

    return lstmdf, fig, end_date

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)



# # APP.PY
# maindf = pd.read_csv('covid_sumut.csv')
# # maindf = pd.read_csv(file)

# # preprocessing
# casef, scaler = preprocessing(maindf)

# # prepare for training 
# X_train, y_train, X_test, y_test, test_data = training_preparation(casef)

# # lets start building the model
# loss, val_loss, epochs, model = build_model(X_train, y_train, X_test, y_test)

# # evaluate the model 
# train_rmse, train_mse, train_mae, test_rmse, test_mse, test_mae, train_variance, test_variance, train_r2, test_r2 = model_evaluation(model, scaler, casef, X_train, X_test, y_train, y_test)

# # start predicting
# lstmdf = predict(test_data, model, scaler, casef)	# lstmdf is real data and prediction combined




# # NOT INCLUDED, JUST PRINTED
# print(train_rmse)
# print(train_mse)
# print(train_mae)
# print(test_rmse)
# print(test_mse)
# print(test_mae)
# print(train_variance)
# print(test_variance)
# print(train_r2)
# print(test_r2)

# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig
