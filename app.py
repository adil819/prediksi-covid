from flask import Flask, render_template, request, jsonify
from flask import send_file
from flask import redirect
import requests

import pandas as pd
# import random
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
from flask import Flask
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# import model as ml
import model2 as ml2
# import testing as ts
app = Flask(__name__, template_folder='templates')
app.run('0.0.0.0', 8085)
# app.run(debug=False)
# app.run(debug=True)

# lstmdf[-14:]
# close_stock['date'][len(close_stock['date'])-1]
# end_date = close_stock['date'] + datetime.timedelta(days=10)
# end_date[-30:]

@app.route('/')
def home():
	file1 = open("modelEvaluation.txt","r+")
	file1.seek(0)
	# angka = file1.readline()
	# angka2 = file1.readline()

	train_rmse = file1.readline()[:-4]
	train_mse = file1.readline()[:-4]
	train_mae = file1.readline()[:-4]
	train_variance = file1.readline()[:-4]
	train_r2 = file1.readline()[:-4]
	test_rmse = file1.readline()[:-4]
	test_mse = file1.readline()[:-4]
	test_mae = file1.readline()[:-4]
	test_variance = file1.readline()[:-4]
	test_r2 = file1.readline()[:-4]
	date_now = file1.readline()
	temppp = file1.readline()	# temppp read useless lines
	date_first_prediction = file1.readline()
	temppp = file1.readline()	
	date_last_prediction = file1.readline()

	file1.close()

	resp_sumut = requests.get('https://data.covid19.go.id/public/api/prov_detail_SUMATERA_UTARA.json')
	cov_sumut_raw = resp_sumut.json()

	cov_sumut = pd.DataFrame(cov_sumut_raw['list_perkembangan'])

	# Menjinakkan Data
	cov_sumut_tidy = (cov_sumut.drop(columns=[item for item in cov_sumut.columns
	if item.startswith('AKUMULASI')
	or item.startswith('DIRAWAT')])
	.rename(columns=str.lower)
	.rename(columns={'kasus': 'kasus_baru'})
	)
	cov_sumut_tidy['tanggal'] = pd.to_datetime(cov_sumut_tidy['tanggal']*1e6, unit='ns')

	dataForCSV = pd.DataFrame(cov_sumut_tidy)
	dataForCSV = dataForCSV.rename(columns={"tanggal":"Date"})
	dataForCSV = dataForCSV.rename(columns={"kasus_baru":"Kasus Baru"})
	dataForCSV = dataForCSV.rename(columns={"meninggal":"Meninggal"})
	dataForCSV = dataForCSV.rename(columns={"sembuh":"Sembuh"})
	dataForCSV['Date'] = dataForCSV['Date'].dt.strftime('%m/%d/%Y')
	dataForCSV.to_csv('data_covid_sumut_newest.csv', index = False) 
	# cov_sumut_raw['last_date']

	# return render_template('home.html')
	return render_template('home.html', date_updated_data=cov_sumut_raw['data']['last_update'], date_now=date_now[6:-1], date_first_prediction=date_first_prediction[6:-1], date_last_prediction=date_last_prediction[6:-1], train_rmse=train_rmse, train_mse=train_mse, train_mae=train_mae, test_rmse=test_rmse, test_mse=test_mse, test_mae=test_mae, train_variance=train_variance, test_variance=test_variance, train_r2=train_r2, test_r2=test_r2)


@app.route('/predict', methods=['POST','GET'])
def predict():
	global fig
	
	if request.method=="POST":
		file = request.files['file_train']
			
		# maindf = pd.read_csv('covid_sumut.csv')
		maindf = pd.read_csv(file)

		# preprocessing
		casef, scaler = ml2.preprocessing(maindf)

		# prepare for training 
		X_train, y_train, X_test, y_test, test_data = ml2.training_preparation(casef)

		# lets start building the model
		loss, val_loss, epochs, model = ml2.build_model(X_train, y_train, X_test, y_test)

		# evaluate the model 
		train_rmse, train_mse, train_mae, test_rmse, test_mse, test_mae, train_variance, test_variance, train_r2, test_r2 = ml2.model_evaluation(model, scaler, casef, X_train, X_test, y_train, y_test)

		# start predicting
		lstmdf, fig, end_date = ml2.predict(test_data, model, scaler, casef)	# lstmdf is real data and prediction combined

		train_rmse = round(train_rmse, 4)
		train_mse = round(train_mse, 4)
		train_mae = round(train_mae, 4)
		train_variance = round(train_variance, 4)
		train_r2 = round(train_r2, 4)
		test_rmse = round(test_rmse, 4)
		test_mse = round(test_mse, 4)
		test_mae = round(test_mae, 4)
		test_variance = round(test_variance, 4)
		test_r2 = round(test_r2, 4)

		with open('modelEvaluation.txt', 'w') as f:
			f.write('%f \n' % train_rmse)
			f.write('%f \n' % train_mse)
			f.write('%f \n' % train_mae)
			f.write('%f \n' % train_variance)
			f.write('%f \n' % train_r2)
			f.write('%f \n' % test_rmse)
			f.write('%f \n' % test_mse)
			f.write('%f \n' % test_mae)
			f.write('%f \n' % test_variance)
			f.write('%f \n' % test_r2)
			f.write('%s \n' % end_date[-15:-14])
			f.write('%s \n' % end_date[-14:-13])
			f.write('%s \n' % end_date[-1:])
			

	return redirect("http://127.0.0.1:8085/#predict", code=302) # 302 means URL redirection

	# return render_template('home.html')
	# return jsonify({})
	# return home()
	

	# return render_template('home.html', train_rmse=train_rmse, train_mse=train_mse, train_mae=train_mae, test_rmse=test_rmse, test_mse=test_mse, test_mae=test_mae, train_variance=train_variance, test_variance=test_variance, train_r2=train_r2, test_r2=test_r2)


@app.route('/predict2', methods=['POST','GET'])
def predict2():
	global fig
	
	if request.method=="POST":
		# file = request.files['file_train']
			
		maindf = pd.read_csv('data_covid_sumut_newest.csv')
		# maindf = pd.read_csv(file)

		# preprocessing
		casef, scaler = ml2.preprocessing(maindf)

		# prepare for training 
		X_train, y_train, X_test, y_test, test_data = ml2.training_preparation(casef)

		# lets start building the model
		loss, val_loss, epochs, model = ml2.build_model(X_train, y_train, X_test, y_test)

		# evaluate the model 
		train_rmse, train_mse, train_mae, test_rmse, test_mse, test_mae, train_variance, test_variance, train_r2, test_r2 = ml2.model_evaluation(model, scaler, casef, X_train, X_test, y_train, y_test)

		# start predicting
		lstmdf, fig, end_date = ml2.predict(test_data, model, scaler, casef)	# lstmdf is real data and prediction combined

		train_rmse = round(train_rmse, 4)
		train_mse = round(train_mse, 4)
		train_mae = round(train_mae, 4)
		train_variance = round(train_variance, 4)
		train_r2 = round(train_r2, 4)
		test_rmse = round(test_rmse, 4)
		test_mse = round(test_mse, 4)
		test_mae = round(test_mae, 4)
		test_variance = round(test_variance, 4)
		test_r2 = round(test_r2, 4)

		with open('modelEvaluation.txt', 'w') as f:
			f.write('%f \n' % train_rmse)
			f.write('%f \n' % train_mse)
			f.write('%f \n' % train_mae)
			f.write('%f \n' % train_variance)
			f.write('%f \n' % train_r2)
			f.write('%f \n' % test_rmse)
			f.write('%f \n' % test_mse)
			f.write('%f \n' % test_mae)
			f.write('%f \n' % test_variance)
			f.write('%f \n' % test_r2)
			f.write('%s \n' % end_date[-15:-14])
			f.write('%s \n' % end_date[-14:-13])
			f.write('%s \n' % end_date[-1:])
			

	return redirect("http://127.0.0.1:8085/#predict", code=302) # 302 means URL redirection

	# return render_template('home.html')
	# return jsonify({})
	# return home()
	

	# return render_template('home.html', train_rmse=train_rmse, train_mse=train_mse, train_mae=train_mae, test_rmse=test_rmse, test_mse=test_mse, test_mae=test_mae, train_variance=train_variance, test_variance=test_variance, train_r2=train_r2, test_r2=test_r2)


@app.route('/plot')
def plot_png():
	fig.show()
	return render_template('home.html')

# @app.route('/plot.png')
# def plot_png():
#     # fig = ml2.create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return send_file(output.getvalue(), mimetype='image/png')


# @app.route('/training', methods=['POST','GET'])
# def training():
# 	if request.method=="POST":
# 		file = request.files['file_train']
# 		#read data
# 		raw_docs_train, raw_docs_val, y_train, y_val = ml.read_data(file)

# 		#preprocessing
# 		processed_docs_train = ml.preprocessing(raw_docs_train)
# 		processed_docs_val= ml.preprocessing(raw_docs_val)

# 		#tokenization
# 		word_index, word_seq_train, word_seq_val = ml.tokenization_training(processed_docs_train, processed_docs_val)

# 		#model
# 		loss_plot, acc_plot, loss_train, accuracy_train, loss_val, accuracy_val = ml.model_building(word_index, word_seq_train, word_seq_val, y_train, y_val)

# 		#step preproccessing untuk interface
# 		step_preproccess = ml.step_preproccess(raw_docs_train)

# 		text_input = str(raw_docs_train[2])
# 		text_case_folding = raw_docs_train[2].lower()
# 		text_stopword = step_preproccess[0].lower()
# 		text_punct = step_preproccess[1].lower()
# 		text_tokenisasi = str(word_seq_train[2])
	
# 		#accuracy = tr.predict(model, word_seq_train, y_train)
# 	return jsonify({'loss_plot':loss_plot,'acc_plot':acc_plot,'loss_train':loss_train, 'accuracy_train':accuracy_train,'loss_val':loss_val, 'accuracy_val':accuracy_val, 'text_input':text_input, 'text_case_folding':text_case_folding, 'text_stopword':text_stopword, 'text_punct':text_punct, 'text_tokenisasi':text_tokenisasi})
# 		# return render_template('home.html', message=accuracy, loss_plot=loss_plot, acc_plot=acc_plot)

# @app.route('/testing', methods=['POST','GET'])
# def testing():
# 	if request.method=="POST":
# 		file = request.files['file_test']
# 		accuracy, output_pred = ml.testing(file)

# 		data_output = []
# 		for i in range (len(output_pred)):
# 			data_output.append({
# 					'id': str(i+1),
#                     'title': str(output_pred['title'][i]),
#                     'label_score': str(output_pred['label_score'][i]),
#                     'prediction': str(output_pred['pred'][i]),
#                 })
# 	return jsonify({'accuracy':accuracy, 'data_output':data_output})
# 	#return jsonify({'accuracy':accuracy})

# @app.route('/predict', methods=['POST','GET'])
# def predict():
# 	if request.method=="POST":
# 		data_pred = request.form['data_pred']
# 		hasil_pred = ml.predict(data_pred)
		
# 	return jsonify({'hasil_pred':hasil_pred})
# 	#return jsonify({'accuracy':accuracy})


if __name__ == "__main__":
	app.run('0.0.0.0', 8085)
	# app.run(debug=True)
	# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1