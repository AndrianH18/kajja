from numpy import mean
from numpy import std
from numpy import dstack
#from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
import json
import os

exercise = ['curl_1', 'curl_2', 'curl_3', 'pushup_1', 'pushup_2', 'pushup_3','squat_1', 'squat_2']
exercise_test = ['curl_1', 'curl_2', 'curl_3', 'pushup_1', 'pushup_2', 'squat_1', 'squat_2']

kpf = []
y = []
kptemp=[]
counter = 0
count_val = 20
n = 0

kptemp_test=[]
kpf_test=[]
y_test=[]


def load_exercise(exercise):
    global counter,kptemp,n
    counter=0
    filepath = 'train/' + exercise
    directory = len(os.listdir(filepath))
    for i in range(directory):
        filename = filepath + '/' + str(i) + '_keypoints.json'
        kp = load_json(filename)
        if kp:
            kptemp.append(kp)
            counter+=1
            if len(kptemp) == count_val:
                kpf.append(kptemp)
                kptemp=list()
                counter=0
                y.append(exercise.split('_')[0])

def load_json(filename):
	with open(filename, 'r') as f:
		keypoint=json.load(f)
		if keypoint["people"]:
			kp = keypoint["people"][0]["pose_keypoints_2d"]
			for i in range(74,0,-3):
				kp.pop(i)
			return kp
		else:
			return None


 
# load the dataset, returns train and test X and y elements
def load_dataset(dataset):
	for exercise in dataset:
		load_exercise(exercise)

def load_exercise_test(exercise):
    global counter,kptemp_test,n
    counter=0
    filepath = 'test/' + exercise
    directory = len(os.listdir(filepath))
    for i in range(directory):
        filename = filepath + '/' + str(i) + '_keypoints.json'
        kp = load_json(filename)
        if kp:
            kptemp_test.append(kp)
            counter+=1
            if len(kptemp_test) == count_val:
                kpf_test.append(kptemp_test)
                kptemp_test=list()
                counter=0
                y_test.append(exercise.split('_')[0])

 
# load the dataset, returns train and test X and y elements
def load_dataset_test(dataset):
	for exercise in dataset:
		load_exercise_test(exercise)

	
 
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 1, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
	model.add(Dropout(0.3))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(200, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
	model.save(f'models/{accuracy}')
	return accuracy
 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# load data
	train_y=[]
	test_y=[]
	global kpf, kpf_test, y, y_test
	load_dataset(exercise)
	kpf = dstack(kpf)
	kpf = tf.transpose(kpf, perm=[2, 0, 1])
	#kpf.resize(349,count_val,50)
	print(kpf.shape)
	
	# print(y)

	load_dataset_test(exercise_test)
	kpf_test = dstack(kpf_test)
	print(kpf_test.shape)
	kpf_test = tf.transpose(kpf_test, perm=[2, 0, 1])
	print(kpf_test.shape)
	# print(y_test)
	for i in range(len(y)):
		if y[i] == 'curl':
			train_y.append(0)
		elif y[i] == 'pushup':
			train_y.append(1)
		elif y[i] == 'squat':
			train_y.append(2)
		elif y[i] == 'none':
			train_y.append(3)
	
	for i in range(len(y_test)):
		if y_test[i] == 'curl':
			test_y.append(0)
		elif y_test[i] == 'pushup':
			test_y.append(1)
		elif y_test[i] == 'squat':
			test_y.append(2)
		elif y_test[i] == 'none':
			test_y.append(3)

	train_y = to_categorical(train_y)
	test_y = to_categorical(test_y)

	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(kpf, train_y, kpf_test, test_y)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment
run_experiment()
