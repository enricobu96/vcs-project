import sys
import pandas as pd
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as skm
import pickle
from matplotlib import pyplot as plt
import numpy as np

"""
CLASSIFICATION ALGORITHM PARAMETERS
Parameters for the different classification algorithms
"""
LR_C = 1e-2
LR_MAX_ITER = 150
RC_ALPHA = 1e2
RF_N_ESTIMATORS = 30
RF_MAX_DEPTH = 3
RF_MIN_SAMPLES_LEAF = 10
SVM_MAX_ITER = 200
SVM_C = 1e-1
MLP_ALPHA = 2e1
MLP_HIDDEN_LAYER_SIZES = (32,)

"""
PLOT PARAMETERS
Parameters for the learning curves plot
"""
LEARNING_CURVE_STEPS = 5
CPU_THREADS = 22

class Train:
    
    def train(self, is_nite: bool):

        """
        READ CSV AND PREPARE
        Read the csv and prepare the model(s) to predict
        """
        print('Loading dataset...', end='')
        sys.stdout.flush()
        if not is_nite:
            df = pd.read_csv('./dataset/keypoints/coords_mediapipe.csv')
        else:
            df = pd.read_csv('./dataset/keypoints/coords_kinect.csv')
        df = shuffle(df)
        X = df.drop('class', axis=1) # features
        y = df['class'] # target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)
        print('DONE')
        sys.stdout.flush()

        """
        TRAIN THE ML MODEL(S)
        Train the ML model(s), evaluate and serialize
        """
        print('Training the model...')
        sys.stdout.flush()
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER)), 
            'rc':make_pipeline(StandardScaler(), RidgeClassifier(alpha=RC_ALPHA)),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_SAMPLES_LEAF)),
            'svm': make_pipeline(StandardScaler(), SVC(probability=True, max_iter=SVM_MAX_ITER, C=SVM_C)),
            'mlp': make_pipeline(StandardScaler(), MLPClassifier(alpha=MLP_ALPHA, random_state=42, hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES))
        }
        fit_models = {}
        for alg, pipeline in pipelines.items():
            print("Training", alg, "...", end='')
            sys.stdout.flush()
            model = pipeline.fit(X_train, y_train)
            fit_models[alg] = model
            print('DONE')
            sys.stdout.flush()

        print('Training the model(s)...DONE')
        sys.stdout.flush()

        """
        MODEL TESTING
        Test the models and get stats and charts
        """
        print('Assessing model(s) accuracy...')
        sys.stdout.flush()
        # Accuracy
        self.test_accuracy(fit_models, X_test, y_test)
        # Confusion matrices
        self.print_confusion_matrix('lr', fit_models['lr'], X_test, y_test)
        self.print_confusion_matrix('rc', fit_models['rc'], X_test, y_test)
        self.print_confusion_matrix('rf', fit_models['rf'], X_test, y_test)
        self.print_confusion_matrix('svm', fit_models['svm'], X_test, y_test)
        self.print_confusion_matrix('mlp', fit_models['mlp'], X_test, y_test)
        # Learning curves
        self.print_learning_curves('lr', fit_models['lr'], X_test, y_test, X_train, y_train)
        self.print_learning_curves('rc', fit_models['rc'], X_test, y_test, X_train, y_train)
        self.print_learning_curves('rf', fit_models['rf'], X_test, y_test, X_train, y_train)
        self.print_learning_curves('svm', fit_models['svm'], X_test, y_test, X_train, y_train)
        self.print_learning_curves('mlp', fit_models['mlp'], X_test, y_test, X_train, y_train)

        """
        SERIALIZE MODELS
        Save trained model(s) into pickle files
        """
        if not is_nite:
            for k in pipelines.keys():
                fileName = ('./model/mediapipe/prediction_models/prediction_model_' + k + '.pkl')
                with open(fileName, 'wb') as f:
                    pickle.dump(fit_models[k], f)
        else:
            for k in pipelines.keys():
                fileName = ('./model/kinect_depth/prediction_models/prediction_model_' + k + '.pkl')
                with open(fileName, 'wb') as f:
                    pickle.dump(fit_models[k], f)

    """
    TEST ACCURACY
    Get accuracy, precision, recall, F1 score and support for every class in every model. Save results into file
    """
    def test_accuracy(self, fit_models, X_test, y_test):
        out = ''
        for alg, model in fit_models.items():
            out += alg + '\n\n'
            y_predict = model.predict(X_test)
            out += skm.classification_report(y_test, y_predict)
            out += '\n'
        with open('./extra/test_reports/test_reports.txt', mode='w', newline='') as f:
            f.write(out)

    """
    CONFUSION MATRIX
    Calculate confusion matrices for every class in every model. Save results into file(s)
    """
    def print_confusion_matrix(self, alg, model, X_test, y_test):
        fig = plt.figure()
        # xy = ['balc', 'bend', 'boxx', 'clap', 'marc', 'onew', 'wave'] # x,y labels for APE dataset
        xy = ['dab', 'tp', 'rarmm', 'rarmt', 'larmm', 'larmt', 'st'] # x,y labels for custom dataset
        y_predict = model.predict(X_test)
        c_matrix = skm.confusion_matrix(y_test, y_predict)
        disp = plot_confusion_matrix(model, X_test, y_test, display_labels=xy, cmap='plasma')
        plt.xlabel('Predicted label')
        plt.savefig('./extra/test_reports/conf_matrix_' + alg + '.png')

    """
    LEARNING CURVES
    Compute learning curves for every model. Save results into file(s)
    """
    def print_learning_curves(self, alg, model, X_test, y_test, X_train, y_train):
        fig = plt.figure()
        train_sizes, train_scores, validation_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1,1.0,LEARNING_CURVE_STEPS), n_jobs=CPU_THREADS)
        train_scores_mean = train_scores.mean(axis = 1)
        validation_scores_mean = validation_scores.mean(axis = 1)
        plt.plot(train_sizes, train_scores_mean, train_sizes, validation_scores_mean)
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(['Training score', 'Validation score'])
        plt.savefig('./extra/test_reports/learning_' + alg + '.png')