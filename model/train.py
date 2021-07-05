import sys
import pandas as pd
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as skm
from sklearn.metrics import precision_recall_fscore_support as score
import pickle
from matplotlib import pyplot as plt
# import cmat2scores

class Train:
    
    def train(self, is_nite: bool):
        """
        READ CSV AND PREPARE
        Read the csv and prepare the model to predict
        """
        print('Loading dataset...', end='')
        sys.stdout.flush()
        if not is_nite:
            df = pd.read_csv('./dataset/keypoints/coords_mediapipe.csv')
        else:
            df = pd.read_csv('./dataset/keypoints/coords_kinect.csv')
        X = df.drop('class', axis=1) # features
        y = df['class'] # target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)
        print('DONE')
        sys.stdout.flush()

        """
        TRAIN THE ML MODEL
        Train the ML model, evaluate and serialize
        """
        print('Training the model...')
        sys.stdout.flush()
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression(C=1e-4, max_iter=200)),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=33, max_depth=1)),
            #'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
            'svm': make_pipeline(StandardScaler(), SVC(probability=True)),
            'cnn': make_pipeline(StandardScaler(), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1))
        }
        fit_models = {}
        for alg, pipeline in pipelines.items():
            print("Training", alg, "...", end='')
            sys.stdout.flush()
            model = pipeline.fit(X_train, y_train)
            fit_models[alg] = model
            print('DONE')
            sys.stdout.flush()

        print('Training the model...DONE')
        sys.stdout.flush()

        # Test the models
        self.test_accuracy(fit_models, X_test, y_test)

        # Serialize models
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

    def test_accuracy(self, fit_models, X_test, y_test):
        xy = ['dab', 'tp', 'rarmm', 'rarmt', 'larmm', 'larmt', 'st']
        out = ''
        for alg, model in fit_models.items():
            out += alg + '\n\n'
            y_predict = model.predict(X_test)
            out += skm.classification_report(y_test, y_predict)
            c_matrix = skm.confusion_matrix(y_test, y_predict)
            disp = plot_confusion_matrix(model, X_test, y_test, display_labels=xy, cmap='plasma')
            plt.savefig('./extra/test_reports/conf_matrix_' + alg + '.png')
            #plt.show()
            out += '\n'

        #Save to file
        with open('./extra/test_reports/test_reports.txt', mode='w', newline='') as f:
            f.write(out)

        


        
