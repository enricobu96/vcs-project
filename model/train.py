import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
        print('DONE')
        sys.stdout.flush()

        """
        TRAIN THE ML MODEL
        Train the ML model, evaluate and serialize
        """
        print('Training the model...')
        sys.stdout.flush()
        pipelines = {
            'lr':make_pipeline(StandardScaler(), LogisticRegression()),
            'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
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
        print('ACCURACY')
        sys.stdout.flush()
        for alg, model in fit_models.items():
            _y = model.predict(X_test)
            print(alg, accuracy_score(y_test, _y))
