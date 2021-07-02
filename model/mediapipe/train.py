import mediapipe as mp
import cv2
import csv
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# 4 diversi modelli di classificazione
# TODO fare un confronto di questi in termini di precision and recall
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

class Train:
    def train(self):
        """
        READ CSV AND PREPARE
        Read the csv and prepare the model to predict
        """
        print('Loading dataset...', end='')
        sys.stdout.flush()
        df = pd.read_csv('./dataset/keypoints/coords.csv')
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

        # Test the model
        self.test_accuracy(fit_models, X_test, y_test)

        # Serialize model
        with open('./model/mediapipe/prediction_models/prediction_model.pkl', 'wb') as f:
            pickle.dump(fit_models['rf'], f)

    def test_accuracy(self, fit_models, X_test, y_test):
        print('ACCURACY')
        sys.stdout.flush()
        for alg, model in fit_models.items():
            _y = model.predict(X_test)
            print(alg, accuracy_score(y_test, _y))
