#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
from my_preprocessing import *
from my_model import *

import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    ########################################################################### My Custom Processing
    wave_info = list()
    wave_info2 = list()
    demo_features = list()
    murmurs = list()
    outcomes = list()
    # Get Data
    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = custom_load_recordings(data_folder, current_patient_data)
        current_recordings2 = load_recordings(data_folder, current_patient_data)
        current_features = custom_get_features(current_patient_data, current_recordings2)

        # Extract features.
        wave_info.append(current_recordings)
        wave_info2.append(current_recordings)
        demo_features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
    
    demo_features = np.vstack(demo_features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)
    
    # Get Segmentation data
    for i in range(0, len(wave_info)): # Use only segmentation part
        n_loc = len(wave_info[i])
        for j in range(0, n_loc):
            tsv = wave_info[i][j][1]
            tsv = tsv[tsv.seg != 0].reset_index(drop = True)
            start = int(tsv.start[0]*4000)
            end = int(tsv.end[len(tsv)-1]*4000)
            wave_info[i][j][0] = wave_info[i][j][0][start:end]
    
    # Get Data per location & label
    pos_pos = []
    pos_neg = []
    unk_pos = []
    unk_neg = []
    neg_pos = []
    neg_neg = []
    for i in range(0, len(murmurs)):
        temp_mur = murmurs[i]
        temp_ab = outcomes[i]
        if temp_mur[0] == 1:
            if temp_ab[0] == 1:
                pos_pos.append(i)
            if temp_ab[1] == 1:
                pos_neg.append(i)

        if temp_mur[1] == 1:
            if temp_ab[0] == 1:
                unk_pos.append(i)
            if temp_ab[1] == 1:
                unk_neg.append(i)

        if temp_mur[2] == 1:
            if temp_ab[0] == 1:
                neg_pos.append(i)
            if temp_ab[1] == 1:
                neg_neg.append(i)
    
    pos_train, pos_valid = train_test_split(pos_pos + pos_neg, test_size = 68, random_state = 42)
    neg_train, neg_valid = train_test_split(neg_neg, test_size = 263, random_state = 42)

    train = pos_train + neg_train
    valid = pos_valid + neg_valid
    test = valid + unk_pos + unk_neg + neg_pos
    
    train_AV_x = []
    train_PV_x = []
    train_TV_x = []
    train_MV_x = []

    train_AV_y = []
    train_PV_y = []
    train_TV_y = []
    train_MV_y = []

    for i in train:
        temp_info = wave_info[i]
        n_loc = len(temp_info)
        for j in range(0, n_loc):
            x = temp_info[j][0]
            loc = temp_info[j][2]
            y = temp_info[j][3]
            if loc == 'AV':
                train_AV_x.append(x)
                train_AV_y.append(y)
            if loc == 'PV':
                train_PV_x.append(x)
                train_PV_y.append(y)
            if loc == 'TV':
                train_TV_x.append(x)
                train_TV_y.append(y)
            if loc == 'MV':
                train_MV_x.append(x)
                train_MV_y.append(y)
    
    valid_AV_x = []
    valid_PV_x = []
    valid_TV_x = []
    valid_MV_x = []

    valid_AV_y = []
    valid_PV_y = []
    valid_TV_y = []
    valid_MV_y = []

    for i in valid:
        temp_info = wave_info[i]
        n_loc = len(temp_info)
        for j in range(0, n_loc):
            x = temp_info[j][0]
            loc = temp_info[j][2]
            y = temp_info[j][3]
            if loc == 'AV':
                valid_AV_x.append(x)
                valid_AV_y.append(y)
            if loc == 'PV':
                valid_PV_x.append(x)
                valid_PV_y.append(y)
            if loc == 'TV':
                valid_TV_x.append(x)
                valid_TV_y.append(y)
            if loc == 'MV':
                valid_MV_x.append(x)
                valid_MV_y.append(y)
    # Get Training-Validation Data
    AV_train_seg_x, AV_train_seg_y = gen_inputs(train_AV_x, train_AV_y)
    PV_train_seg_x, PV_train_seg_y = gen_inputs(train_PV_x, train_PV_y)
    TV_train_seg_x, TV_train_seg_y = gen_inputs(train_TV_x, train_TV_y)
    MV_train_seg_x, MV_train_seg_y = gen_inputs(train_MV_x, train_MV_y)

    AV_valid_seg_x, AV_valid_seg_y = gen_inputs(valid_AV_x, valid_AV_y)
    PV_valid_seg_x, PV_valid_seg_y = gen_inputs(valid_PV_x, valid_PV_y)
    TV_valid_seg_x, TV_valid_seg_y = gen_inputs(valid_TV_x, valid_TV_y)
    MV_valid_seg_x, MV_valid_seg_y = gen_inputs(valid_MV_x, valid_MV_y)
    # Random Shuffle for Training Data
    arr = np.arange(len(AV_train_seg_x))
    np.random.seed(1)
    np.random.shuffle(arr)
    AV_train_seg_x = AV_train_seg_x[arr]
    AV_train_seg_y = AV_train_seg_y[arr]
    
    arr = np.arange(len(PV_train_seg_x))
    np.random.seed(1)
    np.random.shuffle(arr)
    PV_train_seg_x = PV_train_seg_x[arr]
    PV_train_seg_y = PV_train_seg_y[arr]
    
    arr = np.arange(len(TV_train_seg_x))
    np.random.seed(1)
    np.random.shuffle(arr)
    TV_train_seg_x = TV_train_seg_x[arr]
    TV_train_seg_y = TV_train_seg_y[arr]
    
    arr = np.arange(len(MV_train_seg_x))
    np.random.seed(1)
    np.random.shuffle(arr)
    MV_train_seg_x = MV_train_seg_x[arr]
    MV_train_seg_y = MV_train_seg_y[arr]
    ########################################################################### Custom Dataset End

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Location Murmur Detection Model
    # set parameters
    n_filters = 128
    epochs = 100
    batch_size = 64
    patience = 10
    
    AV_model = EEGNet(n_filters = n_filters)
    PV_model = EEGNet(n_filters = n_filters)
    TV_model = EEGNet(n_filters = n_filters)
    MV_model = EEGNet(n_filters = n_filters)

    es1 = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = patience)
    es2 = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = patience)
    es3 = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = patience)
    es4 = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = patience)

    mc1 = ModelCheckpoint(filepath = os.path.join(model_folder, 'AV_model.h5'), monitor = 'val_auc', mode = 'max', save_best_only = True, verbose = 1)
    mc2 = ModelCheckpoint(filepath = os.path.join(model_folder, 'PV_model.h5'), monitor = 'val_auc', mode = 'max', save_best_only = True, verbose = 1)
    mc3 = ModelCheckpoint(filepath = os.path.join(model_folder, 'TV_model.h5'), monitor = 'val_auc', mode = 'max', save_best_only = True, verbose = 1)
    mc4 = ModelCheckpoint(filepath = os.path.join(model_folder, 'MV_model.h5'), monitor = 'val_auc', mode = 'max', save_best_only = True, verbose = 1)
    
    hist1 = AV_model.fit(AV_train_seg_x, AV_train_seg_y,
                     epochs = epochs,
                     batch_size = batch_size, 
                     callbacks = [es1, mc1],
                     validation_data = (AV_valid_seg_x, AV_valid_seg_y)
                    )
    hist2 = PV_model.fit(PV_train_seg_x, PV_train_seg_y,
                     epochs = epochs,
                     batch_size = batch_size,
                     callbacks = [es2, mc2],
                     validation_data = (PV_valid_seg_x, PV_valid_seg_y)
                    )
    hist3 = TV_model.fit(TV_train_seg_x, TV_train_seg_y,
                     epochs = epochs,
                     batch_size = batch_size, 
                     callbacks = [es3, mc3],
                     validation_data = (TV_valid_seg_x, TV_valid_seg_y)
                    )
    hist4 = MV_model.fit(MV_train_seg_x, MV_train_seg_y,
                     epochs = epochs,
                     batch_size = batch_size, 
                     callbacks = [es4, mc4],
                     validation_data = (MV_valid_seg_x, MV_valid_seg_y)
                    )
    
    # Load Deep Learning Model
    AV_model = load_model(os.path.join(model_folder, 'AV_model.h5'))
    PV_model = load_model(os.path.join(model_folder, 'PV_model.h5'))
    TV_model = load_model(os.path.join(model_folder, 'TV_model.h5'))
    MV_model = load_model(os.path.join(model_folder, 'MV_model.h5'))
    
    # Training Murmur Decision Rule
    if verbose >= 1:
        print('Decision Modeling...')
    df = pd.DataFrame(np.zeros((len(test), 38)), 
                      columns = ['AV_mean', 'AV_median', 'AV_q75', 'AV_q25', 'AV_std', 'AV_ratio', 'AV_cossim', 'AV_kurt', 'AV_skew',
                                 'PV_mean', 'PV_median', 'PV_q75', 'PV_q25', 'PV_std', 'PV_ratio', 'PV_cossim', 'PV_kurt', 'PV_skew',
                                 'TV_mean', 'TV_median', 'TV_q75', 'TV_q25', 'TV_std', 'TV_ratio', 'TV_cossim', 'TV_kurt', 'TV_skew',
                                 'MV_mean', 'MV_median', 'MV_q75', 'MV_q25', 'MV_std', 'MV_ratio', 'MV_cossim', 'MV_kurt', 'MV_skew',
                                 'murmur', 'abnormal'])

    for i in range(0, len(test)):
        temp_idx = test[i]
        temp_patient = wave_info2[temp_idx]
        number_loc = len(temp_patient)
        for j in range(number_loc):
            name_loc = temp_patient[j][2]
            if name_loc == 'Ph': # Ph 나오면 예외처리
                continue
            temp_wave = temp_patient[j][0]
            seg_wave = []
            for k in range(0, len(temp_wave) - 4000, 400):
                seg_wave.append(get_wave_features(temp_wave[k: k+4000]))
            seg_wave = np.array(seg_wave)
            if len(seg_wave) == 0: # seg_wave가 empty set이면 건너뛰기
                continue
            if name_loc == 'AV':
                pred = AV_model.predict(seg_wave)
            if name_loc == 'PV':
                pred = PV_model.predict(seg_wave)
            if name_loc == 'TV':
                pred = TV_model.predict(seg_wave)
            if name_loc == 'MV':
                pred = MV_model.predict(seg_wave)

            pred_mean = np.round(np.mean(pred),2)
            pred_median = np.round(np.median(pred), 2)
            pred_quantile_75 = np.round(np.quantile(pred, q = 0.25), 2)
            pred_quantile_25 = np.round(np.quantile(pred, q = 0.75), 2)
            pred_std = np.round(np.std(pred), 2)
            pred_label = []
            for n in range(0, len(pred)):
                if pred[n] >= 0.5:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            pred_ratio = np.round(1 - (np.sum(pred_label)/len(pred_label)), 2)
            pred_cossim = np.ones(pred.shape)
            pred_cossim = scipy.spatial.distance.cosine(pred_cossim.flatten(), pred.flatten())
            pred_kurt = kurtosis(pred)
            pred_skew = skew(pred)

            if name_loc == temp_patient[j-1][2]:
                if pred_mean < df[name_loc+'_mean'][i]:
                    pred_mean = df[name_loc+'_mean'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_median < df[name_loc+'_median'][i]:
                    pred_median = df[name_loc+'_median'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_quantile_75 < df[name_loc+'_q75'][i]:
                    pred_quantile_75 = df[name_loc+'_q75'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_quantile_25 < df[name_loc+'_q25'][i]:
                    pred_quantile_25 = df[name_loc+'_q25'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_std < df[name_loc+'_std'][i]:
                    pred_std = df[name_loc+'_std'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_ratio < df[name_loc+'_ratio'][i]:
                    pred_ratio = df[name_loc+'_ratio'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_cossim < df[name_loc+'_cossim'][i]:
                    pred_cossim = df[name_loc+'_cossim'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_kurt < df[name_loc+'_kurt'][i]:
                    pred_kurt = df[name_loc+'_kurt'][i]
            if name_loc == temp_patient[j-1][2]:
                if pred_skew < df[name_loc+'_skew'][i]:
                    pred_skew = df[name_loc+'_skew'][i]

            df[name_loc+'_mean'][i] = pred_mean
            df[name_loc+'_median'][i] = pred_median
            df[name_loc+'_q75'][i] = pred_quantile_75
            df[name_loc+'_q25'][i] = pred_quantile_25
            df[name_loc+'_std'][i] = pred_std
            df[name_loc+'_ratio'][i] = pred_ratio
            df[name_loc+'_cossim'][i] = pred_cossim
            df[name_loc+'_kurt'][i] = pred_kurt
            df[name_loc+'_skew'][i] = pred_skew
        df['murmur'][i] = np.argmax(murmurs[temp_idx])
        df['abnormal'][i] = np.argmax(outcomes[temp_idx])
    
    decision_features = df[['AV_mean', 'AV_median', 'AV_q75', 'AV_q25', 'AV_std', 'AV_ratio', 'AV_cossim', 'AV_kurt', 'AV_skew',
                        'PV_mean', 'PV_median', 'PV_q75', 'PV_q25', 'PV_std', 'PV_ratio', 'PV_cossim', 'PV_kurt', 'PV_skew',
                        'TV_mean', 'TV_median', 'TV_q75', 'TV_q25', 'TV_std', 'TV_ratio', 'TV_cossim', 'TV_kurt', 'TV_skew',
                        'MV_mean', 'MV_median', 'MV_q75', 'MV_q25', 'MV_std', 'MV_ratio', 'MV_cossim', 'MV_kurt', 'MV_skew',]].values
    decision_target_murmur = df['murmur'].values
    decision_target_abnormal = df['abnormal'].values
    
    murmur_classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 0)
    murmur_classifier.fit(decision_features, decision_target_murmur)
    outcome_classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 0)
    outcome_classifier.fit(decision_features, decision_target_abnormal)

    # Save the model.
    save_challenge_model(model_folder, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier, AV_model, PV_model, TV_model, MV_model)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose): # data = current patient data, recordings = current recording
    # Load Decision Model
    murmur_classes = model['murmur_classes']
    murmur_classifier = model['murmur_classifier']
    outcome_classes = model['outcome_classes']
    outcome_classifier = model['outcome_classifier']
    # Load Murmur Prediction Model
    AV_model = model['AV_model']
    PV_model = model['PV_model']
    TV_model = model['TV_model']
    MV_model = model['MV_model']
    # Segmentation Prediction -> Decision Rule
    wave, location = custom_get_features2(data, recordings) # wave list and location list
    df = pd.DataFrame(np.zeros((1, 36)),
                      columns = ['AV_mean', 'AV_median', 'AV_q75', 'AV_q25', 'AV_std', 'AV_ratio', 'AV_cossim', 'AV_kurt', 'AV_skew',
                                 'PV_mean', 'PV_median', 'PV_q75', 'PV_q25', 'PV_std', 'PV_ratio', 'PV_cossim', 'PV_kurt', 'PV_skew',
                                 'TV_mean', 'TV_median', 'TV_q75', 'TV_q25', 'TV_std', 'TV_ratio', 'TV_cossim', 'TV_kurt', 'TV_skew',
                                 'MV_mean', 'MV_median', 'MV_q75', 'MV_q25', 'MV_std', 'MV_ratio', 'MV_cossim', 'MV_kurt', 'MV_skew'])
    
    number_loc = len(location)
    for j in range(number_loc):
        name_loc = location[j]
        if name_loc in 'Phc': # except Phc location
            continue
        temp_wave = wave[j]
        seg_wave = []
        for k in range(0, len(temp_wave) - 4000, 400):
            seg_wave.append(get_wave_features(temp_wave[k: k+4000]))
        seg_wave = np.array(seg_wave)
        if len(seg_wave) == 0: # if seg_wave is empty set
            continue
        if name_loc == 'AV':
            pred = AV_model.predict(seg_wave)
        if name_loc == 'PV':
            pred = PV_model.predict(seg_wave)
        if name_loc == 'TV':
            pred = TV_model.predict(seg_wave)
        if name_loc == 'MV':
            pred = MV_model.predict(seg_wave)

        pred_mean = np.round(np.mean(pred),2)
        pred_median = np.round(np.median(pred), 2)
        pred_quantile_75 = np.round(np.quantile(pred, q = 0.25), 2)
        pred_quantile_25 = np.round(np.quantile(pred, q = 0.75), 2)
        pred_std = np.round(np.std(pred), 2)
        pred_label = []
        for n in range(0, len(pred)):
            if pred[n] >= 0.5:
                pred_label.append(1)
            else:
                pred_label.append(0)
        pred_ratio = np.round(1 - (np.sum(pred_label)/len(pred_label)), 2)
        pred_cossim = np.ones(pred.shape)
        pred_cossim = scipy.spatial.distance.cosine(pred_cossim.flatten(), pred.flatten())
        pred_kurt = kurtosis(pred)
        pred_skew = skew(pred)

        df[name_loc+'_mean'][0] = pred_mean
        df[name_loc+'_median'][0] = pred_median
        df[name_loc+'_q75'][0] = pred_quantile_75
        df[name_loc+'_q25'][0] = pred_quantile_25
        df[name_loc+'_std'][0] = pred_std
        df[name_loc+'_ratio'][0] = pred_ratio
        df[name_loc+'_cossim'][0] = pred_cossim
        df[name_loc+'_kurt'][0] = pred_kurt
        df[name_loc+'_skew'][0] = pred_skew
    
    features = df.values
    ####################################################################

    # Get classifier probabilities.
    murmur_probabilities = murmur_classifier.predict_proba(features)
    murmur_probabilities = np.asarray(murmur_probabilities, dtype=np.float32)
    outcome_probabilities = outcome_classifier.predict_proba(features)
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float32)

    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities.flatten(), outcome_probabilities.flatten()))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier, AV_model, PV_model, TV_model, MV_model):
    d = {'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier,
        'AV_model' : AV_model, 'PV_model' : PV_model, 'TV_model' : TV_model, 'MV_model' : MV_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)


###########################################################################################
### Custom Function

# Load recordings(custom ver).
def custom_load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]
    
    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        ##############################
        loc_murmur = data.split('\n')[1+num_locations+5][9:]
        recording_file = entries[2]
        tsv_file = entries[3]
        recording_loc = entries[0][:2] # location 추가
        if loc_murmur == 'Present':
            if recording_loc in data.split('\n')[1+num_locations+6]:
                loc_murmur = 'Present'
            else:
                loc_murmur = 'Absent'
        ##############################
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        segment = pd.read_csv(os.path.join(data_folder, tsv_file), sep = '\t', names = ['start', 'end', 'seg'])
        recordings.append([recording, segment, recording_loc, loc_murmur])
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings
    
def custom_get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = 0
    if compare_strings(sex, 'Female'):
        sex_features = 0
    elif compare_strings(sex, 'Male'):
        sex_features = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)
    bmi = np.round(weight / ((height/100)**2), 2)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)
    num_recordings = len(recordings)

    features = np.hstack(([age], sex_features, [height], [weight], [bmi], [is_pregnant], num_recordings))

    return np.asarray(features, dtype=np.float32)

def gen_inputs(train_x, train_y):
    features = []
    target = []
    
    train_y_num = []
    for n in range(0, len(train_y)):
        if train_y[n] == 'Present':
            train_y_num.append(0)
        else:
            train_y_num.append(1)
    for i in range(0, len(train_x)):
        for j in range(0, len(train_x[i]) - 4000, 400):
            features.append(get_wave_features(train_x[i][j: j+4000]))
            target.append(train_y_num[i])
    
    features = np.array(features)
    target = np.array(target)
    return features, target

# Extract features from the data.
def custom_get_features2(data, recordings):

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV']
    num_recording_locations = len(recording_locations)
    
    recording_features = []
    location_features = []
    
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features.append(recordings[i])
                    location_features.append(locations[i])
                    
    return recording_features, location_features
