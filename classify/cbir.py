#!/usr/bin/env python

"""Methods for CBIR workflow, assembly and retrieval"""

from libsvm.svmutil import svm_train,svm_predict,svm_save_model,evaluations
import numpy as np
import os

CLASSES = [7,8,9,10,11,12,13]

# ---
def features_read_file(datafile,delimiter=','):
    """
    Read features file from images
    
    Input:
     - datafile  'str' : Blockdata filename;
                         1st column are the class labels, 2nd column is any ID for
                         instances and the rest of the columns are the features (int/float)
     - delimiter 'str' : Default, for a CSV file: ","
    
    Output:
     - Y_classes    ndarray(int)   : Classes (1st column of datafile)
     - X_features   ndarray(float) : Features (from 3rd to last column)
     - Image_files  ndarray(str)   : Ids (2nd column of datafile)
    
    ---
    """

    #  So far done in Octave: cosine modes    
    data = np.loadtxt(datafile,dtype='S11',delimiter=',')
    Y_classes = data[:,0]
    Y_classes = Y_classes.reshape((len(data),1))
    X_features = data[:,2:]
    Image_files = data[:,1]
    
    return Y_classes,X_features,Image_files

# ---
def sample_selection(Y_classes,classe,k=2):
    """
    Selects "main" and "other" sample instances
    
    The "other" sample is 'k' times bigger then "main" sample, aimed for
    the one-against-all SVM training approach.
    
    Input:
     - Y_classes  ['int'] : List of labels/classes
     - classe      'int'  : Particular class to select as main class
    
    Output:
     - indx_main  ndarray : main class indexes array
     - indx_other ndarray : other (then main) classes indexes array
    
    ---
    """
    
    i_class = classe
    
    indx_main = np.where(Y_classes==i_class)[0]
    size_main = len(indx_main)
    
    indx_other = np.where(Y_classes)[0]
    indx_other = np.asarray(list( set(indx_other)-set(indx_main) ))
    np.random.shuffle(indx_other)
    
    return indx_main,indx_other[:k*size_main]

# ---
def _convert_arrays2lists(Y,X):
    """
    Converts numpy ndarrays to LibSVM lists input format
    """
    
    Y_list = list(Y.flatten().astype(int))
    X_list = []
    nraws,ncols = X.shape
    labels = range(0,ncols)
    
    for lin in range(nraws):
        x_dic = {}
        for col in range(ncols):
            x_dic[col] = X[lin,col]
        X_list.append(x_dic)
        
    return Y_list,X_list

# ---
def svr_training(X_features,Y_classes,classes=[],output='svr_',training_options = '-s 3 -t 0 -b 1'):
    """
    Configure multiple SV Machines based on a one-against-all (1AA) approach
    
    Input:
     - X_features  ndarray(float) : Array of instance features (instances on rows)
     - Y_classes   ndarray(int)   : Array of classes identification
     - classes            ['int'] : Classes to be used for the 1AA approach
    
    Output:
     - model_classes  [] : SVM models for classes given in 'classes'
    
    ---
    """
    
    model_classes = []
    training_options = '-s 3 -t 0 -b 1'
    
    diro = 'models/'
    try:
        os.mkdir(diro)
    except:
        pass;
    
    for i_class in classes:
        classe = 'class'+str(i_class)
        
        this_class_indx, other_class_indx = sample_selection(Y_classes,i_class)
        X = X_features[np.concatenate((this_class_indx,other_class_indx))]
        Y = np.zeros((len(X),1))
        Y[:len(this_class_indx)] = 1
        Y[len(this_class_indx):] = -1
        Y_list,X_list = _convert_arrays2lists(Y,X)
        
        model_classes.append(svm_train(Y_list,X_list,training_options))

        svm_save_model(diro+output+classe+'.model',model_classes[-1])
        np.savetxt(output+classe+'_svr.dat',np.concatenate((Y,X),axis=1),fmt='%f')
        
    return model_classes

# ---
def svr_prediction(X_features,Y_classes,models=[],predict_options='-b 1'):
    """
    Predict the class members probability
    """
    
    Y_svfeatures = Y_classes.copy()
    
    Y_list,X_list = _convert_arrays2lists(Y_classes,X_features)
    Y_list = [0]*len(Y_list)
    
    labels = []
    accur = []
    vals = []
    for i_model in models:
    
        # predict model
        p_labels,p_accur,p_vals = svm_predict(Y_list,X_list,i_model,predict_options)
        
        labels.append(p_labels)
    
    return labels

# ---
def run_1AA_assembly(datafile='feature_vectors.csv'):
    """
    This function builds up (trains) the machines (SV) necessary for CBIR
    
    Input:
     - datafile  'str' : Blockdata filename;
                         1st column are the class labels, 2nd column is any ID for
                         instances and the rest of the columns are the features (int/float)

    Output:
     - models_r  <LibSVM-models> : Models for dimensionality reduction
     - models_m  <LibSVM-models> : Models for classes classification

    """
    
    # Read list of images, their corresponding classes and features blockdata
    Y_classes,X_features,Image_files = features_read_file(datafile)
    Y_classes = Y_classes.astype(int)
    X_features = X_features.astype(float)

    # Reduce instances dimensionality
    models_r = svr_training(X_features,Y_classes,CLASSES,'dimreduc_')
    Y_svprobs = svr_prediction(X_features,Y_classes,models=models_r)
    X_svfeats = np.asarray(Y_svprobs).T
    
    # Train the N machines for final classification
    models_m = svr_training(X_svfeats,Y_classes,CLASSES,'classify_','-s 3 -b 1')
    
    return models_r,models_m

# ---
def run_1AA_retrieve(datafile,reduction_models=[],class_models=[],DBfile='feature_vectors.csv'):
    """
    This function builds up (trains) the machines (SV) necessary for CBIR
    
    Input:
     - datafile  'str' : Blockdata filename;
                         1st column are the class labels, 2nd column is any ID for
                         instances and the rest of the columns are the features (int/float)
     - reduction_models   [] : LibSVM models for dim. reduciton
     - class_models       [] : LibSVM models for classification
     - DBfile            str : CSV file with the image database

    """
    
    # Read list of images, their corresponding classes and features blockdata
    Y_classes,X_features,Image_files = features_read_file(datafile)
    Y_classes = Y_classes.astype(int)
    X_features = X_features.astype(float)

    # Reduce instances dimensionality
    Y_svprobs = svr_prediction(X_features,Y_classes,models=reduction_models)
    X_svfeats = np.asarray(Y_svprobs).T
    
    # Train the N machines for final classification
    Y_classed = svr_prediction(X_svfeats,Y_classes,models=class_models)
    Y_out = np.asarray(Y_classed).T
    
    # Access the DB contents to retrieve similar images
    class_match = CLASSES[np.argmax(Y_out[0])]
    Y_DBclasses,X_DBfeatures,Image_DBfiles = features_read_file(DBfile)
    Y_DBclasses = Y_DBclasses.astype(int)
    X_DBfeatures = X_DBfeatures.astype(float)
    indx_class = np.where(Y_DBclasses==class_match)
    X_dists = np.dot(X_features[0],X_DBfeatures[indx_class[0]].T)
    X_args2sort = np.argsort(X_dists)#[::-1]
    Image_DBmatch = Image_DBfiles[indx_class[0]]
    
    first_three = X_args2sort[:3]
    print ""
#    print "Classe pertencente a imagem %s"%(Image_files[0])
    print "Imagens mais proximas a %s, na classe %d "%(Image_files[0],class_match)
    print "1) %s"%(Image_DBmatch[first_three[0]])
    print "2) %s"%(Image_DBmatch[first_three[1]])
    print "3) %s"%(Image_DBmatch[first_three[2]])
    print ""
    
    return Y_out
    
