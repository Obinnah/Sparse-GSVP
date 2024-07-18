import numpy as np
def getdata(data):
    if data not in ['Breast Cancer Dataset','Ovarian Cancer Dataset','Prostate Cancer Dataset']:
        raise ValueError(f"Invalid dataset name: {data}")
    else:
        
        if data == 'Breast Cancer Dataset':
            pp1 = 'Train: Benign'
            pp2 = 'Train: Malignant'
            pp3 = 'Val: Benign'
            pp4 = 'Val: Malignant'
            pp = 'Breast Cancer Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_breast_cancer.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_breast_cancer.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_breast_cancer.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_breast_cancer.npy')

        elif data == 'Ovarian Cancer Dataset':
            pp1 = 'Train: Normal'
            pp2 = 'Train: Cancer'
            pp3 = 'Val: Normal'
            pp4 = 'Val: Cancer'
            pp = 'Ovarian Cancer Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Ovarian.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Ovarian.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Ovarian.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Ovarian.npy')

    return X_train, y_train, X_val, y_val, pp1, pp2, pp3, pp4,pp
