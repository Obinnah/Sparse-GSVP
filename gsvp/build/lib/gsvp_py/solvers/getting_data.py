import numpy as np
def getdata(data):
    if data not in ['Alpaca Dataset', 'Camel Dataset', 'Camelid Dataset','Diabetes Dataset','Heart Disease Dataset','Breast Cancer Dataset','Ovarian Cancer Dataset', 'Prostate Cancer Dataset', 'Glioma Dataset',
                    'Lung Cancer Dataset','Leukemia Dataset', 'Influenza I', 'Influenza II', 'Influenza III','Influenza IV', 'Influenza V','Influenza VI']:
        raise ValueError(f"Invalid dataset name: {data}")
    else:
        if data == 'Diabetes Dataset':
            pp1 = 'Train: No'
            pp2 = 'Train: Yes'
            pp3 = 'Test: No'
            pp4 = 'Test: Yes'
            pp = 'Diabetes Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_diabetes.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_diabetes.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_diabetes.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_diabetes.npy')


        elif data == 'Alpaca Dataset':
            pp1 = 'Train: Controls'
            pp2 = 'Train: Infected'
            pp3 = 'Test: Controls'
            pp4 = 'Test: Infected'
            pp = 'Alpaca MERS-CoV Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_alpaca.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_alpaca.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_alpaca.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_alpaca.npy')

        elif data == 'Camel Dataset':
            pp1 = 'Train: Controls'
            pp2 = 'Train: Infected'
            pp3 = 'Test: Controls'
            pp4 = 'Test: Infected'
            pp = 'Camel MERS-CoV Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_camel.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_camel.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_camel.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_camel.npy')

        elif data == 'Camelid Dataset':
            pp1 = 'Train: Controls'
            pp2 = 'Train: Infected'
            pp3 = 'Test: Controls'
            pp4 = 'Test: Infected'
            pp = 'Camelid MERS-CoV Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_camelid.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_camelid.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_camelid.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_camelid.npy')

        elif data == 'Heart Disease Dataset':
            pp1 = 'Train: No'
            pp2 = 'Train: Yes'
            pp3 = 'Test: No'
            pp4 = 'Test: Yes'
            pp = 'Heart Disease Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_heart_cleveland.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_heart_cleveland.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_heart_cleveland.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_heart_cleveland.npy')

        elif data == 'Breast Cancer Dataset':
            pp1 = 'Train: Benign'
            pp2 = 'Train: Malignant'
            pp3 = 'Test: Benign'
            pp4 = 'Test: Malignant'
            pp = 'Breast Cancer Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_breast_cancer.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_breast_cancer.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_breast_cancer.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_breast_cancer.npy')

        elif data == 'Ovarian Cancer Dataset':
            pp1 = 'Train: Normal'
            pp2 = 'Train: Cancer'
            pp3 = 'Test: Normal'
            pp4 = 'Test: Cancer'
            pp = 'Ovarian Cancer Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Ovarian.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Ovarian.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Ovarian.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Ovarian.npy')

        elif data == 'Prostate Cancer Dataset':
            pp = 'Prostate Cancer Dataset'
            pp1 = 'Train: Normal'
            pp2 = 'Train: Cancer'
            pp3 = 'Test: Normal'
            pp4 = 'Test: Cancer'
            pp =  'Prostate Cancer Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Prostate_GE.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Prostate_GE.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Prostate_GE.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Prostate_GE.npy')

        elif data == 'Glioma Dataset':
            pp1 = 'Train: Normal'
            pp2 = 'Train: Tumor'
            pp3 = 'Test: Normal'
            pp4 = 'Test: Tumor'
            pp = 'Glioma Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_GLI_85.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_GLI_85.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_GLI_85.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_GLI_85.npy')

        elif data == 'Lung Cancer Dataset':
            pp1 = 'Train: Smokers without Cancer'
            pp2 = 'Train: Smokers with Cancer'
            pp3 = 'Test: Smokers without Cancer'
            pp4 = 'Test: Smokers with Cancer'
            pp = 'Lung cancer Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_SMK_CAN_187.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_SMK_CAN_187.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_SMK_CAN_187.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_SMK_CAN_187.npy')

        elif data == 'Leukemia Dataset':
            pp1 = 'Train: ALL'
            pp2 = 'Train: AML'
            pp3 = 'Test: ALL'
            pp4 = 'Test: AML'
            pp = 'Leukemia Dataset'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Leukemia.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Leukemia.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Leukemia.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Leukemia.npy')

        elif data == 'Influenza I':
            pp1 = 'Train: H1N1 Non-Shedders+Shedders'
            pp2 = 'Train: H3N2 Non-Shedders+Shedders'
            pp3 = 'Test: H1N1 Shedders'
            pp4 = 'Test: H3N2 Shedders'
            pp = 'Influenza I'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Influenza_I.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Influenza_I.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Influenza_I.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Influenza_I.npy')

        elif data == 'Influenza II':
            pp1 = 'Train: H1N1 Non-Shedders'
            pp2 = 'Train: H3N2 Non-Shedders'
            pp3 = 'Test: H1N1 Shedders'
            pp4 = 'Test: H3N2 Shedders'
            pp = 'Influenza II'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Influenza_II.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Influenza_II.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Influenza_II.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Influenza_II.npy')


        elif data == 'Influenza III':
            pp1 = 'Train: H1N1 Non-Shedders'
            pp2 = 'Train: H1N1 Shedders'
            pp3 = 'Test: H3N2 Non-Shedders'
            pp4 = 'Test: H3N2 Shedders'
            pp = 'Influenza III'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Influenza_III.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Influenza_III.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Influenza_III.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Influenza_III.npy')

        elif data == 'Influenza IV':
            pp1 = 'Train: H1N1+H3N2: Non-Shedders'
            pp2 = 'Train: H1N1+H3N2: Shedders'
            pp3 = 'Test: H1N1+H3N2: Non-Shedders'
            pp4 = 'Test: H1N1+H3N2: Shedders'
            pp = 'Influenza IV'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Influenza_IV.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Influenza_IV.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Influenza_IV.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Influenza_IV.npy')

        elif data == 'Influenza V':
            pp1 = 'Train: H1N1+H3N2: Non-Shedders'
            pp2 = 'Train: H1N1+H3N2: Shedders'
            pp3 = 'Test: H1N1+H3N2: Non-Shedders'
            pp4 = 'Test: H1N1+H3N2: Shedders'
            pp = 'Influenza V'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Influenza_V.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Influenza_V.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Influenza_V.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Influenza_V.npy')


        elif data == 'Influenza VI':
            pp1 = 'Train: H1N1+H3N2: Non-Shedders'
            pp2 = 'Train: H1N1+H3N2: Shedders'
            pp3 = 'Test: H1N1+H3N2: Non-Shedders'
            pp4 = 'Test: H1N1+H3N2: Shedders'
            pp = 'Influenza VI'
            X_train = np.load('C:/Users/uugob/GSVP/Training_data/X_train_Influenza_VI.npy')
            y_train = np.load('C:/Users/uugob/GSVP/Training_data/y_train_Influenza_VI.npy')
            X_val = np.load('C:/Users/uugob/GSVP/Validation_data/X_val_Influenza_VI.npy')
            y_val = np.load('C:/Users/uugob/GSVP/Validation_data/y_val_Influenza_VI.npy')

    return X_train, y_train, X_val, y_val, pp1, pp2, pp3, pp4,pp
