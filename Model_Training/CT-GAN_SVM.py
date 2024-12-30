from packages import *
def generate_data():
    data = pd.read_csv('bluebottle_modeldata.csv')
    X = data.drop(columns=['Presence'])
    y = data['Presence']
    continuous_vars = ['SST', 'Curr_Speed', 'WD_Speed', 'Curr_Dir', 'WD_Dir']
    discrete_vars = [col for col in X.columns if col not in continuous_vars]
    X_positive = data[data["Presence"] == 1].drop(columns=['Presence'])  # Positive class (bluebottle presence)
    X_negative = data[data['Presence'] == 0].drop(columns=['Presence'])  # Negative class (bluebottle absence)

    # Split negative class data first, to avoid leakage in CTGAN training
    X_negative_train, X_negative_test = train_test_split(X_negative, test_size=0.4, random_state=42)

    # Train the CTGAN only on the negative training samples
    ctgan = CTGAN()
    ctgan.fit(X_negative_train, discrete_columns=discrete_vars)

    return X_positive, ctgan, continuous_vars, X_negative_test

def sample_data(X_positive, ctgan, prob, run):
    X_positive_train, X_positive_test = train_test_split(X_positive, test_size=0.4, random_state=run)
    if prob == 'One-class svm':
        synthetic_negative_samples = ctgan.sample(X_positive_test.shape[0])
        synthetic_negative_samples['Curr_Dir'] = synthetic_negative_samples['Curr_Dir'].clip(lower=0, upper=360)
        synthetic_negative_samples['WD_Dir'] = synthetic_negative_samples['WD_Dir'].clip(lower=0, upper=360)
        X_test = pd.concat([X_positive_test, synthetic_negative_samples])
        y_test = pd.concat([pd.Series(1, index=X_positive_test.index), pd.Series(0, index=synthetic_negative_samples.index)])
    else:
        samples = ctgan.sample(X_positive.shape[0])
        samples['Curr_Dir'] = samples['Curr_Dir'].clip(lower=0, upper=360)
        samples['WD_Dir'] = samples['WD_Dir'].clip(lower=0, upper=360)
        ctgan_data = pd.concat([X_positive, samples], ignore_index=True)
        y_combined = pd.concat([pd.Series(1, index=X_positive.index), pd.Series(0, index=samples.index)]).reset_index(drop=True)
        ctgan_data['Presence'] = y_combined

    return X_positive_train, X_test, y_test

def train_oneclass_svm(X_positive_train, X_test, y_test, continuous_vars, run, runs, train_f1_scores, train_accuracies, test_f1_scores, test_accuracies, test_aucs):
    scaler = StandardScaler()
    X_positive_train[continuous_vars] = scaler.fit_transform(X_positive_train[continuous_vars].copy())
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars].copy())
    one_class_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(X_positive_train)
    
    # Predict on training data for training metrics
    y_train_pred = one_class_svm.predict(X_positive_train)
    y_train_pred = [1 if x == 1 else 0 for x in y_train_pred]  # Convert -1 to 0 for negative
    y_pred = one_class_svm.predict(X_test)
    y_pred = [1 if x == 1 else 0 for x in y_pred]
    
    # Calculate training metrics
    train_f1 = f1_score(np.ones(len(y_train_pred)), y_train_pred)
    train_accuracy = accuracy_score(np.ones(len(y_train_pred)), y_train_pred)
    class_report_train = classification_report(np.ones(len(y_train_pred)), y_train_pred, target_names=["Absence", "Presence"], output_dict=True)
    class_report_test = classification_report(y_test, y_pred, output_dict=True) 

    # Store training metrics
    train_f1_scores.append(train_f1)
    train_accuracies.append(train_accuracy)

    # Calculate test metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # Store test metrics
    test_f1_scores.append(f1)
    test_accuracies.append(accuracy)
    test_aucs.append(auc)

    # Print confusion matrix and classification report for the current run
    if run == runs - 1:
        print("Confusion Matrix on Test:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report (Train):")
        train_report = pd.DataFrame(class_report_train).transpose()
        print(train_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))

        # Print Classification Report for Test with four decimal places
        print("Classification Report (Test):")
        test_report = pd.DataFrame(class_report_test).transpose()
        print(test_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))
        print("-" * 60)

def main():
    # Initialize lists to store metrics for each run
    train_f1_scores = []
    train_accuracies = []
    test_f1_scores = []
    test_accuracies = []
    test_aucs = []

    # Number of runs
    runs = 10

    # Generate data
    X_positive, ctgan, continuous_vars, X_negative_test = generate_data()

    # Perform multiple runs
    for run in range(runs):
        X_positive_train, X_test, y_test = sample_data(X_positive, ctgan, 'One-class svm', run)
        train_oneclass_svm(X_positive_train, X_test, y_test, continuous_vars, run, runs, train_f1_scores, train_accuracies, test_f1_scores, test_accuracies, test_aucs)

    # Print overall mean and standard deviation for metrics
    print("Training F1 Score (Mean ± Std):", np.mean(train_f1_scores), '±', np.std(train_f1_scores))
    print("Training Accuracy (Mean ± Std):", np.mean(train_accuracies), '±', np.std(train_accuracies))
    print("Test F1 Score (Mean ± Std):", np.mean(test_f1_scores), '±', np.std(test_f1_scores))
    print("Test Accuracy (Mean ± Std):", np.mean(test_accuracies), '±', np.std(test_accuracies))
    print("Test AUC (Mean ± Std):", np.mean(test_aucs), '±', np.std(test_aucs))

# Run the main function
if __name__ == "__main__":
    main()