from packages import *
from Exploratory_analysis import *
from sklearn.model_selection import RandomizedSearchCV

def read_data(data_type, sample = False):
    if data_type == 'bluebottle':
        data = pd.read_csv('bluebottle_modeldata.csv')
        if sample == True:
            class_0 = data[data['Presence'] == 0]
            class_1 = data[data['Presence'] == 1]
            class_0_sample = class_0.sample(class_1.shape[0], random_state=42)
            class_1_sample = class_1
            data = pd.concat([class_0_sample, class_1_sample]).reset_index(drop=True)
    elif data_type == 'GAN':
        data = pd.read_csv('CTGAN_data.csv')
    return data 

def split_data(data, selected_features, run, augment=False, normalize=True):
    # Determine feature set
    if selected_features == 'yes':
        X = data.drop(columns=['Presence']) #if 'Presence' in data.columns else data
    elif isinstance(selected_features, list):
        # Use only the specified selected features
        X = data[selected_features]
    else:
        raise ValueError("selected_features should be 'yes' or a list of feature names.")
        
    y = data['Presence']
    continuous_vars = ['SST', 'WD_Dir', 'WD_Speed', 'Curr_Dir', 'Curr_Speed']
    discrete_vars = [col for col in X.columns if col not in continuous_vars]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=run)
    if normalize == True:
    # Standardize continuous variables
        scaler = StandardScaler()
        X_train[continuous_vars] = scaler.fit_transform(X_train[continuous_vars].copy())
        X_test[continuous_vars] = scaler.transform(X_test[continuous_vars].copy())
    else: 
        pass
    
    smote_data = None  # Initialize smote_data as None by default

    # Apply SMOTE and undersampling if augment is True
    if augment:
        smotenc = SMOTENC(categorical_features=[X.columns.get_loc(col) for col in discrete_vars],
                          sampling_strategy='minority', random_state=run)
        under = RandomUnderSampler(sampling_strategy='majority', random_state=run)
        pipeline = Pipeline(steps=[('over', smotenc), ('under', under)])
        
        X_train, y_train = pipeline.fit_resample(X_train, y_train)
        smote_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    
    return X_train, X_test, y_train, y_test, continuous_vars, smote_data

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Train F1 Score': f1_score(y_train, y_train_pred),
        'Train AUC': roc_auc_score(y_train, y_train_prob),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test F1 Score': f1_score(y_test, y_test_pred),
        'Test AUC': roc_auc_score(y_test, y_test_prob),
        'Confusion Matrix Train': confusion_matrix(y_train, y_train_pred),
        'Confusion Matrix Test': confusion_matrix(y_test, y_test_pred),
        'Classification Report Train': classification_report(y_train, y_train_pred, output_dict=True),
        'Classification Report Test': classification_report(y_test, y_test_pred, output_dict=True),
    }
    return metrics, y_train_prob, y_test_prob

def train_classifiers(X_train, y_train, X_test, y_test, run, aggregate_metrics, model_gan = False):
    
    best_models = {model: {'auc': 0, 'run': None, 'y_train': None, 'y_train_prob': None, 'y_test': None, 
                           'y_test_prob': None} for model in ['mlp', 'rf', 'xgb']}
    param_grids = {
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01],
            'solver': ['adam', 'sgd'],
        },
        'rf': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'xgb': {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 150],
            'scale_pos_weight': [1, 2, 3],
        }
    }
    if model_gan == True:
        all_models = {
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.1, solver='adam', max_iter=400, alpha=0.0001,early_stopping=True, random_state=run),
        'rf': RandomForestClassifier(bootstrap=False, max_depth=7, min_samples_split=10, max_features='log2', n_estimators=150, min_samples_leaf=2, max_leaf_nodes=20, ccp_alpha=0.01, random_state=run),
        'xgb': xgb.XGBClassifier(learning_rate=0.1, max_depth=1, n_estimators=100, min_child_weight=1, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss')
    }
    else:
        all_models = {
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.1, solver='adam', max_iter=400, alpha=0.0001,early_stopping=True, random_state=run),
        'rf': RandomForestClassifier(random_state=run),
        'xgb': xgb.XGBClassifier(learning_rate=0.1, max_depth=2, n_estimators=100, min_child_weight=1, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss')
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=run)
    model_metrics_dict = {}

    for model_name, model in all_models.items():
        #random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_name], 
            #n_iter=10, scoring='roc_auc', cv=5, random_state=run, n_jobs=-1)
        #random_search.fit(X_train, y_train)
        
        # Use the best estimator found by RandomizedSearchCV
        #best_model = random_search.best_estimator_
        
        # Train and evaluate the tuned model
        model.fit(X_train, y_train)
        metrics, y_train_prob, y_test_prob = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        model_metrics_dict[model_name] = metrics
        
        for metric in ['Train Accuracy', 'Train F1 Score', 'Train AUC', 'Test Accuracy', 'Test F1 Score', 'Test AUC']:
            aggregate_metrics[model_name][metric].append(metrics[metric])
        
        if metrics['Test AUC'] > best_models[model_name]['auc']:
            best_models[model_name].update({'auc': metrics['Test AUC'], 'run': run, 'y_train': y_train, 
                                            'y_train_prob': y_train_prob, 'y_test': y_test, 
                                            'y_test_prob': y_test_prob})

    return best_models, all_models, model_metrics_dict

def plot_metrics(best_models, all_models, X_train):
    for model_name, best_model in best_models.items():
        print(f"Best Model: {model_name.upper()} on run {best_model['run']} with Test AUC: {best_model['auc']:.4f}")
        
        # ROC Curve
        fpr_train, tpr_train, _ = roc_curve(best_model['y_train'], best_model['y_train_prob'])
        fpr_test, tpr_test, _ = roc_curve(best_model['y_test'], best_model['y_test_prob'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f"Train ROC (AUC = {roc_auc_score(best_model['y_train'], best_model['y_train_prob']):.2f})")
        plt.plot(fpr_test, tpr_test, color='orange', lw=2, label=f"Test ROC (AUC = {best_model['auc']:.2f})")
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.title(f"ROC Curve for Best {model_name.upper()} Model")
        plt.legend(loc="lower right")
        plt.savefig(f'model_{model_name}_Bluebottle_ROC Curve.svg')
        plt.show()

        precision_train, recall_train, _ = precision_recall_curve(best_model['y_train'], best_model['y_train_prob'])
        precision_test, recall_test, _ = precision_recall_curve(best_model['y_test'], best_model['y_test_prob'])

        plt.figure(figsize=(12, 6))
        plt.plot(recall_train, precision_train, color='blue', lw=2, label="Train Precision-Recall curve")
        plt.plot(recall_test, precision_test, color='orange', lw=2, label="Test Precision-Recall curve")
        plt.xlabel("Recall", fontsize=16, color='black')
        plt.ylabel("Precision", fontsize=16, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.title(f"Precision-Recall Curve for Best {model_name.upper()} Model", fontsize = 18, color='black')
        plt.legend(loc="lower left")
        plt.savefig(f'model_{model_name}_Blue_Precision-Recall Curve.svg')
        plt.show()

        # Feature Importance
    for model_name in ['rf', 'xgb']:  # Only models with feature importances
        model = all_models[model_name]
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            sorted_idx = np.argsort(feature_importances)
            sorted_features = X_train.columns[sorted_idx]
            sorted_importances = feature_importances[sorted_idx]
    # Create a bar plot with features on the x-axis
            plt.figure(figsize=(15, 8))  # Adjusting the width to accommodate long feature names
            plt.bar(sorted_features, sorted_importances, color='skyblue')
            plt.ylabel('Feature Importance', fontsize=16, color = 'black') 
            plt.xlabel('Features', fontsize=16, color = 'black') 
            plt.title(f'Feature Importance in {model_name.upper()}', fontsize = 18,  color = 'black') 
            # Rotate the feature labels for better readability
            plt.xticks(rotation=90, fontsize=16)
            plt.yticks(fontsize=16)
            plt.tick_params(axis='both', colors='black')
            plt.tight_layout()  
            plt.savefig(f'model_{model_name}_feature_importance.svg')
            plt.show()
def main():
    runs = 30
    selected_features = 'yes'
    X_selected = ['SST', 'Curr_Speed', 'Curr_Dir', 'WD_Speed', 'WD_Dir', 'Month_Jan', 'Month_Feb', 'Month_Dec', 'Month_Oct']
    aggregate_metrics_all_runs = {model: {'Train Accuracy': [], 'Train F1 Score': [], 'Train AUC': [], 
                                          'Test Accuracy': [], 'Test F1 Score': [], 'Test AUC': []} 
                                  for model in ['mlp', 'rf', 'xgb']}
    for run in range(runs):
        data = read_data('bluebottle', sample = True)
        X_train, X_test, y_train, y_test, continuous_var, smote_data = split_data(data, selected_features=X_selected, run = run, augment=False)
        best_models, all_models, model_metrics_dict = train_classifiers(X_train, y_train, X_test, y_test, run, aggregate_metrics_all_runs, model_gan=True)
        if run == runs - 1:
            for model_name, metrics in model_metrics_dict.items():
                print(f"Model: {model_name}")
                # Print Confusion Matrices
                print("Confusion Matrix (Train):")
                print(metrics['Confusion Matrix Train'])
                print("Confusion Matrix (Test):")
                print(metrics['Confusion Matrix Test'])

                # Print Classification Report for Train with four decimal places
                print("Classification Report (Train):")
                train_report = pd.DataFrame(metrics['Classification Report Train']).transpose()
                print(train_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))

                # Print Classification Report for Test with four decimal places
                print("Classification Report (Test):")
                test_report = pd.DataFrame(metrics['Classification Report Test']).transpose()
                print(test_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))
            continuous_analysis(data = data, continuous_vars = continuous_var)
    
    for model_name, metrics in aggregate_metrics_all_runs.items():
        print(f"Model: {model_name}")
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")    
    plot_metrics(best_models, all_models, X_train)

if __name__ == "__main__":
    main()