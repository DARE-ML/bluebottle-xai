from packages import *
from model_bluebottledata import *

def discretize(data):
    noncircular_vars = ['SST', 'Curr_Speed', 'WD_Speed']
    circular_vars = ['Curr_Dir', 'WD_Dir']
    selected_features = [
        'SST', 'Curr_Speed', 'Curr_Dir', 'WD_Speed', 'WD_Dir', 
        'Month_Jan', 'Month_Feb', 'Month_Dec', 'Month_Oct', 'Month_Nov', 
        'Month_Sep', 'Month_Mar', 'Month_Apr', 'Presence'
    ]
    data_selected = data[selected_features]

    # Discretize continuous variables
    n_bins = 4
    strategy = 'uniform'
    kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    data_cont = kbins.fit_transform(data_selected[noncircular_vars])
    df_binned = pd.DataFrame(data_cont, columns=noncircular_vars)
    labels = ['L', 'M', 'H', 'VH']
    
    # Label encoding for continuous variables
    for i, var in enumerate(noncircular_vars):
        bin_edges = kbins.bin_edges_[i]
        df_binned[var] = df_binned[var].apply(lambda x: labels[int(x)])
        print(f"Bin edges for '{var}':", bin_edges)
        for j in range(len(bin_edges) - 1):
            print(f"{labels[j]}: {bin_edges[j]} to {bin_edges[j+1]}")
        print()

    # Concatenate binned and remaining data
    data2 = data_selected.reset_index(drop=True)
    df_binned = df_binned.reset_index(drop=True)
    data3 = pd.concat([data2.drop(columns=noncircular_vars), df_binned], axis=1)

    # Discretize circular variables with labels
    n_bins_circular = 8
    kbins_circular = KBinsDiscretizer(n_bins=n_bins_circular, encode='ordinal', strategy='uniform')
    data3[circular_vars] = kbins_circular.fit_transform(data3[circular_vars]).astype(int)
    circular_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for column in circular_vars:
        data3[column] = data3[column].map(lambda x: circular_labels[x])

    # One-hot encode categorical variables
    new_cat = ['Month', 'SST', 'Curr_Speed', 'Curr_Dir', 'WD_Speed', 'WD_Dir']
    new_cat = [col for col in new_cat if col in data3.columns]
    encoder = OneHotEncoder()
    encoded_df = encoder.fit_transform(data3[new_cat])
    encoded_df = pd.DataFrame(encoded_df.toarray(), columns=encoder.get_feature_names_out(new_cat))

    # Concatenate encoded_df and remaining data
    new_data = pd.concat([encoded_df, data3.drop(columns=new_cat)], axis=1)
    print(new_data.columns)
    print(new_data.head(10))
    
    return new_data

def main():
    data = pd.read_csv('CTGAN_data.csv')
    selected_features = 'yes'
    runs = 30
    aggregate_metrics_all_runs = {
        model: {'Train Accuracy': [], 'Train F1 Score': [], 'Train AUC': [], 
                'Test Accuracy': [], 'Test F1 Score': [], 'Test AUC': []} 
        for model in ['mlp', 'rf', 'xgb']
    }
    for run in range(runs):
        new_data = discretize(data)
        
        # Split data and check for potential None values
        X_train, X_test, y_train, y_test, continuous_vars, smote_data = split_data(new_data, selected_features=selected_features, run=run, augment=False, normalize=False)
        
        best_models, all_models, model_metrics_dict = train_classifiers(X_train, y_train, X_test, y_test, run, aggregate_metrics_all_runs, model_gan=True)
        
        # Print metrics for the last run
        if run == runs - 1:
            for model_name, metrics in model_metrics_dict.items():
                print(f"Model: {model_name}")
                print("Confusion Matrix (Train):", metrics['Confusion Matrix Train'])
                print("Confusion Matrix (Test):", metrics['Confusion Matrix Test'])

                # Format classification report with four decimal places
                print("Classification Report (Train):")
                train_report = pd.DataFrame(metrics['Classification Report Train']).transpose()
                print(train_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))
                print("Classification Report (Test):")
                test_report = pd.DataFrame(metrics['Classification Report Test']).transpose()
                print(test_report.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))
            
    # Print aggregate metrics for all runs
    for model_name, metrics in aggregate_metrics_all_runs.items():
        print(f"Model: {model_name}")
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")    
    plot_metrics(best_models, all_models, X_train)

if __name__ == "__main__":
    main()