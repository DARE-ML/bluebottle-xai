from packages import *
def read_data(data):
    # Load and preprocess the data
    data['Month'] = data['Date'].dt.strftime('%b')
    data = data[['Month', 'Beach', 'Council_Report', 'Beach_Key', 'Surf_Club', 'Lat', 'Lon', 'Orient', 'Embayment', 'SST',
                'Curr_Speed', 'Curr_Dir', 'WD_Speed', 'WD_Dir', 'Presence']]
    categorical_vars = ['Month', 'Beach', 'Council_Report', 'Beach_Key', 'Surf_Club', 'Lat', 'Lon', 'Orient', 'Embayment', 'Presence']
    continuous_vars = ['SST', 'Curr_Speed', 'Curr_Dir', 'WD_Speed', 'WD_Dir']
    data1 = data[['Presence', 'Month', 'Beach', 'SST', 'Curr_Speed', 'Curr_Dir', 'WD_Speed', 'WD_Dir']]
    new_cat = ['Month', 'Beach', 'Presence']
    encoded_df = pd.get_dummies(data[new_cat])
    corr_data = pd.concat([encoded_df, data1.drop(columns=new_cat)], axis=1)
    return data,categorical_vars, continuous_vars, corr_data

def continuous_analysis(data, continuous_vars):
    y = data['Presence']
    plt.style.use("ggplot")
    ax = data[continuous_vars].hist(bins=30, figsize=(20, 15))
    ax = ax.flatten()
    # Loop through each axis and label the x and y axes, ensuring the index is within bounds
    for i in range(min(len(ax), len(continuous_vars))):
        ax[i].set_xlabel(continuous_vars[i], fontsize=18, color='black') 
        ax[i].set_ylabel('Frequency', fontsize=18, color='black')         
        ax[i].tick_params(axis='both', which='major', labelsize=18, colors='black') 
        ax[i].set_title('')  # Remove title 
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig('Histogram of Continuous Variables.svg')
    plt.show()

    # Prepare data for the pairplot
    data_subset = data[continuous_vars].copy()
    data_subset['Presence'] = y  # Add the target (presence) for color codin
    # Create a pairplot
    pairplot = sns.pairplot(data_subset, hue='Presence', markers=["o", "s"], diag_kind='kde')
    # Customize legend
    pairplot._legend.set_title('Presence')
    pairplot._legend.set_bbox_to_anchor((1.05, 1))
    pairplot._legend.set_loc('upper left')
    pairplot._legend.get_title().set_fontsize(18)
    for text in pairplot._legend.get_texts():
        text.set_fontsize(18)
        text.set_color('black')
    # Adjust tick parameters for pairplot axes and remove titles
    for ax in pairplot.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=18, color='black')
        ax.set_ylabel(ax.get_ylabel(), fontsize=18, color='black')
        ax.tick_params(axis='both', which='major', labelsize=18, colors='black')  # Set tick label size and color
        ax.set_title('')  # Remove title
    plt.savefig('Pairplot.svg')
    plt.show()

def histogram_plots(data, continuous_vars):
    circular_vars = ['Curr_Dir', 'WD_Dir']
    non_circular_vars = [var for var in continuous_vars if var not in circular_vars]
    # Plot circular histograms
    fig, axes = plt.subplots(1, len(circular_vars), subplot_kw=dict(polar=True), figsize=(18, 6))
    for i, var in enumerate(circular_vars):
        angles = np.deg2rad(data[var])
        n_bins = 30
        bins = np.linspace(0, 2 * np.pi, n_bins + 1)     
        ax = axes[i]
        ax.hist(angles, bins=bins, edgecolor='k', alpha=0.7)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_title(f'Circular Histogram of {var}', fontsize=20, color='black')
        ax.tick_params(axis='x', labelsize=18, labelcolor='black')
        ax.tick_params(axis='y', labelsize=18, labelcolor='black')
    plt.tight_layout()
    plt.savefig('Circular_histograms.svg')
    plt.show()

    # Plot non-circular variable histograms in subplots
    num_vars = len(non_circular_vars)
    n_cols = 3  # Number of columns for subplots
    n_rows = (num_vars + n_cols - 1) // n_cols  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
    axes = axes.flatten()  # Flatten axes to make indexing easier
    for i, var in enumerate(non_circular_vars):
        sns.histplot(data=data, x=var, kde=True, ax=axes[i], edgecolor='k')
        axes[i].set_xlabel(var, fontsize=18, color='black')
        axes[i].set_ylabel('Frequency', fontsize=18, color='black')
        # Set tick label sizes to match axis labels
        axes[i].tick_params(axis='x', labelsize=18, labelcolor='black')
        axes[i].tick_params(axis='y', labelsize=18, labelcolor='black')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig('Density_plots.svg')
    plt.show()

def histogram_plot(data, continuous_vars):
    circular_vars = ['Curr_Dir', 'WD_Dir']
    non_circular_vars = [var for var in continuous_vars if var not in circular_vars]

    for var in circular_vars:
        angles = np.deg2rad(data[var])
        n_bins = 30
        bins = np.linspace(0, 2 * np.pi, n_bins + 1)

        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 6))
        ax.hist(angles, bins=bins, edgecolor='k', alpha=0.7)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_title(f'Circular Histogram of {var}', fontsize=14)
        plt.xticks(fontsize=12, color='black')
        plt.yticks(fontsize=12, color='black')
        plt.savefig(f'Cicular_plot_of_{var}.svg')  
    plt.tight_layout()
    plt.show()
    # Create histograms for each non-circular variable
    for var in non_circular_vars:
        sns.displot(data=data, x=var, kind='hist', kde=True, aspect=1.5)
        #plt.title(f'Histogram of {var}', fontsize=20)
        plt.xlabel(var, fontsize=14, color='black')
        plt.ylabel('Frequency', fontsize = 14, color='black')
        plt.xticks(fontsize=14, color='black')
        plt.yticks(fontsize=14, color='black')
        plt.tight_layout()
        plt.savefig(f'Density_plot_of_{var}.svg')  
        plt.show() 

def continuous_corr(data, continuous_vars):
    contin_corr = data[continuous_vars].corr(method='spearman')
    sns.heatmap(contin_corr, vmin=-1, vmax=1, annot=True, fmt='.3f', cmap='coolwarm', annot_kws={"size": 8, "rotation": 0, "color": "black"})
    plt.xticks(fontsize=8, color='black')
    plt.yticks(fontsize=8, color='black')
    colorbar = plt.gca().collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(labelsize=8, labelcolor='black', width=1.5)
    colorbar.ax.yaxis.set_tick_params(width=1.5)
    for label in colorbar.ax.get_yticklabels():
        label.set_color('black')
    plt.savefig('Spearman Correlation Heatmap.svg')
    plt.show()

def cat_corr(data, categorical_vars):
    chi2_results = []
    for i, var1 in enumerate(categorical_vars):
        for j, var2 in enumerate(categorical_vars):
            if i < j:
                contingency_table = pd.crosstab(data[var1], data[var2])
                chi2, p_value, DoF, expected = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, k = contingency_table.shape
                cramers_v = np.sqrt(phi2 / min((k - 1), (r - 1)))
                chi2_results.append({
                    'variable 1': var1,
                    "variable 2": var2,
                    "Chi_square": round(chi2, 4),
                    'P_value': round(p_value, 4),
                    "Cramers V": round(cramers_v, 4)
                })
    # Create DataFrame and pivot for heatmap
    cat_results = pd.DataFrame(chi2_results)
    cat_corr_matrix = cat_results.pivot(index="variable 1", columns="variable 2", values="Cramers V")
    # Plot heatmap without variable labels on axes
    sns.heatmap(cat_corr_matrix, vmin=-1, vmax=1, fmt='.2f', annot=True, cmap='coolwarm', annot_kws={"size": 8, "rotation": 0, "color": "black"})
    plt.xticks(fontsize=8, color="black")
    plt.yticks(fontsize=8, color="black")
    plt.xlabel('')  # Remove X-axis label
    plt.ylabel('')  # Remove Y-axis label
    colorbar = plt.gca().collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(labelsize=8, labelcolor='black', width=1.5)
    colorbar.ax.yaxis.set_tick_params(width=1.5)
    for label in colorbar.ax.get_yticklabels():
        label.set_color('black')
    plt.savefig('Cramers V Correlation Heatmap.svg')
    plt.show()
    return cat_results

def cat_contin_corr(corr_data, continuous_vars):
    binary_catVars = [col for col in corr_data.columns if corr_data[col].nunique() == 2]
    point_biserialr_results = []
    for cont_var in continuous_vars:
        for bin_cat_var in binary_catVars:
            correlation, p_value = stats.pointbiserialr(corr_data[cont_var], corr_data[bin_cat_var])
            point_biserialr_results.append({'Continuous Variable': cont_var,
                                            'Binary Categorical Variable': bin_cat_var,
                                            'Correlation': round(correlation, 3),
                                            'P_value': round(p_value, 3)})
    point_biserialr_df = pd.DataFrame(point_biserialr_results)
    point_corr_matrix = point_biserialr_df.pivot(index="Continuous Variable", columns="Binary Categorical Variable", values="Correlation")
    sns.heatmap(point_corr_matrix, vmin=-1, vmax=1, cmap='coolwarm', annot=True, linewidths=1.5, fmt='.2f', annot_kws={"size": 8, "rotation": 90, "color": "black"})
    plt.xticks(fontsize=8, color='black')
    plt.yticks(fontsize=8, color='black')
    plt.xlabel('Continuous Variable', fontsize=8, color='black')  # Remove X-axis label
    plt.ylabel('Categorical Variable', fontsize=8, color='black')
    colorbar = plt.gca().collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(labelsize=8, labelcolor='black', width=1.5)
    colorbar.ax.yaxis.set_tick_params(width=1.5)
    for label in colorbar.ax.get_yticklabels():
        label.set_color('black')
    plt.savefig('Point Biserial Correlation Heatmap.svg')
    plt.show()
    return point_biserialr_df

def skewness(corr_data, continuous_vars):
    skewness = corr_data[continuous_vars].skew()
    deskew_data = ['Curr_Speed', 'WD_Speed']
    # Print skewness before transformation
    print("Skewness before log transformation:")
    print(skewness)
    # Copy the columns to be transformed
    transformed_data = corr_data[deskew_data].copy()
    # Apply log transformation to the selected columns
    for col in deskew_data:
        transformed_data[col] = np.log1p(corr_data[col])  
    # Replace the original columns with the transformed columns in the main DataFrame
    corr_data[deskew_data] = transformed_data
    # Print skewness after transformation
    print("Skewness after log transformation:")
    print(corr_data[continuous_vars].skew())
    return corr_data

def presence_outlier(corr_data):
    plt.figure(figsize=(10, 5))
    sns.countplot(x="Presence", data=corr_data)
    plt.title('Presence Distribution')
    plt.show()
    if 'Beach' in corr_data.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=corr_data, x="Beach", hue="Presence")
        plt.title('Presence in Each Beach')
        plt.show()

def handle_outliers(corr_data):
    corr_data.plot(kind='box', subplots=True, layout=(6, 3), sharex=False, sharey=False, figsize=(20, 15))
    plt.show()
    col_outliers = corr_data[['Curr_Speed', 'WD_Speed']]
    data2 = corr_data.copy()
    for col in col_outliers:
        q1, q3 = np.percentile(data2[col], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data2 = data2[(data2[col] >= lower_bound) & (data2[col] <= upper_bound)]

    print(data2['Presence'].value_counts())
    return data2

def handle_outliers(corr_data):
    corr_data.plot(kind='box', subplots=True, layout=(6, 3), sharex=False, sharey=False, figsize=(20, 15))
    plt.show()
    col_outliers = corr_data[['Curr_Speed', 'WD_Speed']]
    data2 = corr_data.copy()
    for col in col_outliers:
        q1, q3 = np.percentile(data2[col], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data2 = data2[(data2[col] >= lower_bound) & (data2[col] <= upper_bound)]     
    plt.pie(data2['Presence'].value_counts(), labels = ['Absence', 'Presence'], autopct='%.f', shadow=True)
    plt.show()
    print(data2['Presence'].value_counts())
    return data2
def main():
    data = pd.read_csv('Randwick_council.csv', parse_dates=['Date'])
    data, categorical_vars, continuous_vars, corr_data = read_data(data)
    continuous_analysis(data, continuous_vars)
    histogram_plots(data, continuous_vars)
    histogram_plot(data, continuous_vars)
    continuous_corr(data, continuous_vars)
    cat_corr(data, categorical_vars)
    cat_contin_corr(corr_data, continuous_vars)
    corr_data = skewness(corr_data, continuous_vars)
    presence_outlier(corr_data)
    data2 = handle_outliers(corr_data)

if __name__ == "__main__":
    main()