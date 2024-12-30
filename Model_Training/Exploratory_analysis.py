from packages import *
def read_data(data):
    # Load and preprocess the data
    data['Month'] = data['Date'].dt.strftime('%b')
    data = data[['Month', 'Beach', 'Council_Report', 'Beach_Key', 'Surf_Club', 'Lat', 'Lon', 'Orient', 'Embayment', 'SST',
                'Current Speed', 'Current Direction', 'Wind Speed', 'Wind Direction', 'Presence']]
    categorical_vars = ['Month', 'Beach', 'Council_Report', 'Beach_Key', 'Surf_Club', 'Lat', 'Lon', 'Orient', 'Embayment', 'Presence']
    continuous_vars = ['SST', 'Current Speed', 'Current Direction', 'Wind Speed', 'Wind Direction']
    data1 = data[['Presence', 'Month', 'Beach', 'SST', 'Current Speed', 'Current Direction', 'Wind Speed', 'Wind Direction']]
    new_cat = ['Month', 'Beach', 'Presence']
    encoded_df = pd.get_dummies(data[new_cat])
    corr_data = pd.concat([encoded_df, data1.drop(columns=new_cat)], axis=1)
    stats = data.groupby('Beach')[continuous_vars].agg(['mean', 'std'])
    print(stats)
    return data,categorical_vars, continuous_vars, corr_data
def seasonal_analysis(data):
    data['Presence'] = data['Presence'].astype(int)

    # Extract month from the 'Date' column
    data['Month'] = data['Date'].dt.month_name()  # Extract month name

    # Group by Month and sum the Presence counts
    monthly_presence_count = data.groupby('Month')['Presence'].sum().reset_index()

    # Sort months to have them in calendar order
    monthly_presence_count['Month'] = pd.Categorical(monthly_presence_count['Month'], 
                                                    categories=['January', 'February', 'March', 
                                                                'April', 'May', 'June', 
                                                                'July', 'August', 'September', 
                                                                'October', 'November', 'December'], 
                                                    ordered=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(data=monthly_presence_count, x='Month', y='Presence', palette='viridis', width=0.6)
    #plt.title('Total Count of Bluebottle Presence per Month', fontsize=16)
    plt.xlabel('Month', fontsize=14, color='black')
    plt.ylabel('Count of Presence', fontsize=14, color='black')
    plt.xticks(rotation=45, fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')
    plt.legend(fontsize=12)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def yearly_analysis(data):
    data['Year'] = data['Date'].dt.year
    data['Presence'] = data['Presence'].astype(int)
    data['Presence_Label'] = data['Presence'].map({0: 'Absence', 1: 'Presence'})
    yearly_beach_presence_count = data.groupby(['Year', 'Beach'])['Presence'].sum().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    yearly_beach_presence_count.plot(kind='line', figsize=(12, 6))
    plt.xlabel('Year', fontsize=22, color='black')
    plt.ylabel('Count of Presence', fontsize=22, color='black')

    plt.xticks(ticks=yearly_beach_presence_count.index, labels=yearly_beach_presence_count.index.astype(int), 
        rotation=45,fontsize=20,color='black')

    plt.yticks(fontsize=20, color='black')
    plt.legend(title='Beach', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, title_fontsize=20)
    plt.tight_layout()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='Beach', hue='Presence_Label', width=0.6)
    #plt.title('Distribution of Bluebottle Presence and Absence by Beach')
    plt.xlabel('Beach', fontsize=14, color='black')
    plt.ylabel('Count', fontsize=14, color='black')
    plt.xticks(rotation=45, fontsize=14, color='black')
    plt.yticks(fontsize=14, color='black')
    plt.tight_layout()
    plt.grid()
    plt.show()

    beach_counts = data.groupby('Beach')['Presence_Label'].value_counts().unstack(fill_value=0)
    print(beach_counts)

def map_plot(data):
    gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['Lon'], data['Lat']),
    crs="EPSG:4326"  
    )
    gdf = gdf.to_crs(epsg=3857)

    print("X min:", gdf.geometry.x.min())
    print("X max:", gdf.geometry.x.max())
    print("Y min:", gdf.geometry.y.min())
    print("Y max:", gdf.geometry.y.max())
    # Define bounding box (crop to beach regions)
    buffer = 3000  # Adjust this value as needed
    x_min, x_max = gdf.geometry.x.min() - buffer, gdf.geometry.x.max() + buffer
    y_min, y_max = gdf.geometry.y.min() - buffer, gdf.geometry.y.max() + buffer 
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot the GeoDataFrame
    gdf.plot(ax=ax, color='blue', markersize=400, alpha=0.7, zorder=2)
    #plt.title('Bluebottle Presence in Eastern Beaches of Sydney')
    plt.xlabel('Longitude', fontsize=16, color='black')
    plt.ylabel('Latitude', fontsize=16, color='black')
 
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.tick_params(axis='both', which='major', labelsize=14, color='black')
    # Add a basemap
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap='Set1', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax)
    #cbar.set_label('Bluebottle Presence', fontsize=16)
    plt.show()

def continuous_analysis(data, continuous_vars):
    y = data['Presence']
    plt.style.use("ggplot")

    # Create a grid layout with enough columns to allow centering the last plot
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))  # 2 rows, 3 columns
    ax = ax.flatten()

    # Loop through continuous variables and plot histograms
    for i in range(len(continuous_vars)):
        data[continuous_vars[i]].hist(bins=30, ax=ax[i], color='skyblue')
        ax[i].set_xlabel(continuous_vars[i], fontsize=26, color='black') 
        ax[i].set_ylabel('Frequency', fontsize=26, color='black')         
        ax[i].tick_params(axis='both', which='major', labelsize=24, colors='black') 
        ax[i].set_title('')  # Remove title 

    for j in range(len(continuous_vars), len(ax)):
        fig.delaxes(ax[j])  

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
    circular_vars = ['Current Direction', 'Wind Direction']
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
    circular_vars = ['Current Direction', 'Wind Direction']
    non_circular_vars = [var for var in continuous_vars if var not in circular_vars]

    for var in circular_vars:
        angles = np.deg2rad(data[var])
        n_bins = 30
        bins = np.linspace(0, 2 * np.pi, n_bins + 1)

        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 6))
        ax.hist(angles, bins=bins, edgecolor='k', alpha=0.7)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        #ax.set_title(f'Circular Histogram of {var}', fontsize=14)
        plt.xticks(fontsize=14, color='black')
        plt.yticks(fontsize=14, color='black')
        plt.savefig(f'Cicular_plot_of_{var}.svg')  
    plt.tight_layout()
    plt.show()
    # Create histograms for each non-circular variable
    for var in non_circular_vars:
        sns.displot(data=data, x=var, kind='hist', kde=True, aspect=1.5)
        #plt.title(f'Histogram of {var}', fontsize=20)
        plt.xlabel(var, fontsize=18, color='black')
        plt.ylabel('Frequency', fontsize = 18, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.tight_layout()
        plt.savefig(f'Density_plot_of_{var}.svg')  
        plt.show() 

def continuous_corr(data, continuous_vars):
    contin_corr = data[continuous_vars].corr(method='spearman')
    sns.heatmap(contin_corr, vmin=-1, vmax=1, annot=True, fmt='.3f', cmap='coolwarm', annot_kws={"size": 11, "rotation": 45, "color": "black"})
    plt.xticks(fontsize=11, color='black')
    plt.yticks(fontsize=11, color='black')
    colorbar = plt.gca().collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(labelsize=11, labelcolor='black', width=1.5)
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
    sns.heatmap(point_corr_matrix, vmin=-1, vmax=1, cmap='coolwarm', annot=True, linewidths=1.5, fmt='.2f', annot_kws={"size": 10, "rotation": 90, "color": "black"})
    plt.xticks(fontsize=8, color='black')
    plt.yticks(fontsize=8, color='black')
    plt.xlabel('Categorical Variable', fontsize=10, color='black')  # Remove X-axis label
    plt.ylabel('Continuous Variable', fontsize=10, color='black')
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
    deskew_data = ['Current Speed', 'Wind Speed']
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


def handle_outliers(corr_data):
    corr_data.plot(kind='box', subplots=True, layout=(6, 3), sharex=False, sharey=False, figsize=(20, 15))
    plt.show()
    col_outliers = corr_data[['Current Speed', 'Wind Speed']]
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
    seasonal_analysis(data)
    yearly_analysis(data)
    map_plot(data)
    data, categorical_vars, continuous_vars, corr_data = read_data(data)
    continuous_analysis(data, continuous_vars)
    histogram_plots(data, continuous_vars)
    histogram_plot(data, continuous_vars)
    continuous_corr(data, continuous_vars)
    cat_corr(data, categorical_vars)
    cat_contin_corr(corr_data, continuous_vars)
    corr_data = skewness(corr_data, continuous_vars)
    data2 = handle_outliers(corr_data)
    print(data2.head())

if __name__ == "__main__":
    main()