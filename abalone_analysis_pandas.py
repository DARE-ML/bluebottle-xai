import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    # Define the file path where the data is located
    file_path = r'C:\Users\USER\Desktop\UNSW\abalone (2)\abalone.data'
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

# Read the data from the URL directly
    data = pd.read_csv(file_path, header=None, names=column_names)
    data['Sex'] = data['Sex'].replace({'M': 0, 'F': 1, 'I':2})
    return data

def features_data(data):
    features = data.iloc[:,0:8]
    return features

def features_histogram(features):
    plt.style.use("ggplot")
    features.hist(bins=10, figsize = (15, 10))
    plt.suptitle('Histograms of Each Feature')
    plt.savefig('Histograms.png')
    plt.tight_layout()
    plt.show()

def covariance_Plot(features):
    cov_matrix = features.cov()
    print(cov_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=.5)
    plt.title('Covariance Matrix')
    plt.savefig('Covariance_Matrix.png')
    plt.show()

def boxPlot_features(features):
    plt.figure(figsize=(12, 8))
    features.boxplot()
    plt.title('Box Plot of Each Feature')
    plt.xticks(rotation=45)
    plt.savefig('Boxplot.png')
    plt.show()

def statistics(data):
    mean = data.mean()
    std = data.std()
    print("The mean of the features and outcome variable")
    print(mean)
    print("standard deviation of the features and outcome varialbe")
    print(std)

def ring_histogram(data):
    rings = data['Rings']
    plt.style.use("ggplot")
    rings.hist(bins=10, figsize = (15, 10))
    plt.suptitle('Histograms of outcome variable')
    plt.savefig('Histograms.png')
    plt.tight_layout()
    plt.show()

def ring_categories(data):
    rings = data['Rings']
    quartiles = rings.quantile([0, 0.25, 0.5, 0.75, 1])
    print("Ring Age Quartiles:")
    print(quartiles)
    groups = pd.cut(rings, bins=quartiles, right=True, labels=False)
# Count the number of samples in each group
    class_distribution = groups.value_counts().sort_index()
# Print class distribution of each group
    print("Class Distribution of Each Group:")
    for i, count in enumerate(class_distribution[1:], start=1):
        print(f"Group {i}: {count} samples")
    return class_distribution
    
    #for group, count in class_distribution.items():
        #print(f"Group {group}: {count} samples")
        #return class_distribution
  
    # Check if it's a class imbalance problem
    is_imbalanced = class_distribution.std() > 0
    print("\nIs this a class imbalance problem?", "Yes" if is_imbalanced else "No")

def hot_encoding(categories):
    # One-hot encode the categories using pandas get_dummies()
    one_hot_encoded = pd.get_dummies(categories, prefix='category')
    return one_hot_encoded
def main():
    data = read_data()
    features = features_data(data)
    features_histogram(features)
    covariance_Plot(features)
    boxPlot_features(features)
    statistics(data)
    ring_histogram(data)
    class_distribution = ring_categories(data)
    one_hot_encoding = hot_encoding(class_distribution)
    print("One-hot encoded matrix:")
    print(one_hot_encoding)
if __name__ == "__main__":
    main()