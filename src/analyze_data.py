import pandas as pd
def correlation_analysis():
    directory = "../data/"
    datasets = [('business.csv', "stars"), ('train_reviews.csv',"stars"), ('users.csv', "average_stars")]
    for data in datasets: 
        df = pd.read_csv(directory + data[0])
        corr_df = df.corr("kendall")
        print(corr_df[data[1]])
        
correlation_analysis()
