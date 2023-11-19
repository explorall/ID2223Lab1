import pandas as pd
import random
import numpy as np
import hopsworks

def generate_wine(cut_df):
    """
    Returns a generated wine as a single row in a DataFrame
    """
    means = cut_df.drop(["type"], axis=1).mean()
    stds = cut_df.drop(["type"], axis=1).std()
    
    df = pd.DataFrame({ "fixed_acidity": [np.random.normal(means["fixed_acidity"],stds["fixed_acidity"])],
                       "volatile_acidity": [np.random.normal(means["volatile_acidity"],stds["volatile_acidity"])],
                       "citric_acid": [np.random.normal(means["citric_acid"],stds["citric_acid"])],
                       "residual_sugar": [np.random.normal(means["residual_sugar"],stds["residual_sugar"])],
                       "chlorides": [np.random.normal(means["chlorides"],stds["chlorides"])],
                       "free_sulfur_dioxide": [np.random.normal(means["free_sulfur_dioxide"],stds["free_sulfur_dioxide"])],
                       "total_sulfur_dioxide": [np.random.normal(means["total_sulfur_dioxide"],stds["total_sulfur_dioxide"])],
                       "density": [np.random.normal(means["density"],stds["density"])],
                       "ph": [np.random.normal(means["ph"],stds["ph"])],
                       "sulphates": [np.random.normal(means["sulphates"],stds["sulphates"])],
                       "alcohol": [np.random.normal(means["alcohol"],stds["alcohol"])],
                       "quality": [int(means["quality"])],
                       "type": [cut_df["type"].iat[0]]
                      })
    
    return df

def get_random_wine(wine_fg):
    """
    Returns a DataFrame containing one random wine
    """

    wine_df = wine_fg.read()
    quality_df = wine_df[["quality"]]

    #Randomly pick quality and type
    #Pick quality from a normal dist
    random_quality = int(np.random.normal(quality_df.mean(),quality_df.std()))
    if random_quality<3:
        random_quality=3
    if random_quality>9:
        random_quality=9
    #Pick type uniformly
    random_type=["white","red"]
    random_type=random_type[np.random.randint(0,2)]

    print("Add {} wine with quality {}".format(random_type, random_quality))

    #Cut all rows not fitting quality and type
    cut_df=wine_df.query("quality=={} & type=='{}'".format(random_quality,random_type))

    #Generate random data frame row
    return_df = generate_wine(cut_df)

    return return_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine",version=1)

    wine_df_row = get_random_wine(wine_fg)
    
    wine_fg.insert(wine_df_row)

if __name__ == "__main__":
    g()