import numpy as np
import pandas as pd


def split_data(final_ratings):
    # Group the data frame by the 'UserID' column
    grouped = final_ratings.groupby('User-ID')

    # Calculate the number of reviews for each user
    review_counts = grouped['ISBN'].count()

    # Create an empty list to store the test data frames
    test_data = []

    # Iterate through each group of reviews for a user
    for user, group in grouped:
        # Determining the number of reviews to include in the test data frame
        test_size = int(np.ceil(0.2 * len(group)))
        # Split the group of reviews into train and test data frames
        train = group.iloc[:-test_size]
        test = group.iloc[-test_size:]
        # Append the test data frame to the list of test data frames
        test_data.append(test)

    # Concatenate all of the test data frames into a single test data frame
    test_data = pd.concat(test_data)

    # Drop the groups of reviews from the original data frame that were split into the test data frame
    train_data = final_ratings[~final_ratings.index.isin(test_data.index)]

    return train_data, test_data