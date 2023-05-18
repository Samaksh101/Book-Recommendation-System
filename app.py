# Importing Important Libraries...
import pandas as pd
import numpy as np
from split import split_data
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for

# Initializing Flask App...
app = Flask(__name__)

# Getting Final Ratings Table
final_ratings = pd.read_csv('D:/Hiren/MTech/Semester 2/Recommendation System/Item_Item_Collaborative/ratings_final.csv')
# Fetching train data
train_data = pd.read_csv('D:/Hiren/MTech/Semester 2/Recommendation System/Item_Item_Collaborative/train.csv')
# Fetching test data
test_data = pd.read_csv('D:/Hiren/MTech/Semester 2/Recommendation System/Item_Item_Collaborative/test.csv')

# create pivot table for train data
train_pt = train_data.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')
# calculate similarity scores
similarity_scores = cosine_similarity(train_pt.fillna(0))
# create pivot table for test data
test_pt = test_data.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')

def get_top_k_recommendations(user_id, k, rated_items=[]):
    """
    Get top k recommendations for a given user ID.
    Returns:
    List of top k recommended recipe names for the user.
    """
    if user_id not in train_pt.columns:
        return []

    # update user's ratings with rated items from previous rounds
    for book, rating in rated_items:
        if book in train_pt.index:
            train_pt[user_id][book] = rating

    # Fetching the rated books from train pivot table for given user_id 
    rated_books = train_pt[user_id]
    rated_books = rated_books[rated_books.notna()]
    # Finding the unrated books for given user id
    unrated_books = train_pt.index.difference(rated_books.index)
    predicted_ratings = []

    '''
    Predicting Rating for each book in unrated books.
    '''
    for book in unrated_books:
        similarity_scores_for_book = similarity_scores[train_pt.index.get_loc(book)]
        user_ratings = train_pt[user_id].fillna(0)
        numerator = (user_ratings * similarity_scores_for_book).sum()
        denominator = similarity_scores_for_book.sum()

        if denominator == 0:
            predicted_rating = 0
        else:
            predicted_rating = numerator / denominator

        predicted_ratings.append((book, predicted_rating))
        
    top_book_names = [book for book, predicted_ratings in sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:k]]
    return top_book_names


# define route for login page
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        # check if user_id and password match
        user_id = request.form['user_id']
        password = request.form['password']
        if user_id == password:
            return redirect(url_for('recommendations', user_id=int(user_id)))
        else:
            error = 'Invalid credentials'
    # return render_template('login.html', error=error)
    return render_template('./index.html', error=error)

# define route for recommendation page
@app.route('/recommendations/<user_id>', methods=['GET', 'POST'])
def recommendations(user_id):
    # define parameters for recommendation system
    k = 5
    rated_items = []

    if request.method == 'POST':
        # get selected item and rating from form
        selected_item = request.form['selected_item']
        rating = float(request.form['rating'])
        rated_items.append((selected_item, rating))
        
        # check if user is satisfied
        if request.form['satisfied'] == 'yes':
            # get ingredients of selected item and display to user
            title = final_ratings[final_ratings['ISBN'] == selected_item]['Book-Title'].values[0]
            author = final_ratings[final_ratings['ISBN'] == selected_item]['Book-Author'].values[0]
            publisher = final_ratings[final_ratings['ISBN'] == selected_item]['Publisher'].values[0]
            image = final_ratings[final_ratings['ISBN'] == selected_item]['Image-URL-L'].values[0]
            year = final_ratings[final_ratings['ISBN'] == selected_item]['Year-Of-Publication'].values[0]
            
            book = {
            "title": title,
            "author": author,
            "publisher": publisher,
            "year": year,
            "cover": image
            }

            return render_template('templates\book_details.html', recipe=selected_item, book=book)

    # get top k recommendations for user
    user_id = int(user_id)
    rec = []
    recommendations = get_top_k_recommendations(user_id, k, rated_items)
    for recommendation in recommendations:
        rec.append((recommendation,final_ratings[final_ratings['ISBN'] == recommendation]['Book-Title'].values[0]))
    # render template for recommendation page
    return render_template('templates\recommendations.html', user_id=user_id, recommendations=rec)

if __name__ == '__main__':
    app.run(debug=True)
