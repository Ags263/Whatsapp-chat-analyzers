import numpy as np
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
import calendar

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Fetch the number of messages
    num_messages = df.shape[0]

    # Fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Fetch the number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Fetch the number of links shared
    num_links = len(extract.find_urls(' '.join(df['message'])))

    return num_messages, len(words), num_media_messages, num_links


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def most_talkative_users_bar_chart(df):
    plt.figure(figsize=(12, 6))
    most_talkative = df['user'].value_counts().nlargest(10)
    sns.barplot(x=most_talkative.index, y=most_talkative.values, palette="Blues_d")
    plt.title("Top 10 Most Talkative Users")
    plt.xlabel("User")
    plt.ylabel("Message Count")
    plt.xticks(rotation='vertical')
    return plt

def hourly_timeline(selected_user, df):
    hourly_distribution_df = df.groupby('Hour').size().reset_index(name='Message Count')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Hour', y='Message Count', data=hourly_distribution_df, palette="Purples")
    plt.title("Hourly Message Distribution")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Message Count")

    return plt  # Return the Matplotlib chart


def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def weekday_chatting_circle(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    weekday_counts = df['day_name'].value_counts().sort_index()

    # Order weekdays starting from Monday
    weekdays = list(calendar.day_name)
    weekday_counts = weekday_counts.reindex(weekdays)

    # Create a polar plot for the weekdays
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    theta = np.linspace(0, 2*np.pi, len(weekdays), endpoint=False)
    bars = ax.bar(theta, weekday_counts, align="center", alpha=0.7)

    ax.set_xticks(theta)
    ax.set_xticklabels(weekdays)

    ax.set_title("Weekday Chatting Circle")

    return fig


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def sentiment_distribution_by_user_chart(df):
    # Calculate sentiment distribution
    sentiment_distribution = df.groupby(['user', 'sentiment']).size().unstack(fill_value=0)
    sentiment_distribution['Total'] = sentiment_distribution.sum(axis=1)
    sentiment_distribution_percentage = sentiment_distribution.div(sentiment_distribution['Total'], axis=0) * 100
    sentiment_distribution_percentage = sentiment_distribution_percentage.round(2)

    # Plot horizontal bar chart with cool colors
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different sentiments
    colors = {'Positive': 'blue', 'Neutral': 'gray', 'Negative': 'red'}

    # Plot horizontal bar chart for each user
    for user, row in sentiment_distribution_percentage.iterrows():
        ax.barh(user, row.drop('Total'), color=[colors[sentiment] for sentiment in row.index.drop('Total')])

    ax.set_title("Sentiment Distribution by User")
    ax.set_xlabel("Percentage")
    ax.set_ylabel("User")
    ax.legend(title='Sentiment', bbox_to_anchor=(1, 1), loc='upper left')

    return fig

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def most_active_user_by_sentiment(df, sentiment):
    if sentiment is not None:
        df = df[df['sentiment'] == sentiment]

    most_active_user = df['user'].value_counts().idxmax()
    return most_active_user


def top_positive_reviews(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Perform sentiment analysis on messages
    df['message_sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Select top 5 reviews with the highest positive sentiment polarity score
    top_positive_reviews_df = df[df['message_sentiment'] > 0].nlargest(5, 'message_sentiment')[['user', 'message', 'message_sentiment']]

    return top_positive_reviews_df



def hourly_timeline(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    hourly_distribution = df['hour'].value_counts().sort_index().reset_index()
    hourly_distribution.columns = ['Hour', 'Message Count']

    return hourly_distribution


def most_active_date(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    active_date = df['only_date'].value_counts().idxmax()

    return active_date


def most_and_least_busy_day(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    busy_day_counts = df['day_name'].value_counts()
    most_busy_day = busy_day_counts.idxmax()
    least_busy_day = busy_day_counts.idxmin()

    return most_busy_day, least_busy_day

def most_and_least_busy_month(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    busy_month_counts = df['month'].value_counts()
    most_busy_month = busy_month_counts.idxmax()
    least_busy_month = busy_month_counts.idxmin()

    return most_busy_month, least_busy_month

def num_deleted_messages(df):
    return df[df['deleted']].shape[0]


def most_talkative_users_bar_chart(df, num_users=10):

    if 'user' not in df.columns or 'message' not in df.columns:
        raise ValueError("The DataFrame should have 'user' and 'message' columns.")

    user_message_counts = df['user'].value_counts().head(num_users)

    # Create a Seaborn bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=user_message_counts.values, y=user_message_counts.index, palette="viridis")

    plt.title(f"Top {num_users} Most Talkative Users")
    plt.xlabel("Message Count")
    plt.ylabel("User")
    plt.grid(axis="x")

    # Invert y-axis for better readability (top user at the top)
    plt.gca().invert_yaxis()

    # Save the chart for Streamlit
    fig = plt.gcf()

    return fig

def most_abusive_users_bar_chart(df, num_users=10):

    if 'user' not in df.columns or 'abusive' not in df.columns:
        raise ValueError("The DataFrame should have 'user' and 'abusive' columns.")

    abusive_user_counts = df[df['abusive']]['user'].value_counts().head(num_users)

    # Create a Seaborn bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=abusive_user_counts.values, y=abusive_user_counts.index, palette="Reds")

    plt.title(f"Top {num_users} Most Abusive Users")
    plt.xlabel("Abuse Count")
    plt.ylabel("User")
    plt.grid(axis="x")

    # Invert y-axis for better readability (top user at the top)
    plt.gca().invert_yaxis()

    # Save the chart for Streamlit
    fig = plt.gcf()

    return fig
