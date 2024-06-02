import re
import pandas as pd
from textblob import TextBlob


# List of offensive words
offensive_words = ['randi', 'muh ma la mera', 'jhatu', 'gand mara']
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s[apmAPM]{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    sentiments = []  # Added for sentiment analysis
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            text = " ".join(entry[2:])
            messages.append(text)
            # Perform sentiment analysis
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            if sentiment > 0:
                sentiments.append('Positive')
            elif sentiment < 0:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        else:
            users.append('group_notification')
            messages.append(entry[0])
            sentiments.append('Neutral')  # Assuming group_notification has neutral sentiment

    df['user'] = users
    df['message'] = messages
    df['sentiment'] = sentiments  # Added sentiment column
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period



    deleted_messages = df['message'].apply(lambda x: True if 'This message was deleted' in x else False)
    df['deleted'] = deleted_messages

    abusive_users = []
    for message in df['message']:
        # Check for offensive words
        if any(word in message.lower() for word in offensive_words):
            abusive_users.append(True)
        else:
            abusive_users.append(False)

    df['abusive'] = abusive_users

    return df




