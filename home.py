import streamlit as st
import os
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from textblob import TextBlob
import helper
import preprocessor

# Function to check if user is logged in
def is_logged_in():
    return 'username' in st.session_state and st.session_state.username
def app():
    # Check if the user is logged in
    if 'username' not in st.session_state or not st.session_state.username:
        st.warning("Please log in to access the Whatsapp Chat Analyzer.")
        return


    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        # Calculate total words
        words = df['message'].apply(lambda x: len(str(x).split())).sum()

        # Calculate total number of messages
        num_messages = len(df)
        num_media_messages = len(df)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        # Allow users to select multiple users for comparison
        selected_users = st.multiselect("Select users for comparison", user_list, default=["Overall"])

        # Add the button to generate the PDF report
        if st.button("Generate PDF Report"):
            pdf = FPDF()  # Create a new PDF object

            for selected_user in selected_users:
                pdf.add_page()  # Add a page for each user
                pdf.set_font("Arial", size=22)  # Set the font

                # Add the title and username
                pdf.cell(200, 10, txt="WhatsApp Chat Analysis Report", ln=1, align="C")
                pdf.cell(200, 10, txt=f"Analysis for {selected_user}", ln=1, align="C")

                # Top Statistics
                pdf.cell(200, 10, txt="Top Statistics:", ln=1)
                pdf.cell(200, 10, txt=f"Total Words: {words}", ln=1)
                num_messages = len(df)
                pdf.cell(200, 10, txt=f"Total Messages: {num_messages}", ln=1)
                num_media_messages = len(df)

                # Top 10 Most Talkative Users
                pdf.cell(200, 10, txt="Top 10 Most Talkative Users:", ln=1)

                # Fetch and display the Top 10 Most Talkative Users
                most_talkative_users_chart = helper.most_talkative_users_bar_chart(df)
                temp_file_talkative = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                most_talkative_users_chart.savefig(temp_file_talkative.name, format='png')
                most_talkative_users_chart.clear()  # Clear the figure

                # Create a BytesIO buffer
                buffer = BytesIO()

                # Save the figure to the buffer
                # Save the figure to a temporary file
                plt.figure(figsize=(12, 6))
                plt.xticks(rotation='vertical')
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(temp_file.name, format='png')
                plt.close()  # Close the figure

                # Monthly Timeline
                timeline = helper.monthly_timeline(selected_user, df)
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=timeline, x='time', y='message', marker='o', color='purple')
                plt.title("Monthly Chat Activity Over Time", fontsize=16,
                          color='blue')  # Customize title size and color
                plt.xticks(rotation='vertical')

                # Save the Monthly Timeline figure to a temporary file
                temp_file_timeline = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(temp_file_timeline.name, format='png')
                plt.close()  # Close the figure

                # Most Active Date

                most_active_day = helper.most_active_date(selected_user, df)
                plt.figure(figsize=(10, 6))
                sns.countplot(x='only_date', data=df[df['only_date'] == most_active_day], palette="viridis")
                plt.title("Messages Count on Most Active Date")
                plt.xlabel("Date")
                plt.ylabel("Number of Messages")
                plt.xticks(rotation='vertical')

                # Save the Most Active Date figure to a temporary file
                temp_file_active_date = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(temp_file_active_date.name, format='png')
                plt.close()  # Close the figure

                # WordCloud
                df_wc = helper.create_wordcloud(selected_user, df)
                plt.figure(figsize=(10, 8))
                plt.imshow(df_wc)
                plt.title("Customized Word Cloud Title", fontsize=16,color='yellow')  # Customize title text, size, and color
                plt.axis("off")

                # Save the WordCloud figure to a temporary file
                temp_file_wordcloud = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(temp_file_wordcloud.name, format='png')
                plt.close()  # Close the figure

                # Most Common Words
                most_common_df = helper.most_common_words(selected_user, df)
                plt.figure(figsize=(12, 6))
                plt.barh(most_common_df[0], most_common_df[1])
                plt.xticks(rotation='vertical')

                # Save the Most Common Words figure to a temporary file
                temp_file_most_common = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(temp_file_most_common.name, format='png')
                plt.close()  # Close the figure

               

                # Most and Least Busy Day
                most_busy_day, least_busy_day = helper.most_and_least_busy_day(selected_user, df)
                pdf.cell(200, 10, txt=f"Most Busy Day: {most_busy_day}", ln=1)
                pdf.cell(200, 10, txt=f"Least Busy Day: {least_busy_day}", ln=1)

                    # Add the Most Busy Day chart to the PDF
                busy_day_counts = df['day_name'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size
                ax.bar(busy_day_counts.index, busy_day_counts.values, color='blue')
                ax.set_xlabel('Day of the Week')
                ax.set_ylabel('Message Count')
                ax.set_title('Most Busy Day')
                plt.xticks(rotation=45, ha='center')  # Adjust rotation and alignment

                    # Save the Most Busy Day figure to a temporary file
                temp_file_busy_day = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(temp_file_busy_day.name, format='png')
                plt.close()  # Close the figure

                    # Color-coded bar chart for Least Busy Day
                least_busy_day_counts = df['day_name'].value_counts()
                fig_least_busy, ax_least_busy = plt.subplots(figsize=(10, 7))  # Adjust the figure size
                ax_least_busy.bar(least_busy_day_counts.index, least_busy_day_counts.values, color='red')
                ax_least_busy.set_xlabel('Day of the Week')
                ax_least_busy.set_ylabel('Message Count')
                ax_least_busy.set_title('Least Busy Day')
                ax_least_busy.tick_params(axis='x', rotation=45)  # Adjust rotation

                    # Save the Least Busy Day figure to a temporary file
                temp_file_least_busy_day = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig_least_busy.savefig(temp_file_least_busy_day.name, format='png')
                plt.close(fig_least_busy)  # Close the figure

                    # Most and Least Busy Month
                most_busy_month, least_busy_month = helper.most_and_least_busy_month(selected_user, df)
                pdf.cell(200, 10, txt=f"Most Busy Month: {most_busy_month}", ln=1)
                pdf.cell(200, 10, txt=f"Least Busy Month: {least_busy_month}", ln=1)

                    # Color-coded bar chart for Most Busy Month
                busy_month_counts = df['month'].value_counts()
                fig_most_busy_month, ax_most_busy_month = plt.subplots(figsize=(10, 7))  # Adjust the figure size
                ax_most_busy_month.bar(busy_month_counts.index, busy_month_counts.values, color='green')
                ax_most_busy_month.set_xlabel('Month')
                ax_most_busy_month.set_ylabel('Message Count')
                ax_most_busy_month.set_title('Most Busy Month')
                ax_most_busy_month.tick_params(axis='x', rotation=45)  # Adjust rotation

                    # Save the Most Busy Month figure to a temporary file
                temp_file_most_busy_month = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig_most_busy_month.savefig(temp_file_most_busy_month.name, format='png')
                plt.close(fig_most_busy_month)  # Close the figure

                    # Least Busy Month
                least_busy_month_counts = df['month'].value_counts()
                fig_least_busy_month, ax_least_busy_month = plt.subplots(figsize=(10, 7))  # Adjust the figure size
                ax_least_busy_month.bar(least_busy_month_counts.index, least_busy_month_counts.values,color='orange')
                ax_least_busy_month.set_xlabel('Month')
                ax_least_busy_month.set_ylabel('Message Count')
                ax_least_busy_month.set_title('Least Busy Month')
                ax_least_busy_month.tick_params(axis='x', rotation=45)  # Adjust rotation

                    # Save the Least Busy Month figure to a temporary file
                temp_file_least_busy_month = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig_least_busy_month.savefig(temp_file_least_busy_month.name, format='png')
                plt.close(fig_least_busy_month)  # Close the figure

                # Open the temporary file in binary mode and read its content
                with open(temp_file_talkative.name, 'rb') as img_file_talkative:
                    buffer_talkative = BytesIO(img_file_talkative.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_timeline.name, 'rb') as img_file_timeline:
                    buffer_timeline = BytesIO(img_file_timeline.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_active_date.name, 'rb') as img_file_active_date:
                    buffer_active_date = BytesIO(img_file_active_date.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_wordcloud.name, 'rb') as img_file_wordcloud:
                    buffer_wordcloud = BytesIO(img_file_wordcloud.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_most_common.name, 'rb') as img_file_most_common:
                    buffer_most_common = BytesIO(img_file_most_common.read())

                

                # Open the temporary file in binary mode and read its content
                with open(temp_file_busy_day.name, 'rb') as img_file_busy_day:
                    buffer_busy_day = BytesIO(img_file_busy_day.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_least_busy_day.name, 'rb') as img_file_least_busy_day:
                    buffer_least_busy_day = BytesIO(img_file_least_busy_day.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_most_busy_month.name, 'rb') as img_file_most_busy_month:
                    buffer_most_busy_month = BytesIO(img_file_most_busy_month.read())

                # Open the temporary file in binary mode and read its content
                with open(temp_file_least_busy_month.name, 'rb') as img_file_least_busy_month:
                    buffer_least_busy_month = BytesIO(img_file_least_busy_month.read())

                # Save BytesIO content to a temporary file
                temp_file_talkative_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_talkative_pdf.write(buffer_talkative.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_timeline_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_timeline_pdf.write(buffer_timeline.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_active_date_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_active_date_pdf.write(buffer_active_date.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_wordcloud_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_wordcloud_pdf.write(buffer_wordcloud.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_most_common_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_most_common_pdf.write(buffer_most_common.getvalue())

                # Save BytesIO content to a temporary file
                

                # Save BytesIO content to a temporary file
                temp_file_busy_day_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_busy_day_pdf.write(buffer_busy_day.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_least_busy_day_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_least_busy_day_pdf.write(buffer_least_busy_day.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_most_busy_month_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_most_busy_month_pdf.write(buffer_most_busy_month.getvalue())

                # Save BytesIO content to a temporary file
                temp_file_least_busy_month_pdf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_file_least_busy_month_pdf.write(buffer_least_busy_month.getvalue())

                # Add the image file to the PDF
                pdf.image(temp_file_talkative_pdf.name, w=150, h=100)

                # Add the Monthly Timeline image file to the PDF
                pdf.image(temp_file_timeline_pdf.name, w=150, h=100)

                # Add the Most Active Date image file to the PDF
                pdf.image(temp_file_active_date_pdf.name, w=150, h=100)

                # Add the WordCloud image file to the PDF
                pdf.image(temp_file_wordcloud_pdf.name, w=150, h=100)

                # Add the Most Common Words image file to the PDF
                pdf.image(temp_file_most_common_pdf.name, w=150, h=100)

                pdf.image(temp_file_busy_day.name, w=150, h=100)

                # Add the Least Busy Day image file to the PDF
                pdf.image(temp_file_least_busy_day_pdf.name, w=150, h=100)

                pdf.image(temp_file_most_busy_month_pdf.name, w=150, h=100)

                pdf.image(temp_file_least_busy_month_pdf.name, w=150, h=100)

                # Close the temporary files after use
                temp_file_talkative_pdf.close()
                temp_file_talkative.close()
                temp_file_timeline_pdf.close()
                temp_file_timeline.close()
                temp_file_active_date_pdf.close()
                temp_file_active_date.close()
                temp_file_wordcloud_pdf.close()
                temp_file_wordcloud.close()
                temp_file_most_common_pdf.close()
                temp_file_most_common.close()
                temp_file_busy_day.close()
                temp_file_least_busy_day_pdf.close()
                temp_file_least_busy_day.close()
                temp_file_most_busy_month_pdf.close()
                temp_file_most_busy_month.close()
                temp_file_least_busy_month_pdf.close()
                temp_file_least_busy_month.close()

                # After generating the PDF, delete the temporary files
                os.unlink(temp_file_talkative.name)
                os.unlink(temp_file_talkative_pdf.name)
                os.unlink(temp_file_timeline.name)
                os.unlink(temp_file_active_date.name)
                os.unlink(temp_file_most_common.name)
                os.unlink(temp_file_wordcloud.name)
                os.unlink(temp_file_busy_day.name)
                os.unlink(temp_file_least_busy_day.name)
                os.unlink(temp_file_least_busy_day_pdf.name)
                os.unlink(temp_file_most_busy_month.name)
                os.unlink(temp_file_most_busy_month_pdf.name)
                os.unlink(temp_file_least_busy_month.name)
                os.unlink(temp_file_least_busy_month_pdf.name)

                # Generate the PDF and prompt for download
            pdf.output("analysis.pdf")
            st.success("PDF report generated successfully!")
            st.balloons()  # Add a visual notification

        if st.button("Show Analysis"):
            for selected_user in selected_users:
                # Stats Area
                num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
                st.title(f"Top Statistics for {selected_user}")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)

            # Number of messages deleted after being sent
            num_deleted_messages = helper.num_deleted_messages(df)
            st.title(f"Number of Messages Deleted After Being Sent: {num_deleted_messages}")

            st.title("Top 10 Most Talkative Users")
            most_talkative_chart = helper.most_talkative_users_bar_chart(df)
            st.pyplot(most_talkative_chart)

            # Fetch the hour-wise message distribution
            hourly_distribution = helper.hourly_timeline(selected_user, df)

            # Display the hour-wise message distribution with a bar chart
            st.title("Hourly Message Distribution")
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Hour', y='Message Count', data=hourly_distribution, palette="Purples")
            plt.title("Hourly Message Distribution")
            plt.xlabel("Hour of the Day")
            plt.ylabel("Message Count")
            plt.xticks(rotation='vertical')
            st.pyplot(plt.gcf())

            def topic_modeling(selected_user, df):
                if selected_user != 'Overall':
                    df = df[df['user'] == selected_user]

            # Fetch the top 5 reviews with the highest positive sentiment
            top_positive_reviews_df = helper.top_positive_reviews(selected_user, df)

            # Display the top positive reviews with a cool-looking chart
            st.title("Top 5 Reviews with Highest Positive Sentiment")
            if not top_positive_reviews_df.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='message_sentiment', y='message', data=top_positive_reviews_df, palette="viridis")
                plt.title("Top 5 Reviews with Highest Positive Sentiment")
                plt.xlabel("Positive Sentiment Polarity Score")
                plt.ylabel("Review")
                plt.xticks(rotation='horizontal')
                st.pyplot(plt.gcf())
            else:
                st.info("No reviews with positive sentiment found.")

            # Monthly Timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=timeline, x='time', y='message', marker='o', color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(plt.gcf())

            # Display the most abusive users
            st.title("Most Abusive Users")
            abusive_users_chart = helper.most_abusive_users_bar_chart(df)
            st.pyplot(abusive_users_chart)

            def most_active_date(selected_user, df):

                if selected_user != 'Overall':
                    df = df[df['user'] == selected_user]

                active_date = df['only_date'].value_counts().idxmax()

                return active_date

            # In your Streamlit app after fetching the stats and other analyses
            # Fetch the most active date
            most_active_day = helper.most_active_date(selected_user, df)

            # Display the most active date with a cool-looking chart
            st.title("Most Active Date")
            plt.figure(figsize=(10, 6))
            sns.countplot(x='only_date', data=df[df['only_date'] == most_active_day], palette="viridis")
            plt.title("Messages Count on Most Active Date")
            plt.xlabel("Date")
            plt.ylabel("Number of Messages")
            plt.xticks(rotation='vertical')
            st.pyplot(plt.gcf())

            # Sentiment Distribution by User
            st.title("Sentiment Distribution by User")

            # Call the helper function to get the chart
            sentiment_distribution_chart = helper.sentiment_distribution_by_user_chart(df)

            # Display the chart in Streamlit app
            st.pyplot(sentiment_distribution_chart)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Weekday Chatting Circle
            st.title("Weekday Chatting Circle")
            weekday_circle_chart = helper.weekday_chatting_circle(selected_user, df)
            st.pyplot(weekday_circle_chart)

            # activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # finding the busiest users in the group(Group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

            # WordCloud
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # Most and Least Busy Day
            most_busy_day, least_busy_day = helper.most_and_least_busy_day(selected_user, df)
            st.title(f"Most Busy Day: {most_busy_day}")
            st.title(f"Least Busy Day: {least_busy_day}")

            # Color-coded bar chart for Most Busy Day
            busy_day_counts = df['day_name'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(busy_day_counts.index, busy_day_counts.values, color='blue')
            ax.set_xlabel('Day of the Week')
            ax.set_ylabel('Message Count')
            ax.set_title('Most Busy Day')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Color-coded bar chart for Least Busy Day
            least_busy_day_counts = df['day_name'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(least_busy_day_counts.index, least_busy_day_counts.values, color='red')
            ax.set_xlabel('Day of the Week')
            ax.set_ylabel('Message Count')
            ax.set_title('Least Busy Day')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Most and Least Busy Month
            most_busy_month, least_busy_month = helper.most_and_least_busy_month(selected_user, df)
            st.title(f"Most Busy Month: {most_busy_month}")
            st.title(f"Least Busy Month: {least_busy_month}")

            # Color-coded bar chart for Most Busy Month
            busy_month_counts = df['month'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(busy_month_counts.index, busy_month_counts.values, color='green')
            ax.set_xlabel('Month')
            ax.set_ylabel('Message Count')
            ax.set_title('Most Busy Month')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Color-coded bar chart for Least Busy Month
            least_busy_month_counts = df['month'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(least_busy_month_counts.index, least_busy_month_counts.values, color='purple')
            ax.set_xlabel('Month')
            ax.set_ylabel('Message Count')
            ax.set_title('Least Busy Month')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
            # most common words
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots()

            ax.barh(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')

            st.title('Most common words')
            st.pyplot(fig)


            # Sentiment and Engagement
            st.title("Sentiment and Engagement Analysis")

            # Calculate message length and response time
            df['message_length'] = df['message'].apply(len)
            df['response_time'] = df['date'].diff().dt.seconds / 60  # Response time in minutes

            # Group data by sentiment and calculate average message length and response time
            sentiment_engagement = df.groupby('sentiment').agg({
                'message_length': 'mean',
                'response_time': 'mean'
            }).round(2)

            # Display the calculated averages
            st.dataframe(sentiment_engagement)

            # Plot bar chart for average message length and response time by sentiment (horizontal)
            fig, ax = plt.subplots(figsize=(10, 6))
            sentiment_engagement.plot(kind='barh', ax=ax, color=['blue', 'green'], alpha=0.7)
            ax.set_title("Sentiment and Engagement Analysis")
            ax.set_xlabel("Average")
            ax.set_ylabel("Sentiment")
            ax.legend(["Message Length", "Response Time"], title='Engagement', bbox_to_anchor=(1, 1), loc='upper left')

            # Display the plot using Streamlit
            st.pyplot(fig)

            # Sentiment-Based Conversation Summaries
            st.title("Sentiment-Based Conversation Summaries")

            # Perform sentiment analysis on messages
            df['message_sentiment'] = df['message'].apply(lambda x: TextBlob(x).sentiment.polarity)

            # Categorize sentiment into Positive, Negative, and Neutral
            df['sentiment_category'] = df['message_sentiment'].apply(
                lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

            # Display sentiment distribution summary
            sentiment_summary = df['sentiment_category'].value_counts(normalize=True) * 100
            st.bar_chart(sentiment_summary)

            # Display sentiment-based conversation summaries
            sentiment_groups = df.groupby('sentiment_category')

            for sentiment, group_df in sentiment_groups:
                st.title(f"{sentiment} Sentiment Conversation Summary")
                st.write(f"Total Messages: {len(group_df)}")
                st.write(f"Avg. Message Length: {group_df['message'].apply(len).mean():.2f} characters")

                # Add more relevant summary statistics or information based on your requirements

                # Visualize message length distribution using Matplotlib (horizontal bar chart)
                st.title(f"{sentiment} Sentiment Message Length Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(range(len(group_df)), group_df['message'].apply(len), color='blue', alpha=0.7)
                ax.set_ylabel('Message Index')
                ax.set_xlabel('Message Length (characters)')
                ax.set_title(f"{sentiment} Sentiment Message Length Distribution")
                st.pyplot(fig)
    # Run the Streamlit app
    if __name__ == "__main__":
        app()