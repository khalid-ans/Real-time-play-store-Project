#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os


# In[153]:


nltk.download('vader_lexicon')


# In[154]:


# Step 1: Load the Dataset
apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')


# In[155]:


# Step 2: Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)


# In[156]:


# Merge datasets on 'App' and handle non-matching apps
merge_df = pd.merge(apps_df, reviews_df, on='App', how='inner')


# In[157]:


# Step 3: Data Transformation
apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)


# In[158]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan


# In[159]:


apps_df['Size'] = apps_df['Size'].apply(convert_size)


# In[160]:


# Add log_installs and log_reviews columns
apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Reviews'] = np.log1p(apps_df['Reviews'])


# In[161]:


# Add Rating Group column
def rating_group(rating):
    if rating >= 4:
        return 'Top rated'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'

apps_df['Rating_Group'] = apps_df['Rating'].apply(rating_group)


# In[162]:


# Add Revenue column
apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']


# In[163]:


# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[164]:


# Extract year from 'Last Updated' and create 'Year' column
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Year'] = apps_df['Last Updated'].dt.year


# In[165]:


import plotly.express as px

# Define the path for your HTML files
html_files_path = "./"

# Make sure the directory exists
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)

# Initialize plot_containers
plot_containers = ""

# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    # Append the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')

# Define your plots
plot_width = 400
plot_height = 300
plot_bg_color = 'black'
text_color = 'white'
title_font = {'size': 16}
axis_font = {'size': 12}

# Category Analysis Plot
category_counts = apps_df['Category'].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width,
    height=plot_height
)
fig1.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig1.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig1, "category_analysis.html", "The top categories on the Play Store are dominated by tools, entertainment, and productivity apps. This suggests users are looking for apps that either provide utility or offer leisure activities.")

# Type Analysis Plot
type_counts = apps_df['Type'].value_counts()
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=plot_width,
    height=plot_height
)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig2, "type_analysis.html", "Most apps on the Play Store are free, indicating a strategy to attract users first and monetize through ads or in-app purchases.")

# Rating Distribution Plot
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=plot_width,
    height=plot_height
)
fig3.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig3, "rating_distribution.html", "Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users.")

sentiment_counts = reviews_df['Sentiment_Score'].value_counts()
fig4 = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=plot_width,
    height=plot_height
)
fig4.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig4.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig4, "sentiment_distribution.html", "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.")

# Installs by Category Plot
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5 = px.bar(
    x=installs_by_category.values,
    y=installs_by_category.index,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=plot_width,
    height=plot_height
)
fig5.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig5.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig5, "installs_by_category.html", "The categories with the most installs are social and communication apps, which reflects their broad appeal and daily usage.")

# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "updates_per_year.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")

# Revenue by Category Plot
revenue_by_category = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7 = px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width,
    height=plot_height
)
fig7.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig7.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig7, "revenue_by_category.html", "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential.")

# Genre Count Plot
genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=plot_width,
    height=plot_height
)
fig8.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig8.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig8, "genres_counts.html", "Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.")

# Impact of Last Update on Rating
fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=plot_width,
    height=plot_height
)
fig9.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig9, "update_on_rating.html", "The scatter plot shows a weak correlation between the last update date and ratings, suggesting that more frequent updates don't always result in better ratings.")

# Ratings for Paid vs Free Apps
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Ratings for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=plot_width,
    height=plot_height
)
fig10.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig10, "ratings_paid_free.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for.")

# Split plot_containers to handle the last plot properly
plot_containers_split = plot_containers.split('</div>')
if len(plot_containers_split) > 1:
    final_plot = plot_containers_split[-2] + '</div>'
else:
    final_plot = plot_containers  # Use plot_containers as default if splitting isn't sufficient

# HTML template for the dashboard
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Reviews Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
    </style>
    <script>
        function openPlot(filename) {{
        window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
    <div class="container">
        {plots}
    </div>
</body>
</html>
"""

# Use these containers to fill in your dashboard HTML
final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

# Save the final dashboard to an HTML file
dashboard_path = os.path.join(html_files_path, "dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

# Automatically open the generated HTML file in a web browser
webbrowser.open('file://' + os.path.realpath(dashboard_path))


# In[166]:


apps_df.head()


# In[167]:


pip install wordcloud


# In[168]:


merge_df['Sentiment Score']=merge_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[169]:


user_df=reviews_df


# In[170]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

five_star_Health = merge_df[merge_df['Category']=="HEALTH_AND_FITNESS"][merge_df['Sentiment Score'] >= 0.8]['Translated_Review']
text_Health = ' '.join(five_star_Health.astype(str))

stopwordsHF = set(STOPWORDS)
app_names = set(user_df['App'].unique())
stopwordsHF.update(app_names)

wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      stopwords=stopwordsHF,
                      min_font_size=10).generate(text_Health)


plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[171]:


merge_df['Category'].unique()


# In[172]:


apps_df.head()


# In[173]:


from sklearn.preprocessing import MinMaxScaler
data = apps_df 

filtered_data = data.copy()
filtered_data['Last Updated'] = pd.to_datetime(filtered_data['Last Updated'], errors='coerce')
filtered_data['Size'] = pd.to_numeric(filtered_data['Size'], errors='coerce')
filtered_data = filtered_data[filtered_data['Last Updated'].notna()]


filtered_data = filtered_data[(filtered_data['Rating'] >= 4.0) &
                              (filtered_data['Size'] >= 10) &
                              (filtered_data['Last Updated'].dt.month == 1)]


category_stats = filtered_data.groupby('Category').agg(
    Average_Rating=('Rating', 'mean'),
    Total_Reviews=('Reviews', 'sum'),
    Total_Installs=('Installs', 'sum')
).reset_index()

scaler = MinMaxScaler(feature_range=(1, 5))
category_stats['Scaled_Reviews'] = scaler.fit_transform(category_stats[['Total_Reviews']])
top_categories = category_stats.nlargest(10, 'Total_Installs')

fig11 = px.bar(
    top_categories,
    x='Category',
    y=['Average_Rating', 'Scaled_Reviews'],
    title='Comparison of Average Rating and Scaled Reviews for Top 10 Categories by Installs',
    labels={'value': 'Metrics', 'variable': 'Metrics', 'Category': 'App Category'},
    barmode='group',
    width=400,
    height=300
)

fig11.update_layout(
    title_font_size=24,
    xaxis_title='App Category',
    yaxis_title='Metrics',
    legend_title='Metrics',
    plot_bgcolor='black',
paper_bgcolor='black',
font_color='white',
title_font=dict(size=16),
xaxis=dict(title_font=dict(size=12)),
yaxis=dict(title_font=dict(size=12)),
margin=dict(l=10, r=10, t=30, b=10)
)

fig11.show()
save_plot_as_html(fig11,"Comparision of rating.html","Apps with higher ratings often showcase excellent performance, usability, and customer support, while apps with lower ratings might suffer from bugs, poor UI/UX, or frequent crashes, influencing user satisfaction and overall reception.")


# In[174]:


import random
data = apps_df
countries = [
    "United States", "India", "Brazil", "Russia", "Germany", "United Kingdom", "France", "China", "Australia", "Canada"
]
data['Country'] = [random.choice(countries) for _ in range(len(data))]
filtered_data = data.copy()
filtered_data = filtered_data[~filtered_data['Category'].str.startswith(('A', 'C', 'G', 'S'))]
filtered_data = filtered_data[filtered_data['Installs'] > 1000000]
top_categories = filtered_data.groupby('Category')['Installs'].sum().nlargest(5).index
filtered_data = filtered_data[filtered_data['Category'].isin(top_categories)]


fig12 = px.choropleth(
    filtered_data,
    locations="Country",
    locationmode="country names",
    color="Installs",
    hover_name="Category",
    title="Global Installs by Top 5 App Categories",
    color_continuous_scale="Plasma",
    width=400,
    height=300
)

fig12.update_layout(
title_font_size=20,
    geo=dict(
    showframe=False,
    showcoastlines=False,
    projection_type="equirectangular",
),
plot_bgcolor='black',
paper_bgcolor='black',
font_color='white',
title_font=dict(size=16),
xaxis=dict(title_font=dict(size=12)),
yaxis=dict(title_font=dict(size=12)),
margin=dict(l=10, r=10, t=30, b=10)
)


fig12.show()
save_plot_as_html(fig12, "Global Installs by Top 5 App Categories.html", "Global Installs by Top 5 App Categories, shows the country audience are intrested in which category apps mostly ")



# In[175]:


data = apps_df
filtered_data = data.copy()

filtered_data = filtered_data[(filtered_data['Content Rating'] == 'Teen') &
                              (filtered_data['App'].str.startswith('E')) &
                              (filtered_data['Installs'] > 10000)]
filtered_data['Last Updated'] = pd.to_datetime(filtered_data['Last Updated'], errors='coerce')
filtered_data['Month'] = filtered_data['Last Updated'].dt.to_period('M').astype(str)
installs_trend = filtered_data.groupby(['Month', 'Category'])['Installs'].sum().reset_index()
installs_trend['MoM Change'] = installs_trend.groupby('Category')['Installs'].pct_change() * 100

fig13 = px.line(
    installs_trend,
    x='Month',
    y='Installs',
    color='Category',
    title='Trend of Total Installs Over Time by Category',
    labels={'Month': 'Month', 'Installs': 'Total Installs'},
     width=400,
     height=300
)

for category in installs_trend['Category'].unique():
    category_data = installs_trend[installs_trend['Category'] == category]
    fig13.add_scatter(
        x=category_data['Month'],
        y=category_data['Installs'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        name=f"{category} (Shaded)",
        showlegend=False
    )

for category in installs_trend['Category'].unique():
    category_data = installs_trend[installs_trend['Category'] == category]
    significant_growth = category_data[category_data['MoM Change'] > 20]

    for _, row in significant_growth.iterrows():
        fig13.add_vrect(
            x0=row['Month'],
            x1=row['Month'],
            fillcolor='lightgreen',
            opacity=0.5,
            layer='below',
            line_width=0,
            annotation_text=f">20% Growth ({category})",
            annotation_position="top left",

        )

fig13.update_layout(
    title_font_size=24,
    xaxis_title='Month',
    yaxis_title='Installs',
    plot_bgcolor='black',
paper_bgcolor='black',
font_color='white',
title_font=dict(size=16),
xaxis=dict(title_font=dict(size=12)),
yaxis=dict(title_font=dict(size=12)),
margin=dict(l=10, r=10, t=30, b=10)

)
fig13.show()
save_plot_as_html(fig13, "Installs Trend by category.html", "Line movement shows the growth of an category with the count of installs during respective years")



# In[176]:


data =apps_df  
filtered_data = data.copy()

filtered_data = filtered_data[(filtered_data['Rating'] > 3.5) &
                                  (filtered_data['Category'] == 'GAME') &
                                  (filtered_data['Installs'] > 50000)]
fig14 = px.scatter(
    filtered_data,
    x='Size',
    y='Rating',
    size='Installs',
    color='Category',
    hover_name='App',
    title='Relationship Between App Size and Rating for Games',
    labels={'Size': 'App Size (MB)', 'Rating': 'Average Rating'},
    width=400,
    height=300
)

fig14.update_layout(
title_font_size=24,
xaxis_title='App Size (MB)',
yaxis_title='Average Rating',
plot_bgcolor='black',
paper_bgcolor='black',
font_color='white',
title_font=dict(size=16),
xaxis=dict(title_font=dict(size=12)),
yaxis=dict(title_font=dict(size=12)),
margin=dict(l=10, r=10, t=30, b=10)
)

fig14.show()
save_plot_as_html(fig14, "Relation between app size and rating.html", "larger app sizes may indicate more features or better graphics, they can also lead to lower ratings due to longer download times or higher storage requirements, highlighting the importance of optimization.")



# In[177]:


pip install plotly


# In[ ]:





# In[178]:


insights = [
    "The top categories on the Play Store are dominated by tools, entertainment, and productivity apps. This suggests users are looking for apps that either provide utility or offer leisure activities.",
    "Most apps on the Play Store are free, indicating a strategy to attract users first and monetize through ads or in-app purchases.",
    "Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users.",
    "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.",
    "The categories with the most installs are social and communication apps, which reflects their broad appeal and daily usage.",
    "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.",
    "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential.",
    "Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.",
    "The scatter plot shows a weak correlation between the last update date and ratings, suggesting that more frequent updates don't always result in better ratings.",
    "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for."
]
initial_plots_html = ""
for i in range(1, 11):
        # Loop through figures 1 to 10
       fig_var_name = f"fig{i}" 
       fig = globals()[fig_var_name]  # Get the figure object

       # Generate filename for this plot's HTML file
       filename = f"plot_{i}.html"
       
       initial_plots_html += f"""
       <a href="#" onclick="openPlot('{filename}'); return false;">  
           <div class="plot-container" id="fig{i}">
               <h2>Figure {i}: Description</h2> 
               {fig.to_html(full_html=False, include_plotlyjs='cdn')}
               <div class="insights">{insights[i - 1]}</div>
           </div>
       </a>
       """
       # Save the plot to an individual HTML file:
       fig.write_html(filename)

plot_containers = """
<div class="plot-container" onclick="openPlot('Comparision of rating.html')" id="fig11" style="display: none;">
    <h2>Figure 11: (Visible between 3 PM and 6 PM IST)</h2>
    <div id="content11">
        <p>Apps with higher ratings often showcase excellent performance, usability, and customer support, while apps with lower ratings might suffer from bugs, poor UI/UX, or frequent crashes, influencing user satisfaction and overall reception. </p>
        {fig11_html}
    </div>
</div>
<div class="plot-container" onclick="openPlot('Global Installs by Top 5 App Categories.html')" id="fig12" style="display: none;">
    <h2>Figure 12: (Visible between 6 PM and 9 PM IST)</h2>
    <div id="content12">
        <p>Global Installs by Top 5 App Categories, shows the country audience are intrested in which category apps mostly </p>
        {fig12_html}
    </div>
</div>
<div class="plot-container" onClick="openPlot('Installs Trend by category.html')" id="fig13" style="display: none;">
    <h2>Figure 13: (Visible between 6 PM and 10 PM IST)</h2>
    <div id="content13">
        <p>Line movement shows the growth of an category with the count of installs during respective years</p>
        {fig13_html}
    </div>
</div>
<div class="plot-container" onclick="openPlot('Relation between app size and rating.html')"  id="fig14" style="display: none;">
    <h2>Figure 14:(Visible between 8 PM and midnight IST) </h2>
    <div class="insights"></div>
    <div id="content14">
        <p>larger app sizes may indicate more features or better graphics, they can also lead to lower ratings due to longer download times or higher storage requirements, highlighting the importance of optimization.</p>
        {fig14_html}
    </div>
</div>
"""

# HTML template for the dashboard (updated)
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Reviews Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: 400px; 
            height: 500px; 
            overflow: hidden;
            position: relative;
            background-color: #222;
            display: none; /* Initially hide all plot containers */
        }}
        .plot-container h2 {{
            text-align: center;
        }}
        .plot-container p {{
            text-align: center;
            font-size: 16px;
        }}
        /* ... other styles ... */

.plot-container {{
    /* ... other styles ... */
    display: block; /* or display: flex; if you want flex layout */
}}

.insights {{
    display: block;  /* or display: inline-block; Make the insights visible by default */
    position: absolute;
    right: 10px;
    top: 10px;
    background-color: rgba(0, 0, 0, 0.7);
    padding: 5px;
    border-radius: 5px;
    color: #fff;
}}

/* Optional: If you want to hide/show on hover */
/*.plot-container .insights {{ 
    display: block; 
}}*/

.plot-container:hover .insights {{ 
    display: block; 
}}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
        function checkTimeAndDisplay() {{
            const now = new Date();
            const currentHour = now.getHours();
            const currentMinute = now.getMinutes();

            const fig11Display = (currentHour === 15 && currentMinute >= 0) || (currentHour === 16) || (currentHour === 17 && currentMinute < 60);
            const fig12Display = (currentHour === 18) || (currentHour === 19) || (currentHour === 20 && currentMinute < 60);
            const fig13Display = (currentHour === 18) || (currentHour === 19) || (currentHour === 20) || (currentHour === 21 && currentMinute < 60);
            const fig14Display = (currentHour === 20) || (currentHour === 21) || (currentHour === 22) || (currentHour === 23) || (currentHour === 0 && currentMinute < 60);
           
            
            document.getElementById('fig11').style.display = fig11Display ? 'block' : 'none';
            document.getElementById('fig12').style.display = fig12Display ? 'block' : 'none';
            document.getElementById('fig13').style.display = fig13Display ? 'block' : 'none';
            document.getElementById('fig14').style.display = fig14Display ? 'block' : 'none';

            // Show the notice only if none of the plots are scheduled to be visible at the current time
            if (!fig11Display && !fig12Display && !fig13Display && !fig14Display) {{
                document.getElementById('notice').style.display = 'block';
            }} else {{
                document.getElementById('notice').style.display = 'none';
            }}
        }}
    </script>
</head>
<body onload="checkTimeAndDisplay()">
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
    </div>
<div class="container">
         {initial_plots}  
         {plots}
     </div>
    <div id="notice" style="display:none; text-align:center; padding:20px;">
        <h2>The analytics are available only during the specified hours.</h2>
    </div>
</body>
</html>
"""
final_html = dashboard_html.format(
         initial_plots=initial_plots_html,  # Add this line
         plots=plot_containers.format(
             fig11_html=fig11.to_html(full_html=False, include_plotlyjs='cdn'),
             fig12_html=fig12.to_html(full_html=False, include_plotlyjs='cdn'),
        fig13_html=fig13.to_html(full_html=False, include_plotlyjs='cdn'),
        fig14_html=fig14.to_html(full_html=False, include_plotlyjs='cdn')
             
         )
     )
# Generating the final HTML with embedded plots
# final_html = dashboard_html.format(
#     plots=plot_containers.format(
#         fig11_html=fig11.to_html(full_html=False, include_plotlyjs='cdn'),
#         fig12_html=fig12.to_html(full_html=False, include_plotlyjs='cdn'),
#         fig13_html=fig13.to_html(full_html=False, include_plotlyjs='cdn'),
#         fig14_html=fig14.to_html(full_html=False, include_plotlyjs='cdn')
#     )
# )


html_files_path = "."  
dashboard_path = os.path.join(html_files_path, "dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)
webbrowser.open('file://' + os.path.realpath(dashboard_path)) 


# In[179]:


import os
import shutil

# Define directories
root_dir = "final-project"
visualizations_dir = os.path.join(root_dir, "visualizations")
assets_dir = os.path.join(root_dir, "assets")

# Create directories
os.makedirs(visualizations_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)

# Move generated HTML files into /visualizations
generated_files = ['plot_1.html', 'plot_2.html', 'plot_3.html', 'plot_4.html', 'plot_5.html', 'plot_6.html', 'plot_7.html', 'plot_8.html', 'plot_9.html', 'plot_10.html','Comparision of rating.html','Global Installs by Top 5 App Categories.html','Installs Trend by category.html','Relation between app size and rating.html']



for file in generated_files:
    if os.path.exists(file):
        shutil.move(file, visualizations_dir)

# Create the index.html file
dashboard_path = "dashboard.html"
if os.path.exists(dashboard_path):
    shutil.move(dashboard_path, os.path.join(root_dir, "index.html"))

# Create _redirects file
redirects_content = "/*    /index.html   200\n"
with open(os.path.join(root_dir, "_redirects"), "w") as f:
    f.write(redirects_content)

print(f"Directories and files have been set up in: {root_dir}")


# In[ ]:




