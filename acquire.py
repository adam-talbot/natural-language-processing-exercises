######## WEB SCRAPING EXERCISES ########

import requests
from bs4 import BeautifulSoup
import pandas as pd

##### CODEUP DATA SCIENCE BLOG POSTS #####

def get_blog(url, user_agent):
    '''
    Takes in a url and user_agent and grab the title and content from the blog post found at that url (for codeup.com/data-science)
    '''
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html)
    title = soup.select('.et_pb_title_container')[0].h1.text
    paragraphs = soup.select('.et_pb_post_content_0_tb_body')[0].find_all('p')
    all_text = ''
    for paragraph in paragraphs:
        all_text += (paragraph.text)
    return({
        'title' : title,
        'content' : all_text
    })

def get_blog_articles(url_list, user_agent):
    '''
    Takes in a list of blog urls from codeup.com/data-science and a user agent string and creates a dataframe with title and content from each blog
    '''
    return pd.DataFrame([get_blog(url, user_agent) for url in url_list])

##### INSHORTS ARTICLE SUMMARIES #####

def parse_card(article, topic):
    '''
    Takes in article card html and pulls out title and content and returns a dictionary for that article with title, content, and topic
    '''
    title = article.select('.news-card-title')[0].span.text
    content = article.select('.news-card-content')[0].div.text
    topic = topic
    return {
        'title' : title,
        'content' : content,
        'topic' : topic
    }

def get_article_info(base_url, topic, user_agent):
    '''
    Takes in base url, topic, and user agent and returns a df with columns of title, content, and topic for each article
    '''
    url = base_url + topic
    headers = {'User-Agent': user_agent}
    response = requests.get(url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html)
    articles = soup.select('.news-card.z-depth-1')
    return pd.DataFrame([parse_card(article, topic) for article in articles])

def get_articles(topic_list, base_url, user_agent):
    '''
    Takes list of topics, base url, and user agent id string and return df with columns of title, content, and topic for all article summaries on all topic pages provided
    '''
    final_df = pd.DataFrame()
    for topic in topic_list:
        df = get_article_info(base_url = base_url, topic = topic, user_agent = user_agent)
        final_df = pd.concat([final_df, df], ignore_index=True)
    return final_df