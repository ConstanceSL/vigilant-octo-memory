import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re


class AddingJobDescriptions:

    def __init__(self, path):
        """
        Gets all the job descriptions for the new jobs
        Saves them in new_jobs.csv (overwriting the csv with only the links for new jobs)
        Merges the new_jobs with the old one

        methods:
        .new_jobs_full_description = returns a data frame with all the new job descriptions

        :param path:
        """
        self.path = path
        os.chdir(self.path)
        df = pd.read_csv('new_jobs.csv', sep=',')
        self.new_jobs_full_description = self.get_all_descriptions(df)
        self.new_jobs_full_description.to_csv('new_jobs.csv', sep=",", mode='w', index=False)
        self.new_jobs_full_description.to_csv('current_jobs.csv', sep=",", mode='a', index=False)

    def get_all_descriptions(self, df):
        """
        Getting all the info about the jobs
        :param df: data frame with the links
        :return: the data frame with extra columns for job description
        """
        df['Job description'] = 'Job description'
        for index, row in df.iterrows():
            url = row['Link']
            page = requests.get(url)
            parsed_page = BeautifulSoup(page.content, 'html.parser')
            row['Job description'] = self.get_description(parsed_page)
        return df

    @staticmethod
    def get_description(parsed_page):
        """
        Getting the job description from the parsed page
        CALLED IN get_all_info()
        :param parsed_page: from get_all_info()
        :return: a list with all the text from the description
        """
        parsed_page_results = parsed_page.find_all('p')
        description = []
        for element in parsed_page_results:
            text = element.get_text()
            description.append(text)
        description = ''.join(description)
        description = re.sub('You are not logged in', '', description)
        return description
