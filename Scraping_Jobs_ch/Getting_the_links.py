import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
from time import strftime


class GettingLinks:
    url_start = 'https://www.jobs.ch/en/vacancies/?page=1&term=data+science'
    url_part1 = 'https://www.jobs.ch/en/vacancies/?page='
    url_part2 = '&term=data+science'
    url_job_ch = 'https://www.jobs.ch'

    def __init__(self, path):
        """
        Gets all the job links and saves then to a data frame when the object is initialised
        methods:
        .number_of_pages_results = how many pages of job advert
        .current_jobs_links = returns a data frame of all current jobs links
        :param path:
        """
        self.path = path
        self.number_of_pages_results = self.getting_max_number_pages()
        jobs_list = self.getting_list_jobs()
        self.current_jobs_links = self.results_to_df(jobs_list)
        self.save_job_links()

    def save_job_links(self):
        """
        Saves the data
        :return:
        """
        os.chdir(self.path)
        self.current_jobs_links.to_csv('links_current_jobs.csv', sep=",", mode='w', index=False)

    def getting_max_number_pages(self):
        """
        Getting the max number of pages of results from the first page
        (otherwise risk of having a response from the page but the results are not for the query anymore)
        :param url_start: first page of results
        :return: the max number of pages as displayed at the bottom right of the page AS INT
        """
        page = requests.get(self.url_start)
        soup = BeautifulSoup(page.content, 'html.parser')
        max_pages = soup.find(class_='page hidden-xs last').contents[0].contents[0]
        max_pages = int(max_pages)
        return max_pages

    def getting_list_jobs(self):
        """
        Getting the lists of jobs based on the max number of pages indicated on the first page
        :param max_pages:
        :return: A list where each element is the list, for each job, including: date, title, partial url
        """
        jobs_list = []
        n = 1
        while n <= self.number_of_pages_results:
            url = self.url_part1 + str(n) + self.url_part2
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            job_titles_links = self.getting_titles_links(soup)
            job_company_place = self.getting_company_place(soup)
            jobs = list(zip(job_titles_links, job_company_place))
            list(map(lambda x: x[0].extend(x[1]), jobs))
            jobs = [job[0] for job in jobs]
            jobs_list.extend(jobs)
            n += 1
        return jobs_list

    @staticmethod
    def getting_titles_links(soup):
        """
        Gets the titles and links for the jobs
        CALLED IN: getting_list_jobs()
        :param soup: parsed page
        :return: A list of lists containing, for each job: the date, the title, the partial link
        """
        job_titles_links = []
        jobs_on_page = soup.find_all('h2')
        for element in jobs_on_page:
            try:
                job_title = element.find("a").contents[0].get_text()
                link = element.find("a").get('href')
                job_titles_links.append([strftime("%d.%m.%y"), job_title, link])
            except:
                pass
        return job_titles_links

    @staticmethod
    def getting_company_place(soup):
        """
        Gets the company and place for the jobs
        Some jobs don't have company, hence the except = 'NaN' there
        The get_text() gives both the company and the place
        (but have to use the hyperlink tag for the company, as it might not be in the other one)
        CALLED IN: getting_list_jobs()
        :param soup: parsed page
        :return: A list of lists containing, for each job: the place and the company
        """
        job_company_place = []
        jobs_on_page = soup.find_all('h3')
        for element in jobs_on_page:
            try:
                place = element.get_text()
                try:
                    place = place.split(" — ")
                    place = place[1]
                except:
                    place = place[0]
                try:
                    company = element.find("a").contents[0].get_text()
                except:
                    company = 'NaN'
            except:
                pass
            company_place = [company, place]
            job_company_place.append(company_place)
        return job_company_place

    def results_to_df(self, jobs_list):
        """
        Turns the results into a data frame and creates the full urls
        :param jobs_list: list of jobs (where each element is itself a list of infor about the job
        :return: data frame of jobs with date, title, and full link
        """
        current_jobs_links = pd.DataFrame(jobs_list)
        current_jobs_links.columns = ['Date', 'Title', 'Link', 'Company', 'Place']
        current_jobs_links['Link'] = current_jobs_links['Link'].map(self.create_link_job_description)
        return current_jobs_links

    def create_link_job_description(self, partial_link):
        """
        creates the full links for the job description
        CALLED IN: results_to_df
        :param partial_link: links scraped from job.ch
        :return: full job link
        """
        link = self.url_job_ch + partial_link
        return link
