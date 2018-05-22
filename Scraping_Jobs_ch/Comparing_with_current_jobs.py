import pandas as pd
import os


class CreateCsvNewJobs:

    def __init__(self, path):
        """
        Creates a csv of new jobs directly when the object is initialised
        methods:
        .new_jobs returns the data frame of all new jobs
        :param path:
        """
        self.path = path
        os.chdir(self.path)
        df_all_jobs = pd.read_csv('links_current_jobs.csv', sep=',')
        if os.path.exists('current_jobs.csv'):
            df_old_jobs = pd.read_csv('current_jobs.csv', sep=',')
            new_jobs_list = self.getting_only_new_jobs(df_all_jobs, df_old_jobs)
            self.new_jobs = self.results_to_df(new_jobs_list)
            self.new_jobs.to_csv('new_jobs.csv', sep=",", mode='w', index=False)
            for index, row in df_old_jobs.iterrows():
                if df_all_jobs['Link'].str.contains(row['Link'][39:47]).any():
                    pass
                else:
                    df_old_jobs.drop(index, inplace=True)
            df_old_jobs.to_csv('current_jobs.csv', sep=",", mode='w', index=False)
            if len(self.new_jobs) == 0:
                exit()
        else:
            print('Running for the first time, will collect information on all available jobs')
            self.new_jobs = df_all_jobs
            self.new_jobs.to_csv('new_jobs.csv', sep=",", mode='w', index=False)

    @staticmethod
    def getting_only_new_jobs(df_all_jobs, df_old_jobs):
        """
        Getting a list of all the new jobs, that is the jobs in the list of current links on the website
        that were not there in the previous run of the programme
        :return: list of new jobs
        """
        new_jobs = []
        for index, row in df_all_jobs.iterrows():
            if df_old_jobs['Link'].str.contains(row['Link'][39:47]).any():
                pass
            else:
                new_job = []
                for column in row:
                    new_job.append(column)
                new_jobs.append(new_job)
        return new_jobs

    @staticmethod
    def results_to_df(new_jobs_list):
        """
        Turns the results into a data frame and creates the full urls
        :param new_jobs_list: list of jobs (where each element is itself a list of info about the job
        :return: data frame of jobs with date, title, and full link
        """
        new_jobs = pd.DataFrame(new_jobs_list)
        new_jobs.columns = ['Date', 'Title', 'Link', 'Company', 'Place']
        return new_jobs

