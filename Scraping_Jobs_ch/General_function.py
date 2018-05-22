#!/usr/bin/env python3

import Getting_the_links
import Comparing_with_current_jobs
import Adding_job_descriptions
import Creating_email

"""
Starting in the shell: 
cd (go to directory where this is saved)
nohup python A_General_function.py > my.log 2>&1 &
echo $! > save_pid.txt

stopping in the shell:
cd (go to directory where this is saved)
kill -9 `cat save_pid.txt`
rm save_pid.txt
"""

class Launching_process:

    def __init__(self, email, path):
        self.email = email
        self.path = path
        self.link = Getting_the_links.GettingLinks(self.path)
        try:
            self.new_jobs = Comparing_with_current_jobs.CreateCsvNewJobs(self.path)
        except:
            print('No new jobs this week')
        self.job_descriptions = Adding_job_descriptions.AddingJobDescriptions(self.path)
        self.email_to_send = Creating_email.SendingEmail(self.email, self.path)


# launching the functions that collect the import info
email = 'Your email here'
path = 'Path were to save the csv files'
start = Launching_process(email, path)

