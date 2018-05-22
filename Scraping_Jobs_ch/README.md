# Scraping Jobs.ch
Series of scripts that scrapes the website Jobs.ch. It sends an email summarising 
the new jobs added to jobs.ch since the last time the script was run (20 jobs max in the email). 


Running the script weekly can be automatised with the automator 
(http://naelshiab.com/tutorial-how-to-automatically-run-your-scripts-on-your-computer/).

Running only some parts (e.g. if you don't want to receive an email) can be done by skipping the 'general function' file and using the other files separately. 

#### Main components
* The 'general function' file is the only one that needs to run.
It imports all the other scripts and runs them in order.

* The 'Getting the links' file does the first part of the scraping. 
It goes to all the results pages for 'Data science' and saves basic information
about each job (title, place, company, and link). It saves the results to a csv file called links_current_jobs.csv

* The 'Comparing with current jobs' file compares the list of jobs scrapped (links_current_jobs.csv) with a list of full job descriptions (current_jobs.csv) created  the previous time the script ran (this step is skipped if no previous file is found).
It deletes, from the full job descriptions, the jobs that are not on the website anymore, and creates, as a csv, a list of jobs that were not on the platform the last time the script ran (new_jobs.csv). 
If it is the first time the script runs, the file links_current_jobs.csv is fully copied to new_jobs.csv

* The 'Adding job description' file adds job descriptions for all jobs in new_jobs.csv, using the links found with 'Getting the links'. 
This takes a while if it is the first time the script runs, but it is relatively fast after. It then merges the files new_jobs.csv with current_jobs.csv

* The 'Creating email' file sends an email with the 20 most recent jobs and their descriptions. 
The descriptions are not well formatted, as there is no regularity in the html on the job details pages, making clean parsing difficult.

#### Using the scripts
A working email address where to receive the emails and the path where to save the csv's need to be added at the bottom of the general function file.

If you want to scrape for another category of jobs, you can change the starting url (url_start) at the top of Getting_the_links.py. 
You can clearly see in the url where the search terms go and change them to whatever terms you want (or check the url you get on jobs.ch when you do your search if you want to add the place as well).


#### Content of the csv's
* links_current_jobs.csv is a list of all the data science jobs currently on jobs.ch, with all the basic info except the job description
* current_jobs.csv is the same list, but with all the job descriptions
* new_jobs.csv is the list of all jobs that were new at the last import, with the full job description

These are kept as separate files to make it easier to load only the missing job descriptions (making the script much faster to run), to access new jobs, and to delete old jobs.


#### Note on the format
The script is written entirely as classes, when it could have as easily been written as functions. I just wanted to work on object oriented programming :) 
