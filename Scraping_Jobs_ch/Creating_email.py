import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import pandas as pd


class SendingEmail:
    email_address_sender = "datasciencejobsupdates@gmail.com"
    password = "datascience"

    def __init__(self, email, path):
        """
        Creates a smaller dataframe of jobs to send + turns the titles to html hyperlinks
        Creates the html text for the email
        sends the email

        methods:
        .adverts_to_send = returns a data frame of the jobs that will be sent
        .html_text = returns the html text that will be sent in the email
        :param email:
        :param path:
        """
        self.email = email
        self.path = path
        self.adverts_to_send = self.getting_jobs()
        self.html_text = self.html_formatting()
        self.sending_email()

    def getting_jobs(self):
        """
        Loads the csv of new jobs, turns the title into hyperlinks (using the "title to hyperlink" function)
        If more than 20 new jobs, keeps only the top 20 ones
        :return:
        """
        os.chdir(self.path)
        df_new = pd.read_csv('new_jobs.csv', sep=',')
        if len(df_new) > 20:
            adverts_to_send = df_new.head(20)

        else:
            adverts_to_send = df_new
        adverts_to_send = self.title_to_hyperlink(adverts_to_send)
        return adverts_to_send

    @staticmethod
    def title_to_hyperlink(adverts_to_send):
        """
        Turns the titles in the data frame to html hyperlinks
        CALLED IN getting_jobs
        :param df_new_small:
        :param df_new: data frame of new jobs
        :return:
        """
        for index, row in adverts_to_send.iterrows():
            row['Title'] = '<a href="' + row['Link'] + '" style="color:#07506E">' + row['Title'] + '</a>'
        return adverts_to_send

    def html_formatting(self):
        """
        Turns the small data frame into html format
        :param df_new_small: 20 first jobs
        :return: the html for the message content
        """
        text_jobs_html = []
        for index, row in self.adverts_to_send.head(20).iterrows():
            title = row['Title']
            company = row['Company']
            place = row['Place']
            if len(row['Job description']) > 450:
                limit = row['Job description'].index(' ', 400)
                description = row['Job description'][:limit] + "  <a href='" + row['Link'] + \
                              "' style='color:#07506E'>More...</a>"
            else:
                description = row['Job description'] + "  <a href='" + row['Link'] + \
                              "' style='color:#07506E'>More...</a>"
            text_job_html = "<h3>\n" + title + "</h3>" + \
                            "<h4>At " + company + ' in ' + place \
                            + "</h4>" + "<p style='text-align:justify'>" + description + "</p>"
            text_jobs_html.append(text_job_html)
        html_text = ("<html>\n"
                "<table align='left' border='0' cellpadding='2' style='font-family: Arial, sans-serif>"
                "cellspacing='2' width='650' style='border-collapse: collapse;'><tr><td bgcolor='#70bbd9'"
                "style='color: #07506E; font-family: Arial, sans-serif; font-size: 14px'>"
                "<h2 align= 'center'><br>New data science jobs available on "
                "<a href='https://www.jobs.ch/en/' style='color: #D5EAF3'>Job.ch</a> this week<br></h2>"
                "</td></tr>"
                "<tr><td><br></td></tr>"
                "<tr><td>" + "<br><hr color='#07506E'>".join(text_jobs_html) +
                "<br><br>"
                "<h3>All jobs <a href='https://www.jobs.ch/en/vacancies/?page=1&term=data+science' "
                "style='color:#07506E'>here</a></h3>"
                "<br><br></td></tr></table>"
                "<tr><td><p> <br> </p></td></tr>"
                "<body>\n"
                "</body>\n"
                "</html>\n")
        return html_text

    def sending_email(self):
        """
        Sends the email
        :param self:
        :return:
        """
        message = MIMEMultipart('alte   rnative')
        message['Subject'] = "Jobs in Data Science Alerts"
        message['From'] = self.email_address_sender
        message['To'] = self.email
        message.attach(MIMEText(self.html_text, 'html'))
        server = smtplib.SMTP('smtp.gmail.com', 25)
        server.ehlo()
        server.starttls()
        server.login(self.email_address_sender, self.password)
        server.sendmail(self.email_address_sender, self.email, message.as_string())
        server.quit()
