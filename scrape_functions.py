# Webscraping
import requests
import re
from bs4 import BeautifulSoup

# Analysis
import numpy as np
import pandas as pd

# Find table rows helper function
def row_data(soup):
    # Find Supreme Court opinions 'table'
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Find table headers
    headers = table.find_all('th')

    # Create list of justices from headers
    not_justices = ['#', 'Case name and citation', 'Argued', 'Decided']
    names = [ header.text.strip() for header in headers if header.text.strip() not in not_justices ]

    # Remove duplicate names
    justices = []
    for name in names:
        if name not in justices:
            justices.append(name)

    # Find rows
    first_row = table.find('tr')
    rows = first_row.next_siblings # Data is here
    
    return rows, justices

# Scrape opinions helper function
def scrape_opinions(rows, justices):
    j_votes = []
    cases = []
    n = 0
    for i, row in enumerate(rows):
        if i % 2 == 0:
            votes = []
            data = row.find_all('td')
            for m, datum in enumerate(data):
                if m == 1:
                    cases.append(datum.text.strip())
                elif m in [0, 2, 3]:
                    continue
                else:
                    if re.match(r'padding', datum['style']):
                        vote = datum['data-sort-value']
                        if re.match(r'^(.*)?(?=<)', datum['data-sort-value']):
                            vote = re.match(r'^(.*)?(?=<)', datum['data-sort-value'])[0].strip()
                        votes.append(vote)
            
            if votes != []: # Some rows are not cases
                assert len(votes) == len(justices), 'Number of votes/non-votes different than number of justices'
                j_votes.append(votes)
                
    assert len(cases) == len(j_votes), 'Number of cases different than justice opinions'
    
    return j_votes, cases

# Complete scrape function
def get_data(soup):
    rows, justices = row_data(soup)
    j_votes, cases = scrape_opinions(rows, justices)
    votes_by_case = dict(zip(cases, j_votes)) # Create dictionary (ex. {'case': [votes]})
    return votes_by_case, justices

# Get all urls function
def make_urls(base_url, list_of_var):
    urls = []
    for var in list_of_var:
        url = base_url.format(var)
        urls.append(url)
    return urls