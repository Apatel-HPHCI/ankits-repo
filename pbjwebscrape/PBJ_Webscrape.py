# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:39:01 2018

@author: ankit.patel
"""
from bs4 import BeautifulSoup 
# Selenium for full html parsing when JS files are called
from selenium import webdriver
# urllib to open requested web links
import urllib.request
# Time to delay program
import time
# pandas to write dataset names and link to dataframe
import pandas as pd

#Start timer to see how long it takes script to run
start_time = time.perf_counter()

def pbj():


# Request the needed url. The following link searches all relevant PBJ daily date, sorted alphabetically
    html_page = urllib.request.urlopen("https://data.cms.gov/browse?q=pbj%20daily%20staffing&sortBy=alpha")
# Pass to beautiful soup
    soup = BeautifulSoup(html_page, 'html.parser')

# Initialize empty list to start appending links
    linklist = []

# Loop and get all API weblinks to store into list
    for link in soup.findAll('a', class_="browse2-result-api-link"):
        try:
            linkstring = str(link.get('href'))
      # This will append each weblink until the loop is done
            linklist.append(linkstring)
      # NavigableString is some weird string element that gets caught when looking
      # for html tags. Skip over it if it is found
        except NavigableString:
            pass

# Load up chromedriver options. Headless launches the browser silently
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)

# Initalize two lists, one for PBJ dataset name and one for the dataset links
    jsonlist = []
    namelist = []

# Loop each website link
    for jsonlink in linklist:
    # Pass link to Chrome
        driver.get(jsonlink)
    # Allow 10 second wait for all HTML elements to load
        time.sleep(10)

# Pass all page source elements to object and then to BeautifulSoup
        jsonlink = driver.page_source

        jsonsoup = BeautifulSoup(jsonlink, 'html.parser')

# Loop to get dataset names
        for nametext in jsonsoup.find_all('h1',attrs={'id':'title'}):
            namestring = nametext.text
            namelist.append(namestring)
# Loop to get API dataset links
        
            for jsonlink in jsonsoup.find_all('a',class_="exec target has-tooltip ga-track"):
                jsonstring = str(jsonlink.get('href'))
                jsonlist.append(jsonstring)
            # We want all the dataset names, so break out of most inner loop and go back to top
                break

# End all chromedriver processes
    driver.quit()


# Replace json extension with csv
    jsonlist = [j.replace('json','csv') for j in jsonlist]

#Store lists in dataframe 

    df = pd.DataFrame(list(zip(namelist,jsonlist)),
                      columns = ['Name','Link'])   
    return df

    print("--- %s seconds ---" % (time.perf_counter() - start_time))    

df = pbj()

print("--- %s seconds ---" % (time.perf_counter() - start_time))