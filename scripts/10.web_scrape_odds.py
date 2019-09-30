#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 08:57:33 2018

@author: angus
"""

from selenium import webdriver
from bs4 import BeautifulSoup

url = "https://www.betfair.com/sport/basketball"
driver = webdriver.Chrome()
driver.get(url)
soup = BeautifulSoup(driver.page_source, 'html.parser')

containers = soup.findAll("table", {"class": "marketboard-event-with-header__markets-list"})


resultAndOdds = []    
for container in containers:
    divs = container.findAll('div')
    texts = [div.text for div in divs]
    it = iter(texts)
    resultAndOdds.append(list(zip(it, it)))
    
resultAndOdds[0]

resultAndOdds[1]

titlesElements = soup.findAll("div", {"class":"marketboard-event-with-header__market-name"})
titlesTexts = [title.text for title in titlesElements]


