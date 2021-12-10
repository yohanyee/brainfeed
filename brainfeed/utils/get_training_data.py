#!/usr/bin/env python3

import os
from habanero import Crossref

f = cr.filter_details('works')


BRAINFEED_EMAIL = os.environ['BRAINFEED_EMAIL']

cr = Crossref(mailto=BRAINFEED_EMAIL,
              ua_string="brainfeed:v0.1 \
              (https://github.com/yohanyee/brainfeed)"
              )


journalquery = cr.journals(query='NeuroImage') #Returns a dict, use keys() method

journal = journalquery['message']['items'][0]
journal.keys()
journal['title']
journal['ISSN']

f = cr.filter_details('works')
article_filters = {'has_abstract': True,
                   'from_created_date': '2014-03-03'}
response = cr.journals(ids=journal['ISSN'][0],
                       filter=article_filters,
                       works=True, progress_bar=True)
response['message']['total-results']
len(response['message']['items'])

response = cr.journals(ids=journal['ISSN'][0],
                       cursor = "*", cursor_max = 1000,
                       filter=article_filters,
                       works=True, progress_bar=True)
