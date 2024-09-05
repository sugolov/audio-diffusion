#!/bin/bash

./src/soundcloud_scraper.py ./links/sc_links.txt $1 training_data/raw_audio/soundcloud_audio/ --processes 8

#example : ../../../Downloads/chromedriver-linux64/chromedriver-linux64/chromedriver
