#!/bin/bash

for link in $(cat links); do
	yt-dlp -x --audio-format mp3 -o "/run/media/anton/hdd/data/%(title)s.%(ext)s" $link
done


