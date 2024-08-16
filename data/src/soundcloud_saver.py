import os
import yt_dlp as youtube_dl  # Use yt-dlp instead of youtube_dl
import argparse
from multiprocessing import Pool

def download_soundcloud_track(track_url, download_folder):
    # Extract the artist name and song name from the URL
    url_parts = track_url.strip().split('/')
    artist_name = url_parts[-2]
    song_name = url_parts[-1]

    # Construct the output template to save the file as "song_name - artist_name.mp3"
    output_template = os.path.join(download_folder, f'{song_name} - {artist_name}.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([track_url])
        print(f"Successfully downloaded: {track_url}")
    except Exception as e:
        print(f"Failed to download {track_url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download SoundCloud tracks from a list of URLs.")
    parser.add_argument('input_file', type=str, help="Path to the .txt file containing SoundCloud links")
    parser.add_argument('output_dir', type=str, help="Directory where MP3s will be saved")
    parser.add_argument('--processes', type=int, default=4, help="Number of download processes")

    args = parser.parse_args()

    # Make sure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read the file with the list of URLs
    with open(args.input_file, 'r') as file:
        links = [link.strip() for link in file.readlines() if link.strip()]

    # Download tracks using multiple processes
    with Pool(processes=args.processes) as pool:
        pool.starmap(download_soundcloud_track, [(link, args.output_dir) for link in links])

if __name__ == "__main__":
    main()
