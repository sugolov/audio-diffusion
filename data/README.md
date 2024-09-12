## Data Collection and Preparation

Scripts for scraping, storing, and processing breakcore music from the internet.

### Training data directory structure:
**Note:** The following `./training_data` subdirectory is gitignored in order to prevent uploading tons of training data to GitHub. This directory should exist on a local machine / server database.

```
.
└── training_data/
    ├── raw_audio/
    │   ├── youtube_audio/
    │   │   ├── yt_track1.mp3
    │   │   ├── yt_track2.mp3
    │   │   ├── ...
    │   │   └── yt_track_n.mp3
    │   └── soundcloud_audio/
    │       ├── sc_track1.mp3
    │       ├── sc_track2.mp3
    │       ├── ...
    │       └── sc_track_n.mp3
    └── processed_audio/
        ├── youtube_processed/
        │   ├── yt_track1/
        │   │   ├── yt_track1_000_mel.npy
        │   │   ├── yt_track1_001_mel.npy
        │   │   ├── ...
        │   │   └── yt_track1_xxx_mel.npy
        │   ├── yt_track2/
        │   ├── ...
        │   └── yt_track_n/
        └── soundcloud_processed/
            ├── sc_track1/
            │   ├── sc_track1_000_mel.npy
            │   ├── sc_track1_001_mel.npy
            │   ├── ...
            │   └── sc_track1_xxx_mel.npy
            ├── sc_track2/  
            ├── ...
            └── sc_track_n/
```

---

### Scraping audio from YouTube:

1. Ensure that this is your current working directory: `audio-diffusion/data/`
2. Install dependencies: `pip install -r requirements.txt`
3. Append any new youtube links to: `./links/yt_links.txt`
4. Run: `./scrape_yt.sh`
5. Raw mp3 data will be saved to: `./training_data/raw_audio/youtube_audio/`

---

### Processing YouTube audio into mel-spectrograms:

1. Ensure that this is your current working directory: `audio-diffusion/data/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `./process_yt.sh`
4. Mel-spectrograms processed from all the youtube mp3 data will be saved to: `./training_data/processed_audio/youtube_processed/`

---

### Scrape Audio from Soundcloud

1. Ensure that this is your current working directory: `audio-diffusion/data/`
2. Install Chrome driver: https://developer.chrome.com/docs/chromedriver/downloads. Note it's path relative to the working directory
3. Install dependencies: `pip install -r requirements.txt`
4. Append any soundcloud links to `sc_links.txt`
5. run `./scrape_sc.sh /relative/path/to/chrome-driver`

---

### Processing Soundcloud audio into mel-spectrograms

1. Ensure that this is your current working directory: `audio-diffusion/data/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `./process_sc.sh`
4. Mel-spectrograms processed from all the youtube mp3 data will be saved to: `./training_data/processed_audio/soundcloud_processed/`

## TODO
- scraping pipeline for soundcloud
- scrape harder (for example, download all the mixes of a particular series)
- use spreadsheet to record collected track links and their respective runtimes (goal is ~500 hours)
