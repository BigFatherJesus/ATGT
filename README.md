# Auto Google Drive Translator

An automated system that monitors Google Drive folders for videos and text files, transcribes videos to SRT files, and translates content to Dutch using GPT-4.

## Features

- Automatic monitoring of configured Google Drive folders
- Video transcription to SRT files
- Translation of text files and transcriptions to Dutch
- Support for multiple input languages
- Automatic file organization

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_key

# Google Drive Folder IDs
TEXT_TRANSLATION_FOLDER_ID=your_translation_folder_id
OUTPUT_FOLDER_ID=your_output_folder_id
VIDEO_DUTCH_FOLDER_ID=your_dutch_folder_id
VIDEO_GERMAN_FOLDER_ID=your_german_folder_id
VIDEO_ENGLISH_FOLDER_ID=your_english_folder_id
VIDEO_OTHER_FOLDER_ID=your_other_folder_id

# System Prompt for Translation
SYSTEM_PROMPT=your_system_prompt
```

3. Set up Google Drive API:
   - Place your Google Drive service account credentials in `./credentials/credentials.json`

## Folder Structure

- `/Data` - Working directory for temporary files
- `/Data/Cache` - Temporary cache for processing
- `/Data/Output` - Local copies of processed files

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Monitor configured folders
2. Transcribe videos to SRT
3. Translate non-Dutch content to Dutch
4. Place translations in appropriate output folders

## File Processing

- **Videos**: Transcribed to SRT files
- **Text Files**: Directly translated
- **Dutch Content**: Sent directly to output folder
- **Other Languages**: Sent to translation queue

## Notes

- Videos are processed using WhisperX for transcription
- Translations are performed using GPT-4
- Files are automatically cleaned up after processing