import os
import logging
import whisperx
import torch
import shutil
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from moviepy.editor import VideoFileClip
import soundfile as sf
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000  # WhisperX expects 16kHz audio

class VideoProcessor:
    def __init__(self):
        # Check CUDA availability
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA is available - using GPU")
        else:
            self.device = "cpu"
            logger.info("CUDA is not available - using CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

        self.compute_type = "float32"

        # Configure processing environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU
        self.device = "cpu"
        self.compute_type = "float32"

        # Setup directories
        self.cache_dir = os.path.join(os.getcwd(), "Data", "Cache")
        self.output_dir = os.path.join(os.getcwd(), "Data", "Output")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        try:
            logger.info("Loading WhisperX model...")
            self.model = whisperx.load_model(
                "large-v2",
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(f"WhisperX initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing WhisperX model: {e}")
            self.model = None

    def process_video(self, file_id, file_name, language_info, service):
        """Process a video file and create SRT."""
        video_path = None
        srt_path = None
        try:
            # Extract language code from language_info dict
            lang_code = language_info.get('code') if isinstance(language_info, dict) else None
            
            if not lang_code:
                logger.warning("No language code provided, defaulting to English")
                lang_code = 'en'
                
            # Convert language code to string if it's not already
            lang_code = str(lang_code)
            
            logger.info(f"Processing video in {language_info} (code: {lang_code})")
            
            # Download video to cache
            video_path = os.path.join(self.cache_dir, file_name)
            self._download_file(file_id, video_path, service)
            
            # Create SRT file
            srt_path = self.transcribe_video(video_path, lang_code)
            if not srt_path:
                raise Exception("Failed to create SRT file")

            # Save a copy in local output directory
            local_output_path = os.path.join(self.output_dir, os.path.basename(srt_path))
            shutil.copy2(srt_path, local_output_path)
            logger.info(f"Saved local copy to: {local_output_path}")

            # Determine Google Drive destination based on language
            if lang_code.lower() == 'nl':  # Dutch
                # Dutch files go directly to output folder
                output_folder_id = os.getenv('OUTPUT_FOLDER_ID')
                if not output_folder_id:
                    raise Exception("Output folder ID not configured")
                self._upload_srt(srt_path, output_folder_id, service)
                logger.info("Dutch SRT uploaded to Output folder")
            else:
                # Non-Dutch files go to input folder for translation
                input_folder_id = os.getenv('TEXT_TRANSLATION_FOLDER_ID')
                if not input_folder_id:
                    raise Exception("Input folder ID not configured")
                self._upload_srt(srt_path, input_folder_id, service)
                logger.info(f"{lang_code} SRT uploaded to Input folder for translation")

            return True
            
        except Exception as e:
            logger.error(f"Error processing video {file_name}: {e}")
            return False
            
        finally:
            self._cleanup_files(video_path, srt_path)

    def transcribe_video(self, video_path, language_code=None):
        """Extract audio and transcribe to SRT."""
        audio_path = None
        try:
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Extract audio
            logger.info("Extracting audio...")
            video = VideoFileClip(video_path)
            audio_path = os.path.join(self.cache_dir, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
            video.audio.write_audiofile(audio_path, codec="pcm_s16le", fps=SAMPLE_RATE)
            video.close()

            if not self.model:
                logger.error("WhisperX model not available")
                return None

            # Prepare audio in correct format
            audio_data = self.prepare_audio(audio_path)

            # Run transcription
            logger.info("Starting transcription...")
            try:
                # Add more detailed logging for the transcription process
                logger.info(f"Transcribing with language: {language_code}")
                result = self.model.transcribe(
                    audio_data,
                    batch_size=1,
                    language=language_code
                )
                
                logger.info(f"Raw transcription result: {result}")  # Log the raw result

                if not isinstance(result, dict):
                    logger.error(f"Unexpected result type: {type(result)}")
                    return None

                if 'segments' not in result:
                    logger.error("No segments found in transcription result")
                    logger.error(f"Available keys in result: {result.keys()}")
                    return None

                segments = result['segments']
                if not segments:
                    logger.warning("No segments were transcribed")
                    return None

                logger.info(f"Transcription completed with {len(segments)} segments")

                # Write SRT
                srt_path = os.path.join(self.cache_dir, os.path.splitext(os.path.basename(video_path))[0] + ".srt")
                if self._write_srt(segments, srt_path):
                    logger.info(f"SRT file created successfully: {srt_path}")
                    return srt_path
                else:
                    logger.error("Failed to write SRT file")
                    return None

            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                logger.exception("Detailed transcription error:")
                return None

        except Exception as e:
            logger.error(f"Error in transcribe_video: {str(e)}")
            logger.exception("Detailed error:")
            return None

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info("Cleaned up temporary audio file")
                except Exception as e:
                    logger.error(f"Error cleaning up audio file: {str(e)}")

    def prepare_audio(self, audio_path):
        """Prepare audio data in the correct format for WhisperX."""
        logger.info(f"Loading audio file: {audio_path}")

        # Load audio with soundfile
        audio_data, sr = sf.read(audio_path)
        logger.info(f"Original audio: shape={audio_data.shape}, sr={sr}, dtype={audio_data.dtype}")

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            logger.info("Converted stereo to mono")

        # Ensure float32 dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        logger.info(f"Prepared audio shape: {audio_data.shape}")
        return audio_data

    def _write_srt(self, segments, output_path):
        """Write segments to SRT file."""
        try:
            if not segments:
                logger.warning("No segments to write to SRT file")
                return False

            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_timestamp(segment["start"])
                    end_time = self._format_timestamp(segment["end"])
                    text = segment["text"].strip()

                    if not text:  # Skip empty segments
                        continue

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            return True

        except Exception as e:
            logger.error(f"Error writing SRT file: {e}")
            return False

    def _format_timestamp(self, seconds):
        """Format time in SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def check_existing_translation(self, file_name, service, output_folder_id, translation_folder_id):
        """Check if translation already exists in output or translation folders."""
        base_without_ext = os.path.splitext(file_name)[0]
        possible_names = [
            f"{base_without_ext}_AI_Translated.srt",  # Standard translated name
            f"{base_without_ext}.srt",                # Original name in translation queue
        ]

        logger.info(f"Checking for existing translations of {file_name}...")

        for folder_id in [output_folder_id, translation_folder_id]:
            for name in possible_names:
                query = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
                try:
                    results = service.files().list(
                        q=query,
                        fields="files(id, name)",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True
                    ).execute()

                    if results.get('files', []):
                        logger.info(f"Found existing file in folder: {name}")
                        return True
                except Exception as e:
                    logger.error(f"Error checking folder: {e}")

        logger.info("No existing translations found, proceeding with transcription")
        return False

    def _download_file(self, file_id, destination_path, service):
        """Download a file from Google Drive."""
        try:
            request = service.files().get_media(fileId=file_id)
            with open(destination_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.info(f"Download progress: {int(status.progress() * 100)}%")
            logger.info(f"Downloaded file to: {destination_path}")
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def _upload_srt(self, srt_path, folder_id, service):
        """Upload SRT file to Google Drive."""
        try:
            file_metadata = {
                'name': os.path.basename(srt_path),
                'parents': [folder_id]
            }
            media = MediaFileUpload(srt_path, resumable=True)
            upload_response = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            media._fd.close()
            logger.info(f"Successfully uploaded SRT to folder: {os.path.basename(srt_path)}")
        except Exception as e:
            logger.error(f"Error uploading SRT file: {e}")
            raise

    def _cleanup_files(self, *file_paths):
        """Clean up temporary files."""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up file: {e}")

def process_video_files(folder_info, service):
    """Process video files in a specific language folder."""
    folder_id = folder_info.get("folder_id")
    language = folder_info.get("code")
    
    if not folder_id:
        logger.warning(f"No folder ID configured for {language} videos")
        return
        
    logger.info(f"Checking {language} video folder")
    files = list_files(folder_id)
    
    for file in files:
        if is_video_file(file.get('mimeType', '')):
            video_processor = VideoProcessor()
            success = video_processor.process_video(
                file['id'],
                file['name'],
                folder_info,  # Pass the complete folder info dict
                service
            )
            if success:
                logger.info(f"Successfully processed video: {file['name']}")
            else:
                logger.error(f"Failed to process video: {file['name']}")
