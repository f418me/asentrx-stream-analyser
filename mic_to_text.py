#!/usr/bin/env python3
"""
Simple microphone-to-text program using OpenAI Whisper API
Records audio from default microphone and writes transcription to text file
"""

import os
import asyncio
import tempfile
import logging
from datetime import datetime
from pathlib import Path
import speech_recognition as sr
from openai import OpenAI

# Load .env file
env_file = Path('.env')
if env_file.exists():
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mic_transcription.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MicrophoneTranscriber:
    """Simple microphone transcriber using OpenAI Whisper API."""
    
    def __init__(self, output_file="microphone_transcription.txt"):
        self.output_file = output_file
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_running = False
        
        # Initialize OpenAI Client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env!")
        
        self.openai_client = OpenAI(api_key=api_key)
        
        # Calibrate microphone
        with self.microphone as source:
            logger.info("🎤 Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source)
        
        logger.info(f"✅ Initialized - Output: {self.output_file}")
    
    async def start_listening(self):
        """Starts continuous listening and transcription."""
        logger.info("🎬 Starting microphone transcription with OpenAI Whisper...")
        
        self.is_running = True
        
        # Prepare output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Microphone transcription started: {datetime.now().isoformat()}\n")
            f.write("# OpenAI Whisper API\n")
            f.write("# Press Ctrl+C to stop\n\n")
        
        segment_count = 0
        
        while self.is_running:
            try:
                segment_count += 1
                logger.info(f"🎵 Listening... (Segment {segment_count})")
                
                # Record audio and transcribe
                success = await self._record_and_transcribe()
                
                if success:
                    logger.info(f"✅ Segment {segment_count} transcribed")
                else:
                    logger.debug(f"🔇 Segment {segment_count} - no speech detected")
                
                # Short pause
                await asyncio.sleep(0.5)
                
            except KeyboardInterrupt:
                logger.info("👋 User interruption")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                await asyncio.sleep(2)
        
        self.is_running = False
        logger.info("🛑 Transcription stopped")
    
    async def _record_and_transcribe(self) -> bool:
        """Records audio and transcribes it with OpenAI Whisper."""
        try:
            # Record audio from microphone
            with self.microphone as source:
                # Timeout after 1 second, phrase_time_limit limits recording to 10 seconds
                audio_data = self.recognizer.listen(
                    source, 
                    timeout=1,  # Wait for speech
                    phrase_time_limit=10  # Max 10 seconds recording
                )
            
            # Transcribe with OpenAI Whisper API
            return await self._transcribe_with_openai_whisper(audio_data)
                
        except sr.WaitTimeoutError:
            # No audio in timeout period - normal
            return False
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return False
    
    async def _transcribe_with_openai_whisper(self, audio_data) -> bool:
        """Transcribes with OpenAI Whisper API."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
            # Save audio as WAV
            with open(temp_audio_path, 'wb') as f:
                f.write(audio_data.get_wav_data())
        
        try:
            # Check if audio file is large enough (at least 1KB)
            if os.path.getsize(temp_audio_path) < 1024:
                return False
            
            # Call OpenAI Whisper API
            loop = asyncio.get_event_loop()
            
            def call_whisper_api():
                with open(temp_audio_path, 'rb') as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                return transcript.text
            
            # API call in thread pool (since synchronous)
            text = await loop.run_in_executor(None, call_whisper_api)
            
            if text and text.strip():
                self._write_transcription(text.strip())
                logger.info(f"🎯 Whisper: '{text.strip()}'")
                return True
                    
        except Exception as e:
            logger.error(f"OpenAI Whisper API error: {e}")
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        
        return False
    
    def _write_transcription(self, text: str):
        """Writes transcription to output file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {text}\n")
            f.flush()
    
    def stop(self):
        """Stops transcription."""
        self.is_running = False


async def main():
    """Main function."""
    print("🎤 Microphone-to-Text Transcriber (OpenAI Whisper)")
    print("Press Ctrl+C to stop\n")
    
    # Optional: Output file as argument
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "microphone_transcription.txt"
    
    try:
        transcriber = MicrophoneTranscriber(output_file)
        await transcriber.start_listening()
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        print("\nPlease add OPENAI_API_KEY to .env file:")
        print("OPENAI_API_KEY=sk-...")
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        if 'transcriber' in locals():
            transcriber.stop()


if __name__ == "__main__":
    asyncio.run(main())