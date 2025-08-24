#!/usr/bin/env python3
"""
Standalone AI Analyzer
Reads transcribed text from a file and analyzes it with AI.
"""

import asyncio
import os
import json
import logging
from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Import trading functionality
try:
    from trading.simple_trader import SimpleFedTrader
    TRADING_AVAILABLE = True
except ImportError as e:
    TRADING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Trading functionality not available: {e}")
from pydantic_ai.models.google import GoogleModel


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Models for structured AI responses
class SentimentResult(BaseModel):
    """Result of sentiment analysis."""
    
    overall_sentiment: str = Field(
        description="Overall sentiment: hawkish, dovish or neutral"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the analysis (0-1)"
    )
    expectation_deviation: str = Field(
        description="Deviation from expectations: more_hawkish, more_dovish, as_expected"
    )
    market_prediction: str = Field(
        description="Market prediction: bullish, bearish or neutral"
    )
    reasoning: str = Field(
        min_length=1,
        description="Analysis reasoning"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )


class AIAnalyzer:
    """AI analyzer for sentiment analysis of transcribed texts."""
    
    def __init__(self, gemini_api_key: str, expectations_file: str = "expectations.txt", 
                 model_name: str = "gemini-1.5-pro"):
        """
        Args:
            gemini_api_key: API key for Gemini
            expectations_file: File with market expectations
            model_name: Gemini-Modell Name
        """
        self.expectations_file = expectations_file
        self.expectations_text = self._load_expectations()
        self.model_name = model_name
        
        # Set Gemini API key
        os.environ['GOOGLE_GENAI_API_KEY'] = gemini_api_key
        
        # Pydantic AI Agent erstellen
        self._agent = Agent[None, SentimentResult](
            self.model_name,
            output_type=SentimentResult,
            system_prompt=self._build_system_prompt()
        )
        
        # Initialize trader if available
        self.trader = None
        if TRADING_AVAILABLE:
            try:
                self.trader = SimpleFedTrader()
                logger.info("Trading functionality initialized")
            except Exception as e:
                logger.error(f"Failed to initialize trader: {e}")
        
        logger.info(f"AI Analyzer initialized with Model: {self.model_name}")
    
    def _load_expectations(self) -> str:
        """Loads market expectations from file."""
        try:
            expectations_path = Path(self.expectations_file)
            if expectations_path.exists():
                with open(expectations_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                logger.info(f"Erwartungen geladen aus: {self.expectations_file}")
                return content
            else:
                logger.warning(f"Expectations file not found: {self.expectations_file}")
                return "Keine spezifischen Erwartungen definiert."
        except Exception as e:
            logger.error(f"Error loading expectations: {e}")
            return "Error loading expectations."
    
    def _build_system_prompt(self) -> str:
        """Creates the system prompt for the AI."""
        return f"""You are an expert in financial market sentiment analysis, specialized in analyzing central bank communication and its impact on Bitcoin markets.

TASK:
Analyze the given text for hawkish (restrictive) or dovish (expansive) monetary policy signals and their impact on Bitcoin.

DEFINITIONS:
- HAWKISH: Signals for restrictive monetary policy (interest rate increases, inflation fighting, liquidity withdrawal)
- DOVISH: Signals for expansive monetary policy (interest rate cuts, economic stimulus, liquidity provision)

MARKET EXPECTATIONS:
{self.expectations_text}

ANALYSIS CRITERIA:
1. Identify hawkish/dovish signals in the text
2. Compare with predefined expectations
3. Evaluate signal strength (Confidence 0-1)
4. Derive Bitcoin market direction:
   - Hawkish → Bearish for Bitcoin (higher rates = less risk appetite)
   - Dovish → Bullish for Bitcoin (lower rates = more risk appetite)

CONFIDENCE RATING:
- 0.9-1.0: Very clear, unambiguous signals
- 0.7-0.9: Strong signals with low ambiguity
- 0.5-0.7: Moderate signals, some uncertainty
- 0.3-0.5: Weak signals, high uncertainty
- 0.0-0.3: No clear signals or very ambiguous

Always respond in the specified JSON structure with English reasoning."""
    
    async def analyze_text(self, text: str) -> SentimentResult:
        """Analyzes a text with AI."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            analysis_prompt = f"""Analyze the following text for hawkish/dovish sentiment:

TEXT:
{text}

Perform a detailed analysis considering:
1. Explicit monetary policy statements
2. Implicit signals and tone
3. Comparison with defined expectations
4. Impact on Bitcoin markets

Provide a structured response with high precision."""
            
            logger.info(f"Analyzing text: {text[:100]}...")
            
            # AI analysis with timeout
            result = await asyncio.wait_for(
                self._agent.run(analysis_prompt),
                timeout=30.0
            )
            
            sentiment_result = result.output
            logger.info(f"Analysis completed: {sentiment_result.overall_sentiment} "
                       f"(Confidence: {sentiment_result.confidence:.3f})")
            
            return sentiment_result
            
        except asyncio.TimeoutError:
            logger.error("AI analysis timeout")
            raise
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            raise
    
    def save_analysis_result(self, result: SentimentResult, output_file: str = "analysis_results.jsonl"):
        """Saves analysis result in JSON Lines format and executes trade if conditions are met."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append result as JSON line
            with open(output_path, 'a', encoding='utf-8') as f:
                json_line = result.model_dump_json()
                f.write(json_line + '\n')
                f.flush()
            
            logger.info(f"Analysis result saved to: {output_file}")
            
            # Execute trade if trader is available and confidence is high enough
            if self.trader:
                trade_result = self.trader.execute_sentiment_trade(
                    sentiment=result.overall_sentiment,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    analysis_timestamp=result.timestamp.isoformat()
                )
                
                if trade_result:
                    self.trader.save_trade_log(trade_result)
            
        except Exception as e:
            logger.error(f"Error saving: {e}")


class TextFileMonitor:
    """Monitors a text file for new content."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.last_position = 0
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensures the file exists."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()
    
    def get_new_lines(self) -> list[str]:
        """Returns new lines since last check."""
        try:
            if not self.file_path.exists():
                return []
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()
            
            if new_content.strip():
                return [line.strip() for line in new_content.strip().split('\n') if line.strip()]
            
            return []
            
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []


async def monitor_and_analyze_full_content(input_file: str, analyzer: AIAnalyzer, 
                                        output_file: str = "analysis_results.jsonl", 
                                        check_interval: float = 10.0):
    """Monitors an input file and analyzes the entire content every 10 seconds."""
    logger.info(f"Monitoring file: {input_file}")
    logger.info(f"Results will be saved to: {output_file}")
    logger.info(f"Analysis interval: {check_interval} seconds")
    
    last_content = ""
    
    try:
        while True:
            try:
                # Read entire file content
                if Path(input_file).exists():
                    with open(input_file, 'r', encoding='utf-8') as f:
                        current_content = f.read().strip()
                    
                    # Only analyze if content has changed
                    if current_content and current_content != last_content:
                        # Header-Zeilen und Kommentare herausfiltern
                        lines = current_content.split('\n')
                        content_lines = [line.strip() for line in lines 
                                       if line.strip() and not line.strip().startswith('#')]
                        
                        if content_lines:
                            # Alle Transkriptionen zu einem Text zusammenfassen
                            transcribed_texts = []
                            for line in content_lines:
                                if line.startswith('[') and ']' in line:
                                    timestamp_end = line.find(']')
                                    if timestamp_end > 0:
                                        text_content = line[timestamp_end + 1:].strip()
                                        if text_content:
                                            transcribed_texts.append(text_content)
                            
                            if transcribed_texts:
                                full_text = " ".join(transcribed_texts)
                                
                                try:
                                    # Gesamten Text analysieren
                                    result = await analyzer.analyze_text(full_text)
                                    
                                    # Ergebnis speichern
                                    analyzer.save_analysis_result(result, output_file)
                                    
                                    # Kurzfassung ausgeben
                                    print(f"\n{'='*60}")
                                    print(f"FULL-TEXT ANALYSIS ({len(transcribed_texts)} segments)")
                                    print(f"{'='*60}")
                                    print(f"Text: {full_text[:200]}...")
                                    print(f"Sentiment: {result.overall_sentiment}")
                                    print(f"Marktprognose: {result.market_prediction}")
                                    print(f"Confidence: {result.confidence:.3f}")
                                    print(f"Reasoning: {result.reasoning}")
                                    print(f"{'='*60}\n")
                                    
                                    last_content = current_content
                                    
                                except Exception as e:
                                    logger.error(f"Error in full-text analysis: {e}")
                
            except Exception as e:
                logger.error(f"Fehler beim Lesen der Datei: {e}")
            
            # Wait until next check
            await asyncio.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")


async def monitor_and_analyze(input_file: str, analyzer: AIAnalyzer, 
                            output_file: str = "analysis_results.jsonl", 
                            check_interval: float = 2.0):
    """Monitors an input file and analyzes new content."""
    monitor = TextFileMonitor(input_file)
    logger.info(f"Monitoring file: {input_file}")
    logger.info(f"Results will be saved to: {output_file}")
    
    try:
        while True:
            new_lines = monitor.get_new_lines()
            
            for line in new_lines:
                # Zeile parsen (Format: [TIMESTAMP] TEXT)
                if line.startswith('[') and ']' in line:
                    timestamp_end = line.find(']')
                    if timestamp_end > 0:
                        text_content = line[timestamp_end + 1:].strip()
                        
                        if text_content:
                            try:
                                # Text analysieren
                                result = await analyzer.analyze_text(text_content)
                                
                                # Ergebnis speichern
                                analyzer.save_analysis_result(result, output_file)
                                
                                # Kurzfassung ausgeben
                                print(f"\n{'='*60}")
                                print(f"NEUE ANALYSE")
                                print(f"{'='*60}")
                                print(f"Text: {text_content[:100]}...")
                                print(f"Sentiment: {result.overall_sentiment}")
                                print(f"Marktprognose: {result.market_prediction}")
                                print(f"Confidence: {result.confidence:.3f}")
                                print(f"Begründung: {result.reasoning}")
                                print(f"{'='*60}\n")
                                
                            except Exception as e:
                                logger.error(f"Fehler bei Analyse von '{text_content[:50]}...': {e}")
            
            # Wait until next check
            await asyncio.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")


async def main():
    """Main function for AI analysis."""
    # Configuration from environment variables
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    input_file = os.getenv('TRANSCRIPTION_INPUT_FILE', 'microphone_transcription.txt')
    output_file = os.getenv('ANALYSIS_OUTPUT_FILE', 'analysis_results.jsonl')
    expectations_file = os.getenv('EXPECTATIONS_FILE', 'expectations.txt')
    check_interval = float(os.getenv('CHECK_INTERVAL', '10.0'))
    analysis_mode = os.getenv('ANALYSIS_MODE', 'fulltext')  # 'fulltext' or 'incremental'
    model_name = os.getenv('MODEL', 'gemini-1.5-pro')
    
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable is required!")
        print("Beispiel: export GEMINI_API_KEY='your_api_key_here'")
        return
    
    try:
        # AI Analyzer initialisieren
        analyzer = AIAnalyzer(gemini_api_key, expectations_file, model_name)
        
        print(f"Starting AI analysis...")
        print(f"Model: {model_name}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Erwartungen: {expectations_file}")
        print(f"Modus: {analysis_mode}")
        print(f"Intervall: {check_interval} Sekunden")
        print("Press Ctrl+C to stop...")
        
        # Monitor and analyze file
        if analysis_mode == 'fulltext':
            await monitor_and_analyze_full_content(input_file, analyzer, output_file, check_interval)
        else:
            await monitor_and_analyze(input_file, analyzer, output_file, check_interval)
        
    except KeyboardInterrupt:
        print("\nAI analysis stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    asyncio.run(main())