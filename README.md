# Asentrx Stream Analyser

AI-powered tool that transcribes audio from your microphone and analyzes the content in real-time for financial market sentiment using Google's Gemini AI. It includes an optional, experimental feature for automated trading based on the analysis.

## 🎯 Overview

This project consists of two main components:

1.  **`mic_to_text.py`**: Transcribes audio from your microphone in real-time using the OpenAI Whisper API.
2.  **`ai_analyzer.py`**: Analyzes the transcribed text for hawkish or dovish sentiment using Google's Gemini AI and assesses its potential impact on Bitcoin. It can optionally trigger trades on Bitfinex.

*Note: This project focuses on microphone input as it provides a more stable and reliable audio source for real-time analysis compared to the complexities and potential instability of direct YouTube live streaming.*

## ✨ Features

*   **Real-time Microphone Transcription**: Captures and transcribes your speech using OpenAI Whisper.
*   **Advanced Sentiment Analysis**: Uses Google Gemini to analyze text for nuanced financial sentiment (hawkish/dovish).
*   **Market Context**: Calibrates AI analysis using a custom `expectations.txt` file to align with current market expectations.
*   **Optional Automated Trading**: Can automatically execute trades on Bitfinex based on high-confidence sentiment signals.
*   **Configurable & Extensible**: Easily configured through environment variables.

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.8+
*   FFmpeg for audio processing.
    *   **macOS**: `brew install ffmpeg`
    *   **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
    *   **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Setup

```bash
# Clone the repository
git clone <repository_url>
cd asentrx-stream-analyser

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file by copying the example and fill in your API keys.

```bash
cp .env.example .env
```

Now, edit the `.env` file with your favorite editor:

*   `OPENAI_API_KEY`: Your OpenAI API key (for Whisper transcription).
*   `GEMINI_API_KEY`: Your Google Gemini API key (for sentiment analysis).
*   **Optional (for trading)**:
    *   `TRADING_ENABLED=true`
    *   `BITFINEX_API_KEY`
    *   `BITFINEX_API_SECRET`

### 4. Calibrate the AI

Edit the `expectations.txt` file to provide the AI with the current market context. This helps the model make more accurate assessments.

```
MARKET EXPECTATIONS FOR CENTRAL BANK COMMUNICATION:

1. INTEREST RATES:
   - Expectation: A slight interest rate hike of 0.25% is anticipated.
   - Hawkish Signals: "fight inflation", "restrictive measures"
   - Dovish Signals: "economic support", "flexible monetary policy"
...
```

## 🖥️ Usage

You need to run the two main scripts in separate terminal windows.

**Terminal 1: Start Microphone Transcription**

```bash
source venv/bin/activate
python mic_to_text.py
```

This will start listening to your microphone and write the transcription to `microphone_transcription.txt`.

**Terminal 2: Start AI Analysis**

```bash
source venv/bin/activate
python ai_analyzer.py
```

This script monitors `microphone_transcription.txt` for new content, analyzes it, and writes the results to `analysis_results.jsonl`. If trading is enabled, it will also execute trades based on the analysis.

## 📁 Project Structure

```
asentrx-stream-analyser/
├── mic_to_text.py            # Microphone transcription (OpenAI Whisper)
├── ai_analyzer.py            # AI analysis script (Google Gemini)
├── trading/
│   └── simple_trader.py      # Trading logic
├── requirements.txt          # Python dependencies
├── .env.example              # Configuration template
├── expectations.txt          # Market expectations for the AI
└── README.md                 # This documentation
```

## ⚙️ Configuration Details

The behavior of the scripts can be fine-tuned using environment variables in the `.env` file.

**`ai_analyzer.py`**

*   `MODEL`: The Gemini model to use (default: `gemini-1.5-pro`).
*   `TRANSCRIPTION_INPUT_FILE`: The file to monitor (default: `microphone_transcription.txt`).
*   `ANALYSIS_OUTPUT_FILE`: Where to save analysis results (default: `analysis_results.jsonl`).
*   `CHECK_INTERVAL`: How often to check for new text (default: `10.0` seconds).
*   `ANALYSIS_MODE`: `fulltext` (analyzes the whole file each time) or `incremental` (analyzes only new lines). Default is `fulltext`.

**Trading (`trading/simple_trader.py`)**

*   `TRADING_ENABLED`: Set to `true` to enable trading.
*   `TRADE_SYMBOL`: The symbol to trade on Bitfinex (default: `tBTCF0:USTF0`).
*   `TRADING_CONFIDENCE_THRESHOLD`: Minimum AI confidence to trigger a trade (default: `0.8`).
*   `HAWKISH_BTC_AMOUNT`: Amount to short for a "hawkish" signal (default: `-0.001`).
*   `DOVISH_BTC_AMOUNT`: Amount to long for a "dovish" signal (default: `0.001`).
*   `TRADE_LEVERAGE`: Leverage to use for trades (default: `3`).

⚠️ **Disclaimer**: The trading feature is experimental. Use it at your own risk.

## 🛠️ Troubleshooting

*   **"FFmpeg not found"**: Ensure FFmpeg is installed and accessible in your system's PATH.
*   **Microphone Issues**: Check that your microphone is connected and that the application has the necessary permissions.
*   **API Errors**: Double-check your API keys in the `.env` file and ensure you have sufficient credits/quota on the OpenAI and Google AI platforms.

For more detailed logs, check `mic_transcription.log` and `ai_analysis.log`.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
