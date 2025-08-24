#!/usr/bin/env python3
"""
Simple trading integration for sentiment analysis results
Executes trades based on high-confidence analysis results
"""

import logging
from typing import Optional
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleFedTrader:
    """Simple trader that executes trades based on Fed sentiment analysis."""
    
    def __init__(self):
        """Initialize the trader with configuration from environment variables."""
        # Trading configuration
        self.enabled = os.getenv('TRADING_ENABLED', 'false').lower() == 'true'
        self.symbol = os.getenv('TRADE_SYMBOL', 'tBTCF0:USTF0')
        self.confidence_threshold = float(os.getenv('TRADING_CONFIDENCE_THRESHOLD', '0.8'))
        
        # Trade amounts in Bitcoin and leverage
        self.hawkish_btc_amount = float(os.getenv('HAWKISH_BTC_AMOUNT', '-0.001'))  # Short Bitcoin position
        self.dovish_btc_amount = float(os.getenv('DOVISH_BTC_AMOUNT', '0.001'))     # Long Bitcoin position
        self.leverage = int(os.getenv('TRADE_LEVERAGE', '3'))
        self.limit_offset = float(os.getenv('TRADE_LIMIT_OFFSET', '0.001'))  # 0.1%
        
        # API credentials
        self.api_key = os.getenv('BITFINEX_API_KEY')
        self.api_secret = os.getenv('BITFINEX_API_SECRET')
        
        if self.enabled:
            if not self.api_key or not self.api_secret:
                logger.error("Trading enabled but BITFINEX_API_KEY or BITFINEX_API_SECRET not found!")
                self.enabled = False
            else:
                logger.info(f"Trading enabled for {self.symbol} with confidence threshold {self.confidence_threshold}")
                # Try to import the actual trader
                try:
                    from .trader import Trader
                    from .bitfinex_trader import BitfinexTrader
                    
                    bfx_trader = BitfinexTrader(api_key=self.api_key, api_secret=self.api_secret)
                    self.trader = Trader(bfx_trader)
                    logger.info("Bitfinex trader initialized successfully")
                except ImportError as e:
                    logger.error(f"Could not import trading modules: {e}")
                    self.enabled = False
                except Exception as e:
                    logger.error(f"Error initializing Bitfinex trader: {e}")
                    self.enabled = False
        else:
            logger.info("Trading disabled")
            self.trader = None
    
    def should_trade(self, sentiment: str, confidence: float) -> bool:
        """Check if we should execute a trade based on sentiment and confidence."""
        if not self.enabled:
            return False
            
        if confidence < self.confidence_threshold:
            logger.debug(f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}")
            return False
            
        if sentiment not in ['hawkish', 'dovish']:
            logger.debug(f"Sentiment '{sentiment}' not tradeable (only hawkish/dovish)")
            return False
            
        return True
    
    def execute_sentiment_trade(self, sentiment: str, confidence: float, reasoning: str, 
                              analysis_timestamp: str) -> Optional[dict]:
        """Execute a trade based on sentiment analysis."""
        if not self.should_trade(sentiment, confidence):
            return None
            
        # Determine trade parameters
        if sentiment == 'hawkish':
            btc_amount = self.hawkish_btc_amount  # Short Bitcoin (negative amount)
            trade_description = f"HAWKISH Short"
        else:  # dovish
            btc_amount = self.dovish_btc_amount   # Long Bitcoin (positive amount)
            trade_description = f"DOVISH Long"
        
        logger.info(f"🔥 HIGH CONFIDENCE TRADE SIGNAL:")
        logger.info(f"   Sentiment: {sentiment.upper()}")
        logger.info(f"   Confidence: {confidence:.3f}")
        logger.info(f"   Action: {trade_description}")
        logger.info(f"   Bitcoin Amount: {btc_amount} BTC")
        logger.info(f"   Leverage: {self.leverage}x")
        logger.info(f"   Reasoning: {reasoning[:100]}...")
        
        if not self.trader:
            logger.info("   🚨 SIMULATION MODE - No real trade executed")
            return {
                "status": "simulated",
                "sentiment": sentiment,
                "confidence": confidence,
                "btc_amount": btc_amount,
                "leverage": self.leverage,
                "symbol": self.symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Execute the actual trade
            order_result = self.trader.execute_order(
                symbol=self.symbol,
                amount=btc_amount,
                leverage=self.leverage,
                limit_offset_percentage=self.limit_offset
            )
            
            if order_result:
                logger.info(f"   ✅ Trade executed successfully: {order_result}")
                return {
                    "status": "executed",
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "btc_amount": btc_amount,
                    "leverage": self.leverage,
                    "symbol": self.symbol,
                    "order_result": order_result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error(f"   ❌ Trade execution failed")
                return {
                    "status": "failed",
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "btc_amount": btc_amount,
                    "error": "Order submission failed",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"   ❌ Trade execution error: {e}")
            return {
                "status": "error",
                "sentiment": sentiment,
                "confidence": confidence,
                "btc_amount": btc_amount,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_trade_log(self, trade_result: dict, output_file: str = "trade_results.jsonl"):
        """Save trade results to a log file."""
        try:
            import json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'a', encoding='utf-8') as f:
                json_line = json.dumps(trade_result)
                f.write(json_line + '\n')
                f.flush()
            
            logger.info(f"Trade result logged to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")