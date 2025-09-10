"""
SEC Filing Data Collector for FinGraph - FIXED VERSION
Uses mock relationship data for reliable portfolio demonstration
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import json

logger = logging.getLogger(__name__)

class SECCollector:
    """
    Collects company relationship data from SEC filings
    
    NOTE: For portfolio demonstration, this uses curated mock data
    representing real business relationships. In production, you would
    implement full SEC EDGAR API integration.
    """
    
    def __init__(self, rate_limit: float = 0.1):
        """
        Initialize SEC collector
        
        Args:
            rate_limit: Seconds between requests (SEC requires rate limiting)
        """
        self.rate_limit = rate_limit
        self.use_mock_data = True  # For portfolio demonstration
        
        # Headers required by SEC (even though we're using mock data)
        self.headers = {
            'User-Agent': 'FinGraph Research Project research@email.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
    
    def get_company_cik(self, symbol: str) -> Optional[str]:
        """
        Get company CIK from stock symbol
        
        NOTE: This is a curated mapping for major companies
        """
        # Extended company CIK mappings for demo
        symbol_to_cik = {
            'AAPL': '0000320193',
            'MSFT': '0000789019', 
            'GOOGL': '0001652044',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'META': '0001326801',
            'NVDA': '0001045810',
            'NFLX': '0001065280',
            'CRM': '0001108524',
            'ADBE': '0000796343',
            'ORCL': '0001341439',
            'IBM': '0000051143',
            'INTC': '0000050863',
            'AMD': '0000002488',
            'QCOM': '0000804328'
        }
        
        return symbol_to_cik.get(symbol.upper())
    
    def get_company_relationships(self, symbol: str) -> Dict:
        """
        Get business relationships for a company
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company relationships
        """
        cik = self.get_company_cik(symbol)
        if not cik:
            logger.warning(f"CIK not found for symbol {symbol}")
            return {'relationships': {}, 'error': 'CIK not found'}
        
        # Use mock relationship data (representing real business relationships)
        mock_relationships = self._get_comprehensive_relationships(symbol)
        
        logger.info(f"✅ Extracted relationships for {symbol}")
        return {
            'symbol': symbol,
            'cik': cik,
            'relationships': mock_relationships,
            'filing_date': '2024-08-01',  # Mock recent filing date
            'filing_type': '10-K',
            'source': 'mock_data_for_demo'
        }
    
    def _get_comprehensive_relationships(self, symbol: str) -> Dict[str, List[str]]:
        """
        Comprehensive relationship data based on real business relationships
        This represents what you would extract from actual SEC filings
        """
        relationships_data = {
            'AAPL': {
                'suppliers': [
                    'Taiwan Semiconductor Manufacturing',
                    'Foxconn Technology Group',
                    'Samsung Electronics',
                    'SK Hynix',
                    'Broadcom Inc',
                    'Qualcomm Inc',
                    'Sony Group Corporation'
                ],
                'customers': [
                    'Consumer Electronics Market',
                    'Enterprise Customers',
                    'Education Sector',
                    'Government Agencies'
                ],
                'partners': [
                    'IBM Corporation',
                    'Cisco Systems',
                    'SAP SE',
                    'Microsoft Corporation',
                    'Adobe Inc'
                ],
                'subsidiaries': [
                    'Apple Retail',
                    'FileMaker Inc',
                    'Beats Electronics',
                    'Shazam Entertainment'
                ],
                'competitors': [
                    'Samsung Electronics',
                    'Google LLC',
                    'Microsoft Corporation',
                    'Amazon.com Inc'
                ]
            },
            'MSFT': {
                'suppliers': [
                    'Intel Corporation',
                    'Advanced Micro Devices',
                    'Samsung Electronics',
                    'Taiwan Semiconductor Manufacturing',
                    'Nvidia Corporation'
                ],
                'customers': [
                    'Enterprise Customers',
                    'Government Agencies',
                    'Small Medium Business',
                    'Consumer Market'
                ],
                'partners': [
                    'Dell Technologies',
                    'Hewlett Packard Enterprise',
                    'Lenovo Group',
                    'Accenture',
                    'Ernst & Young'
                ],
                'subsidiaries': [
                    'LinkedIn Corporation',
                    'GitHub Inc',
                    'Activision Blizzard',
                    'Nuance Communications',
                    'Skype Technologies'
                ],
                'competitors': [
                    'Apple Inc',
                    'Google LLC',
                    'Amazon.com Inc',
                    'Oracle Corporation'
                ]
            },
            'GOOGL': {
                'suppliers': [
                    'Samsung Electronics',
                    'LG Electronics',
                    'Foxconn Technology Group',
                    'Taiwan Semiconductor Manufacturing',
                    'Intel Corporation'
                ],
                'customers': [
                    'Digital Advertisers',
                    'Enterprise Customers',
                    'Mobile App Developers',
                    'Content Creators'
                ],
                'partners': [
                    'Samsung Electronics',
                    'LG Electronics',
                    'OnePlus Technology',
                    'Spotify Technology',
                    'Nest Labs'
                ],
                'subsidiaries': [
                    'YouTube LLC',
                    'Android Inc',
                    'Google Cloud',
                    'Waymo LLC',
                    'DeepMind Technologies'
                ],
                'competitors': [
                    'Apple Inc',
                    'Microsoft Corporation',
                    'Amazon.com Inc',
                    'Meta Platforms'
                ]
            },
            'AMZN': {
                'suppliers': [
                    'Intel Corporation',
                    'Advanced Micro Devices',
                    'Nvidia Corporation',
                    'Samsung Electronics',
                    'Broadcom Inc'
                ],
                'customers': [
                    'Individual Consumers',
                    'Enterprise AWS Customers',
                    'Third-party Sellers',
                    'Government Agencies'
                ],
                'partners': [
                    'FedEx Corporation',
                    'United Parcel Service',
                    'Shopify Inc',
                    'Salesforce Inc',
                    'VMware Inc'
                ],
                'subsidiaries': [
                    'Amazon Web Services',
                    'Whole Foods Market',
                    'Twitch Interactive',
                    'Ring Inc',
                    'PillPack Inc'
                ],
                'competitors': [
                    'Microsoft Corporation',
                    'Google LLC',
                    'Walmart Inc',
                    'Alibaba Group'
                ]
            },
            'TSLA': {
                'suppliers': [
                    'Panasonic Corporation',
                    'Contemporary Amperex Technology',
                    'Samsung Electronics',
                    'LG Energy Solution',
                    'Nvidia Corporation'
                ],
                'customers': [
                    'Individual Vehicle Buyers',
                    'Commercial Fleet Customers',
                    'Energy Storage Customers',
                    'Solar Panel Customers'
                ],
                'partners': [
                    'Panasonic Corporation',
                    'Toyota Motor Corporation',
                    'Mercedes-Benz Group',
                    'Hertz Global Holdings'
                ],
                'subsidiaries': [
                    'Tesla Energy',
                    'SolarCity Corporation',
                    'Tesla Insurance Services',
                    'Tesla Semi'
                ],
                'competitors': [
                    'Ford Motor Company',
                    'General Motors Company',
                    'Volkswagen AG',
                    'BYD Company'
                ]
            },
            'META': {
                'suppliers': [
                    'Nvidia Corporation',
                    'Intel Corporation',
                    'Advanced Micro Devices',
                    'Samsung Electronics',
                    'Taiwan Semiconductor Manufacturing'
                ],
                'customers': [
                    'Digital Advertisers',
                    'Small Medium Business',
                    'Enterprise Customers',
                    'Content Creators'
                ],
                'partners': [
                    'Samsung Electronics',
                    'Qualcomm Inc',
                    'Ray-Ban',
                    'Spotify Technology',
                    'Netflix Inc'
                ],
                'subsidiaries': [
                    'Instagram Inc',
                    'WhatsApp Inc',
                    'Oculus VR',
                    'Reality Labs'
                ],
                'competitors': [
                    'Google LLC',
                    'Apple Inc',
                    'TikTok Ltd',
                    'Twitter Inc'
                ]
            },
            'NVDA': {
                'suppliers': [
                    'Taiwan Semiconductor Manufacturing',
                    'Samsung Electronics',
                    'SK Hynix',
                    'Micron Technology',
                    'Advanced Semiconductor Engineering'
                ],
                'customers': [
                    'Data Center Operators',
                    'Gaming Hardware Manufacturers',
                    'Automotive Companies',
                    'Cryptocurrency Miners'
                ],
                'partners': [
                    'Microsoft Corporation',
                    'Google LLC',
                    'Amazon.com Inc',
                    'Tesla Inc',
                    'Mercedes-Benz Group'
                ],
                'subsidiaries': [
                    'Mellanox Technologies',
                    'Arm Holdings',
                    'PhysX Technologies'
                ],
                'competitors': [
                    'Advanced Micro Devices',
                    'Intel Corporation',
                    'Qualcomm Inc',
                    'Broadcom Inc'
                ]
            },
            'NFLX': {
                'suppliers': [
                    'Amazon Web Services',
                    'Google Cloud Platform',
                    'Content Production Studios',
                    'Telecommunications Providers'
                ],
                'customers': [
                    'Individual Subscribers',
                    'Family Subscribers',
                    'International Markets'
                ],
                'partners': [
                    'Comcast Corporation',
                    'Verizon Communications',
                    'AT&T Inc',
                    'Samsung Electronics',
                    'LG Electronics'
                ],
                'subsidiaries': [
                    'Netflix Studios',
                    'Netflix Animation',
                    'Netflix International'
                ],
                'competitors': [
                    'Walt Disney Company',
                    'Amazon Prime Video',
                    'Apple TV+',
                    'HBO Max'
                ]
            }
        }
        
        # Return relationships for the requested symbol, or empty if not found
        return relationships_data.get(symbol.upper(), {
            'suppliers': [],
            'customers': [],
            'partners': [],
            'subsidiaries': [],
            'competitors': []
        })
    
    def batch_get_relationships(self, symbols: List[str]) -> pd.DataFrame:
        """Get relationships for multiple companies"""
        all_relationships = []
        
        logger.info(f"🏢 Getting relationships for {len(symbols)} companies...")
        
        for i, symbol in enumerate(symbols):
            if i > 0:
                time.sleep(self.rate_limit)  # Rate limiting (even for mock data)
            
            if i % 5 == 0:
                logger.info(f"Progress: {i}/{len(symbols)} companies processed")
            
            rel_data = self.get_company_relationships(symbol)
            if rel_data.get('relationships'):
                # Flatten relationship data for DataFrame
                for rel_type, entities in rel_data['relationships'].items():
                    for entity in entities:
                        all_relationships.append({
                            'company_symbol': symbol,
                            'company_cik': rel_data.get('cik', ''),
                            'relationship_type': rel_type,
                            'related_entity': entity,
                            'filing_date': rel_data.get('filing_date', ''),
                            'filing_type': rel_data.get('filing_type', ''),
                            'source': rel_data.get('source', 'mock_data')
                        })
        
        if all_relationships:
            df = pd.DataFrame(all_relationships)
            logger.info(f"✅ Extracted {len(all_relationships)} relationships")
            logger.info(f"Relationship breakdown:")
            for rel_type, count in df['relationship_type'].value_counts().items():
                logger.info(f"  {rel_type}: {count}")
            return df
        else:
            logger.warning("❌ No relationships extracted")
            return pd.DataFrame()

# Test function
def test_sec_collector():
    """Test SEC collector with fixed mock data approach"""
    collector = SECCollector()
    
    # Test CIK lookup
    print("🧪 Testing CIK lookup...")
    aapl_cik = collector.get_company_cik('AAPL')
    if aapl_cik:
        print(f"✅ AAPL CIK: {aapl_cik}")
    
    # Test relationship extraction
    print("\n🧪 Testing relationship extraction...")
    relationships = collector.get_company_relationships('AAPL')
    if relationships.get('relationships'):
        print(f"✅ AAPL relationships found:")
        for rel_type, entities in relationships['relationships'].items():
            print(f"  {rel_type}: {len(entities)} entities")
            if entities:  # Show first few examples
                print(f"    Examples: {', '.join(entities[:3])}")
    
    # Test batch processing
    print("\n🧪 Testing batch relationship extraction...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    rel_df = collector.batch_get_relationships(test_symbols)
    if not rel_df.empty:
        print(f"✅ Relationship data: {len(rel_df)} total relationships")
        print(f"Companies: {rel_df['company_symbol'].nunique()}")
        print(f"Relationship types: {list(rel_df['relationship_type'].unique())}")
        print(f"\nSample relationships:")
        print(rel_df[['company_symbol', 'relationship_type', 'related_entity']].head(10))

if __name__ == "__main__":
    test_sec_collector()