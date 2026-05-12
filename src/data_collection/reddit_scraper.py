import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class RedditScraper:
    """
    Web scraper for Reddit stock discussions without API access.
    Scrapes r/wallstreetbets, r/stocks, r/investing for ticker mentions.
    """
    
    def __init__(self, user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'):
        self.base_url = 'https://old.reddit.com'
        self.headers = {
            'User-Agent': user_agent
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cache = {}  # Simple cache to avoid duplicate scraping
    
    def scrape_subreddit(self, subreddit: str, ticker: str, limit: int = 50) -> pd.DataFrame:
        """
        Scrape a subreddit for posts mentioning a specific ticker.
        
        Args:
            subreddit: Subreddit name (e.g., 'wallstreetbets')
            ticker: Stock ticker to search for
            limit: Maximum number of posts to scrape
            
        Returns:
            DataFrame with post data
        """
        cache_key = f"{subreddit}_{ticker}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            logger.info(f"Using cached data for {subreddit}/{ticker}")
            return self.cache[cache_key]
        
        posts = []
        
        try:
            # Search for ticker in subreddit
            search_url = f"{self.base_url}/r/{subreddit}/search"
            params = {
                'q': ticker,
                'restrict_sr': 'on',
                'sort': 'new',
                'limit': limit
            }
            
            logger.info(f"Scraping r/{subreddit} for ${ticker}...")
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Find all post containers
            post_containers = soup.find_all('div', class_='thing')
            
            for post in post_containers[:limit]:
                try:
                    post_data = self._extract_post_data(post, ticker)
                    if post_data:
                        posts.append(post_data)
                except Exception as e:
                    logger.debug(f"Error extracting post: {e}")
                    continue
            
            logger.info(f"Scraped {len(posts)} posts from r/{subreddit}")
            
        except requests.RequestException as e:
            logger.error(f"Error scraping r/{subreddit}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        
        df = pd.DataFrame(posts)
        self.cache[cache_key] = df
        return df
    
    def _extract_post_data(self, post_element, ticker: str) -> Optional[Dict]:
        """Extract data from a post element."""
        try:
            # Extract post ID
            post_id = post_element.get('data-fullname', '')
            
            # Extract title
            title_elem = post_element.find('a', class_='title')
            if not title_elem:
                return None
            title = title_elem.get_text(strip=True)
            
            # Check if ticker is mentioned in title (case insensitive)
            if not re.search(rf'\b{ticker}\b', title, re.IGNORECASE):
                return None
            
            # Extract score
            score_elem = post_element.find('div', class_='score unvoted')
            if not score_elem:
                score_elem = post_element.find('div', class_='score')
            score = 0
            if score_elem:
                score_text = score_elem.get_text(strip=True)
                try:
                    score = int(score_text) if score_text.isdigit() else 0
                except:
                    score = 0
            
            # Extract number of comments
            comments_elem = post_element.find('a', class_='comments')
            num_comments = 0
            if comments_elem:
                comments_text = comments_elem.get_text(strip=True)
                match = re.search(r'(\d+)', comments_text)
                if match:
                    num_comments = int(match.group(1))
            
            # Extract timestamp
            time_elem = post_element.find('time')
            created_utc = datetime.now()
            if time_elem and time_elem.get('datetime'):
                try:
                    created_utc = datetime.fromisoformat(time_elem['datetime'].replace('Z', '+00:00'))
                except:
                    pass
            
            # Extract author
            author_elem = post_element.find('a', class_='author')
            author = author_elem.get_text(strip=True) if author_elem else 'unknown'
            
            # Extract subreddit
            subreddit_elem = post_element.find('a', class_='subreddit')
            subreddit = subreddit_elem.get_text(strip=True) if subreddit_elem else 'unknown'
            
            # Extract URL
            url = title_elem.get('href', '')
            if url.startswith('/r/'):
                url = self.base_url + url
            
            return {
                'id': post_id,
                'title': title,
                'score': score,
                'num_comments': num_comments,
                'created_utc': created_utc,
                'author': author,
                'subreddit': subreddit,
                'url': url,
                'ticker': ticker,
                'source': 'reddit'
            }
            
        except Exception as e:
            logger.debug(f"Error extracting post data: {e}")
            return None
    
    def scrape_wallstreetbets(self, ticker: str, limit: int = 50) -> pd.DataFrame:
        """Convenience method to scrape r/wallstreetbets."""
        return self.scrape_subreddit('wallstreetbets', ticker, limit)
    
    def scrape_multiple_subreddits(self, ticker: str, 
                                   subreddits: List[str] = None,
                                   limit_per_sub: int = 25) -> pd.DataFrame:
        """
        Scrape multiple subreddits for a ticker.
        
        Args:
            ticker: Stock ticker
            subreddits: List of subreddit names
            limit_per_sub: Posts to scrape per subreddit
            
        Returns:
            Combined DataFrame from all subreddits
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        
        all_posts = []
        
        for subreddit in subreddits:
            df = self.scrape_subreddit(subreddit, ticker, limit_per_sub)
            if not df.empty:
                all_posts.append(df)
            time.sleep(2)  # Rate limiting
        
        if not all_posts:
            return pd.DataFrame()
        
        combined = pd.concat(all_posts, ignore_index=True)
        combined = combined.drop_duplicates(subset=['id'])
        combined = combined.sort_values('created_utc', ascending=False)
        
        logger.info(f"Total posts for ${ticker}: {len(combined)}")
        return combined
    
    def calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic sentiment features from scraped data.
        
        Args:
            df: DataFrame with Reddit posts
            
        Returns:
            DataFrame with sentiment features
        """
        if df.empty:
            return df
        
        # Calculate engagement score
        df['engagement_score'] = df['score'] + (df['num_comments'] * 2)
        
        # Calculate post velocity (posts per hour)
        if len(df) > 1:
            time_span = (df['created_utc'].max() - df['created_utc'].min()).total_seconds() / 3600
            df['post_velocity'] = len(df) / max(time_span, 1)
        else:
            df['post_velocity'] = 0
        
        # Detect bullish/bearish keywords in titles
        bullish_keywords = ['moon', 'rocket', 'buy', 'calls', 'bullish', 'squeeze', 'pump']
        bearish_keywords = ['crash', 'dump', 'puts', 'bearish', 'short', 'sell']
        
        df['bullish_mentions'] = df['title'].apply(
            lambda x: sum(1 for word in bullish_keywords if word in x.lower())
        )
        df['bearish_mentions'] = df['title'].apply(
            lambda x: sum(1 for word in bearish_keywords if word in x.lower())
        )
        
        # Simple sentiment score
        df['sentiment_score'] = df['bullish_mentions'] - df['bearish_mentions']
        
        return df


if __name__ == '__main__':
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    
    scraper = RedditScraper()
    
    # Test scraping
    ticker = 'GME'
    df = scraper.scrape_multiple_subreddits(ticker, limit_per_sub=10)
    
    if not df.empty:
        print(f"\nScraped {len(df)} posts for ${ticker}")
        print("\nSample posts:")
        print(df[['title', 'score', 'num_comments', 'subreddit']].head())
        
        # Calculate sentiment
        df = scraper.calculate_sentiment_features(df)
        print(f"\nAverage sentiment score: {df['sentiment_score'].mean():.2f}")
        print(f"Total engagement: {df['engagement_score'].sum()}")
    else:
        print(f"No posts found for ${ticker}")
