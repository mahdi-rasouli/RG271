import pandas as pd
from snscrape.modules import twitter

from .base import Scraper


class TwitterScraper(Scraper):
    """Scrapes mentions and hashtags specific to a user/username on Twitter."""

    def __init__(self, username, max_date=None, min_date=None):
        super().__init__(username, max_date=max_date, min_date=min_date)

    def scrape(self):
        """Entrypoint for this class - constructs and returns a DataFrame containing all hashtag
        and mentions related to this user.

        Returns:
            pd.DataFrame: Pandas DataFrame with the columns date, content, content_type, user, url.
        """

        return pd.concat(
            [self._scrape_mentions(), self._scrape_hashtags()], ignore_index=True, sort=False
        )

    def _scrape_tweets(self, iterable, content_type):
        """Loop a generator function to extract tweet data.

        Args:
            iterable (generator): Generator function to iterate over.
            content_type (str): Content type, e.g. hashtag or mention.

        Returns:
            pd.DataFrame: DataFrame with data about each tweet - columns are date, content,
            content_type, user, and url.
        """

        results = []
        for tweet in iterable:
            # these are sorted newest to oldest, so we can stop searching once we hit the min_date
            if self.min_date is not None and tweet.date.date() < self.min_date:
                break
            # ignore tweets newer than our max_date or replies from the user
            elif tweet.date.date() > self.max_date or tweet.username == self.username:
                continue

            results.append(
                {
                    "date": tweet.date,
                    "content": tweet.content,
                    "content_type": f"twitter/{content_type}",
                    "user": tweet.username,
                    "url": tweet.url,
                }
            )

        return pd.DataFrame(results)

    def _scrape_mentions(self):
        """Helper method for calling _scrape_tweets() to search for user mentions (@user)."""
        return self._scrape_tweets(
            twitter.TwitterSearchScraper(self.username).get_items(), "mention"
        )

    def _scrape_hashtags(self):
        """Helper method for calling _scrape_tweets() to search for user hashtags (#user)."""
        return self._scrape_tweets(
            twitter.TwitterHashtagScraper(self.username).get_items(), "hashtag"
        )
