from abc import ABC, abstractmethod
from datetime import datetime


class Scraper(ABC):
    """Base class for a social media Scraper."""

    def __init__(self, username, max_date=None, min_date=None):
        self._max_date = (
            self._parse_date(max_date) if max_date is not None else datetime.now().date()
        )
        self._min_date = self._parse_date(min_date) if min_date is not None else None
        self._username = username

    def _parse_date(self, dt):
        """Parses a date and converts it to a datetime object, if it is not one already. The string
        MUST follow ISO 8601 format YYYY-MM-DD, or an exception will be thrown.

        Args:
            dt (str or datetime): Argument to convert to a date.

        Raises:
            ValueError: If the argument is not a str or datetime object.

        Returns:
            datetime.date: Date representation of a datetime object.
        """

        if isinstance(dt, str):
            return datetime.fromisoformat(dt).date()
        elif isinstance(dt, datetime):
            return dt.date()

        raise ValueError(f"Date must be either an ISO formatted string or datetime object.")

    @abstractmethod
    def scrape(self):
        """Scrape a social media platform and return results as a Pandas DataFrame with the columns:
            - date
            - content
            - content_type
            - user
            - url

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """

        raise NotImplementedError()

    @property
    def max_date(self):
        """Getter for attribute max_date."""
        return self._max_date

    @property
    def min_date(self):
        """Getter for attribute min_date."""
        return self._min_date

    @property
    def username(self):
        """Getter for attribute username."""
        return self._username
