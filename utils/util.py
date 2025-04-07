"""
Utility functions for easier analysis
"""
import polars as pl
import numpy as np

from sklearn.preprocessing import LabelEncoder

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
  # Change all column names to lowercase
  df = df.rename(lambda colname: colname.lower())

  # Convert episode sentiment
  # -1: Negative, 0: Neutral, 1: Positive
  sentiment_map = {
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
  }
  # Convert days and times
  day_mapper = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
  }
  time_mapper = {
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3,
    'Night': 4
  }

  # Replace >12 ads with median value
  # Replace with median because we want to exclude these outlier values
  median_ads = df.select(pl.median('number_of_ads')).item()

  df = df.with_columns(
    pl.col('episode_sentiment').replace(sentiment_map).cast(pl.Int64),
    pl.when(pl.col('number_of_ads') > 12)
      .then(median_ads)
      .otherwise('number_of_ads')
      .alias('number_of_ads'),

    # Fill null values on host andguest_popularity_percentage and episode_length_minutes
    # with respective podcast names mean value
    pl.col([
      'episode_length_minutes',
      'guest_popularity_percentage',
      'host_popularity_percentage'
    ]).fill_null(
      pl.col([
        'episode_length_minutes',
        'guest_popularity_percentage',
        'host_popularity_percentage'
      ]).mean().over('podcast_name')
    ),

    # Replace publication day and time with cyclical values
    pl.col('publication_day')
      .replace(day_mapper)
      .cast(pl.Int64)
      .map_batches(np.sin)
      .name.suffix('_sin'),
    pl.col('publication_day')
      .replace(day_mapper)
      .cast(pl.Int64)
      .map_batches(np.cos)
      .name.suffix('_cos'),
    pl.col('publication_time')
      .replace(time_mapper)
      .cast(pl.Int64)
      .map_batches(np.sin)
      .name.suffix('_sin'),
    pl.col('publication_time')
      .replace(time_mapper)
      .cast(pl.Int64)
      .map_batches(np.cos)
      .name.suffix('_cos'),
    
    # Remove Episode from episode_title and convert to integer
    pl.col('episode_title').str.replace(r"Episode ", "").cast(pl.Int64)
  )

  # Label encoded dataframe
  ## Technically this needs to separate between train and test
  enc = LabelEncoder()
  df = df.with_columns(
    pl.col('podcast_name').map_batches(enc.fit_transform),
    pl.col('genre').map_batches(enc.fit_transform),
  )

  df = df.to_dummies(
    columns=['podcast_name', 'genre']
  )

  df = df.drop(['publication_day', 'publication_time'])

  return df