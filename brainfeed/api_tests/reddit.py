import praw
from pprint import pprint

# Initialize reddit instance
reddit = praw.Reddit("brainfeed_script")
reddit.config.user_agent
reddit.config.username

# Stream subreddit submissions
subreddits = reddit.subreddit("science")
for submission in subreddits.stream.submissions():
    print(submission.title)

# Testing
submission.url
subcomments = submission.comments
subcomments[0].body
len(subcomments)
