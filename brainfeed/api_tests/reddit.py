import praw
from pprint import pprint
import pandas as pd

# Initialize reddit instance
reddit = praw.Reddit("brainfeed_script")
print(reddit.config.user_agent)
print(reddit.config.username)

# Get submissions with a certain keyword and store the information in a dataframe
posts_brain = []
subreddits = reddit.subreddit("science")
for submission in subreddits.search("brain"):
    posts_brain.append([submission.title, submission.score, submission.id, submission.subreddit, submission.url, submission.num_comments, submission.selftext, submission.created_utc])
posts_brain = pd.DataFrame(posts_brain, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created_time'])
posts_brain["created_time"] = pd.to_datetime(posts_brain['created_time'], unit='s') # convert the UTC timestamp to readable date
posts_brain.head()

# Get existing submissions
# this is impossible now as Subreddit.submissions() has been removed 

# Get new submissions and store the information in a dataframe
submissions = []
subreddits = reddit.subreddit("science")
for submission in subreddits.stream.submissions():
    submissions.append([submission.title, submission.score, submission.id, submission.subreddit, submission.url, submission.num_comments, submission.selftext, submission.created_utc])
submissions = pd.DataFrame(submissions, columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created_time'])
submissions.head()

# # Stream subreddit submissions
# subreddits = reddit.subreddit("science")
# for submission in subreddits.stream.submissions():
#     print(submission.title)
#     print()

# Testing
submission.url
subcomments = submission.comments
subcomments[0].body
len(subcomments)
