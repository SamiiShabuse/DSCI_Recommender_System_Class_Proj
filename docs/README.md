# Creator Intelligence Recommender System

This folder collects the project brief, design notes, and implementation decisions for the Instagram influencer recommender system.

## Project Summary

We are building a recommender system that helps Instagram influencers decide what kind of content to create next. The system will use influencer metadata and post metadata to recommend content strategies or post types based on engagement patterns.

## Initial Scope

The first version should focus on metadata rather than raw images because the dataset is large and image processing would add a lot of complexity too early.

Primary inputs:

- Influencer profile data
- Post captions
- Hashtags
- Likes and comments
- Timestamp and sponsorship information
- Tagged users

Primary outputs:

- Ranked post type or content strategy recommendations
- Comparisons between similar influencers
- Simple engagement-based scoring for candidate recommendations

## Proposed Approach

1. Content-based recommendations using post text and engagement signals.
2. Collaborative filtering using behavior patterns across influencers.
3. A hybrid approach that combines both methods.

## Slow Build Plan

Phase 1:

- Confirm the exact dataset files we will use.
- Define the smallest useful problem statement.
- Set up folders, environment, and data-loading scripts.

Phase 2:

- Load and inspect the metadata.
- Clean and standardize fields.
- Create a small exploratory analysis.

Phase 3:

- Build a simple baseline recommender.
- Evaluate with engagement-oriented metrics.

Phase 4:

- Add collaborative filtering.
- Combine methods into a hybrid recommender.

## Open Questions

- Which exact metadata files are available to us locally?
- Do we want to predict content type, caption style, or both?
- What is the smallest evaluation set we can build first?
