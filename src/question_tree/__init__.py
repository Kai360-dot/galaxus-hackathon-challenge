"""Zero-shot image labeling with Apertus v1.5, built from small questions.

Apertus answers many short questions about one image instead of choosing all six labels at once. A decision tree
turns the answers into the labels of five dimensions. Two parts are measured in pixels instead of asked: whether
the backdrop is white, and from where the product is seen.

    config      paths, classes and settings
    data        train table, image sets and the scoring frame
    images      image preparation, the same for every request
    client      the Apertus chat endpoint with retries
    questions   the wording of every question
    ask         asking the questions and storing the answers
    pixels      white backdrop, measured in the image corners
    tree        the decision tree
    contours    perspective from silhouette and edge features
    combiner    a small learned model as an alternative to the hand-written tree
    evaluate    scoring with the leaderboard metric
    experiments the strategies that were tested and dropped
    predict     the whole pipeline for a csv of images
"""
