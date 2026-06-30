from typing import List, Dict

# Hand-crafted test questions for the Flask repo
# Format: question + ideal answer keywords for validation
FLASK_TEST_DATASET: List[Dict] = [
    {
        "question": "How does Flask handle URL routing?",
        "ground_truth": (
            "Flask handles URL routing through the route decorator "
            "which maps URL rules to view functions using add_url_rule method."
        ),
    },
    {
        "question": "What is the login_required decorator and how does it work?",
        "ground_truth": (
            "login_required is a view decorator that redirects anonymous "
            "users to the login page by checking if g.user is None."
        ),
    },
    {
        "question": "How do Flask blueprints work?",
        "ground_truth": (
            "Flask blueprints are a way to organize routes and functions "
            "into reusable components that can be registered on an application."
        ),
    },
    {
        "question": "How does Flask handle application context?",
        "ground_truth": (
            "Flask uses application context to store data during a request "
            "using the g object and push/pop context methods."
        ),
    },
    {
        "question": "What is the Flask test client used for?",
        "ground_truth": (
            "The Flask test client simulates HTTP requests to the application "
            "for testing purposes without running a real server."
        ),
    },
]