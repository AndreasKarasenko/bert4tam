PE = """
The construct “perceived enjoyment” reflects the following four items when evaluating an app on 5-point Likert scales:
1. Using the app is fun.
2. The app contains nice gimmicks as functions.
3. It is fun to discover the functions of the app.
4. The app invites you to discover more functions.

Available responses to the items and the construct are “{labels}” on 5-point Likert scales.
These correspond to “1 (strongly disagree)”, “2 (disagree)”, “3 (don’t know, neutral)”, “4 (agree)”, and “5 (disagree)

For the following comment: How would the author evaluate the construct “perceived enjoyment”?

Review: ```{x}```
Provide your response in a JSON format containing a single key `label`
"""
PI = """
The construct “perceived informativeness” summarizes the following five items when a customer evaluates an app:
1. The app showed the information I expected.
2. The app provides detailed information.
3. The app provides complete information.
4. The app provides information that helps.
5. The app provides information for comparisons.

Available responses to the items and the construct are “[1,2,3,4,5]” on 5-point Likert scales.
These correspond to “1 (strongly disagree)”, “2 (disagree)”, “3 (don’t know, neutral)”, “4 (agree)”, and “5 (disagree)

For the following comment: How would the author evaluate the construct “perceived informativeness”?

Review: ```Einfach spitze```
Provide your response in a JSON format containing a single key `label`
Do not provide any other information.
"""