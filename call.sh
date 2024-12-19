curl http://localhost:11434/api/generate -d '{
  "model": "gemma2:9b",
  "prompt": "The construct “perceived informativeness” summarizes the following five items when a customer evaluates an app:\n1. The app showed the information I expected.\n2. The app provides detailed information.\n3. The app provides complete information.\n4. The app provides information that helps.\n5. The app provides information for comparisons.\n\nAvailable responses to the items and the construct are “[1,2,3,4,5]” on 5-point Likert scales.\nThese correspond to “1 (strongly disagree)”, “2 (disagree)”, “3 (don’t know, neutral)”, “4 (agree)”, and “5 (disagree)\n\nFor the following comment: How would the author evaluate the construct “perceived informativeness”?\n\nReview: ```Einfach spitze```\nProvide your response in a JSON format containing a single key `label`, do not provide more information",
  "stream": false
}'
curl http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "
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
",
  "stream": false
}'


curl -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
  "model": "llama3.1",
  "messages": [{"role": "user", "content": "Tell me about Canada."}],
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string"
      },
      "capital": {
        "type": "string"
      },
      "languages": {
        "type": "array",
        "items": {
          "type": "string"
        }
      }
    },
    "required": [
      "name",
      "capital", 
      "languages"
    ]
  }
}'