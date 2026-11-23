# Contributing to Knowledge Conflict Pilot

Thank you for contributing!

## Ways to contribute

- Improve documentation
- Add or correct English, Bangla, or Banglish questions
- Improve the model notebooks
- Report incorrect answers or notebook errors

## Question files

The question lists are in:

- `data/questions_english.json`
- `data/questions_bangla.json`
- `data/questions_banglish.json`

Each entry has an `id` and a `question`, for example:

    {
      "id": 51,
      "question": "Your question here?"
    }

When adding a question, use the next available ID and keep the JSON valid. If appropriate, update the matching question in all three language files.

## Report a problem

Please include the file name, question ID, what you expected, and what happened. A screenshot is helpful but optional.

## Check JSON files

Run these commands in the terminal:

    python -m json.tool data/questions_english.json
    python -m json.tool data/questions_bangla.json
    python -m json.tool data/questions_banglish.json

## Submit a change

Create a branch, make your change, and open a pull request to the project's `main` branch. Describe what you changed so others can review it.
