initial_prompt = '''I will provide you with a question, and you need to provide me with a subquestion, whose answer will help me gather information to answer the main question.
- It is an iterative process, so only give me one subquestion at a time.
- DO NOT GIVE ME "AND" or "OR" queries.
- The subquestion should be a simple factoid question. 
- Before giving the subquestion, please read through the context, if provided and reason through the question step-by-step in order to produce the subquestion.
- Mention the answer to the subquestion, if you think you have enough context.

---
<Example 1>

Context: 
    - [1]:
      - subquestion: "What is the name of the castle that David Gregory inherited?"
      - answer: "Kinnairdy Castle"
"Question": "How many storeys are in the castle that David Gregory inherited?
"Reasoning": "Let's think step by step in order to produce the query. Based on the context we know that David Gregory inherited Kinnairdy Castle, so we need to find information about the castle and its characteristics."
"answer": "N/A"
"factoid_subquestion": "How many storeys did Kinnairdy Castle have?"

---
<Example 2>

"Context": "N/A",
"Question": "In what year was the club founded that played Manchester City in the 1972 FA Charity Shield",
"Reasoning": "Take a deep breath and let's think through the question step-by-step in order to produce the query. We know that the FA Charity Shield is an annual football match played in England between the winners of the previous season's Premier League and FA Cup. In this case, we are looking for the year when Manchester City played against a specific club in the 1972 FA Charity Shield. To find this information, we can search for the history of the FA Charity Shield and the teams that participated in the 1972 edition.",
"answer": "N/A"
"factoid_subquestion": "Which teams participated in the 1972 FA Charity Shield?"

---
<Example 3>

"Context": "N/A",
"Question": "Which is taller, the Empire State Building or the Bank of America Tower?",
"Reasoning": "Take a deep breath and let's think through the question step-by-step in order to produce the query. We need to find the heights of both buildings and compare them.",
"answer": "N/A"
"factoid_subquestion": "What is the height of Empire State Building?"


---
<Example 4>

"Context":
  - [1]:
    - subquestion: "What is the birthdate of Aleksandr Danilovich Aleksandrov?"
    - answer: "August 4, 1912"
  - [2]:
    - subquestion: "What is the birthdate of Anatoly Fomenko?"
    - answer: "March 13, 1945"
"Question": "Who is older, Aleksandr Danilovich Aleksandrov or Anatoly Fomenko?",
"Reasoning": "Take a deep breath and let's think through the question step-by-step in order to produce the query. Based on the context, we know that Aleksandr Danilovich Aleksandrov was born on August 4, 1912, and Anatoly Fomenko was born on March 13, 1945. Therefore, Aleksandr Danilovich Aleksandrov is older."
"answer": "Aleksandr Danilovich Aleksandrov was born on August 4, 1912, and Anatoly Fomenko was born on March 13, 1945. Therefore, Aleksandr Danilovich Aleksandrov is older."
"factoid_subquestion": "N/A"


---
<Actual Query>

"Context": "{context}",
"Question": "{query}",
"Reasoning": "Take a deep breath and let's think through the question step-by-step in order to produce the query...",

---
Return ONLY THE JSON object as the response, with the following keys:
- Reasoning
- answer
- factoid_subquestion
'''

### Prompt for second iteration onwards
is_enough_info_prompt = '''Given the context and original question, do you think you have gathered enough information to answer the question?
Reply in JSON with the following format:
- "is_enough_info": true/false

---
<Actual Query>

context: {context}
original_question: {original_question}
'''

### Prompt for subquestion
subquestion_prompt = '''Given the context, what is the next question you would like to ask to gather more relevant information that is needed to answer the final question?
- It is an iterative process, so only give me one subquestion at a time.
- DO NOT GIVE ME "AND" or "OR" queries.
- The subquestion should be a simple factoid question. 
- Before giving the subquestion, please read through the context, if provided and reason through the question step-by-step in order to produce the subquestion.

Reply in JSON with the following format:
- reasoning
- next_question

---
<Example 1>

context: N/A
original_question: In what year was the club founded that played Manchester City in the 1972 FA Charity Shield
reasoning: Take a deep breath and let's think through the question step-by-step in order to produce the query. We know that the FA Charity Shield is an annual football match played in England between the winners of the previous season's Premier League and FA Cup. In this case, we are looking for the year when Manchester City played against a specific club in the 1972 FA Charity Shield. To find this information, we can search for the history of the FA Charity Shield and the teams that participated in the 1972 edition.
next_question: Which teams participated in the 1972 FA Charity Shield?
---
<Example 2>

context:
  - [1]:
    - subquestion: What is the name of the castle that David Gregory inherited?
    - answer: Kinnairdy Castle
original_question: How many storeys are in the castle that David Gregory inherited?
reasoning: Let's think step by step in order to produce the query. Based on the context we know that David Gregory inherited Kinnairdy Castle, so we need to find information about the castle and its characteristics.
next_question: How many storeys did Kinnairdy Castle have?
---
<Example 3>

context:
  - [1]:
    - subquestion: What is the birthdate of Aleksandr Danilovich Aleksandrov?
    - answer: August 4, 1912
original_question: Who is older, Aleksandr Danilovich Aleksandrov or Anatoly Fomenko?
reasoning: Take a deep breath and let's think through the question step-by-step in order to produce the query. Based on the context, we know that Aleksandr Danilovich Aleksandrov was born on August 4, 1912, but we still need to find the birthdate of Anatoly Fomenko.
next_question: What is the birthdate of Anatoly Fomenko?
---
<Actual Query>

context: {context}
original_question: {original_question}
'''

final_answer_prompt = '''Given the context, what is the answer to the original question?
Reply in JSON with the following format:
- answer: <answer>
---
context: {context}
original_question: {original_question}
'''

'''
---
<Example 1>

context: N/A,
original_question: In what year was the club founded that played Manchester City in the 1972 FA Charity Shield
is_enough_info: false
---
<Example 2>

context:
  - [1]:
    - subquestion: What is the name of the castle that David Gregory inherited?
    - answer: Kinnairdy Castle
original_question: How many storeys are in the castle that David Gregory inherited?
is_enough_info: false
---
<Example 3>

context:
  - [1]:
    - subquestion: What is the name of the castle that David Gregory inherited?
    - answer: Kinnairdy Castle
  - [2]:
    - subquestion: How many storeys did Kinnairdy Castle have?
    - answer: 5
original_question: How many storeys are in the castle that David Gregory inherited?
is_enough_info: true

---
<Example 4>

context:
  - [1]:
    - subquestion: What is the birthdate of Aleksandr Danilovich Aleksandrov?
    - answer: August 4, 1912
  - [2]:
    - subquestion: What is the birthdate of Anatoly Fomenko?
    - answer: March 13, 1945
original_question: Who is older, Aleksandr Danilovich Aleksandrov or Anatoly Fomenko?
is_enough_info: true'''