Okay, lets think through it.
RAG works perfectly fine. But it is a single rquest-response model (single-hop).
DSP will help in getting it to a multi request-response model (multi-hop).

To make it multi-hop, we need to know a) where to hop (what to ask, basically) and b) when to stop.

b] When to stop
  - We can prompt the model to think.
  - Give it the current context, main compound question to answer, and then prompt it to think.
  - The prompt would be something like, "Given the context, do you think you have gathered enough information to answer the question? Reply with in JSON with the following format: "is_enough_info": true/false "

a] Where to hop/What's the subquestion that will help gather more information needed to answer the question.
  - Here, again we are in the thinking mode.
  - Prompt could be something like, "Given the context, what is the next question you would like to ask to gather more relevant information that is needed to answer the final question? Reply in JSON with the following format: "next_question": "<subquestion>" "