from datasets import load_dataset
import torch
import transformers
from evaluate import load
import numpy as np
from openai import OpenAI
import re

dataset = load_dataset("commonsense_qa")

question = dataset["train"][1]["question"]
answer = dataset["train"][1]["choices"]

answer_key_lines = []
for label, text in zip(answer["label"], answer["text"]):
    answer_key_lines.append(f"{label}. {text}")

answer_key = "\n".join(answer_key_lines)
full_question = f"""Question: {question} \n Answers: {answer_key}"""

client = OpenAI()


class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

    def run(self, input_state):
        prompt = self.construct_prompt(input_state)

        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            instructions=prompt,
            input=input_state,
        )

        return self.parse_output(response.output_text)

    def construct_prompt(self, input_state):
        return f"System Prompt:\n{self.system_prompt}\n\nTask Info:\n{input_state}\nInstructions: Respond appropriately."

    def parse_output(self, response):
        if self.name == "Planner":
            return {"plan": response}

        elif self.name == "Solver":
            # robust regex: captures single uppercase letter after 'Answer:' ignoring whitespace
            match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
            return {"solver_text": response, "final_answer": match.group(1).upper() if match else None}

        elif self.name == "Critic":
            # robust regex: captures confidence (1-5) and optional suggested answer
            conf_match = re.search(r'Confidence:\s*(\d)', response)
            ans_match = re.search(r'Suggested Answer:\s*([A-Z])', response)
            return {
                "confidence": int(conf_match.group(1)) if conf_match else None,
                "suggested_answer": ans_match.group(1).upper() if ans_match else None,
                "critic_reasoning": response
            }


planner_prompt = """You are the Planner agent. Analyze the question and identify what is being asked. 
Focus on:
1. The type of question (e.g., fill-in-the-blank, cause-effect, etc.)
2. The key entities and relationships
3. Any constraints or hints in the wording

Do NOT guess the answer. Output your reasoning in a clear, structured way that can be passed to the Solver.
"""

solver_prompt_with_planner = """
You are the Solver agent. Using the Planner's reasoning, choose the correct multiple choice answer.
Output Format: 
 - One or more lines of reasoning connected with the Planner's reasoning
 - One line: Answer: X
...
"""

solver_prompt_no_planner = """
You are the Solver agent. The Planner is unavailable.
Reason directly from the question and answer choices.
Output Format: 
 - One or more lines of reasoning.
 - One line: Answer: X
...
"""

critic_prompt = """You are the Critic agent. Review the Planner and Solver outputs.

First, provide a short explanation of whether you agree with the Solver's answer or not, and why.
Then:
1. Rate your confidence that the Solver's answer is correct on a scale 1–5 (Confidence: X).
2. If confidence <=2, suggest an alternative multiple choice answer (Suggested Answer: Y).

Output format:
- One or more lines of explanation.
- One line: Confidence: X
- Optional line: Suggested Answer: Y
"""

critic_prompt_no_planner = """
You are the Critic agent. The Planner is unavailable.

Review only the Solver’s reasoning and final answer.
Evaluate whether the answer logically follows from the question.

Then:
- Confidence: X (1–5)
- Suggested Answer: Y (only if confidence <=2)
"""


def orchestrator(input_question, input_question_answer, usePlanner, useCritic):
    planner_agent = Agent("Planner", planner_prompt)
    solver_agent = Agent("Solver", solver_prompt_with_planner)
    critic_agent = Agent("Critic", critic_prompt)

    log_file = "shapley_orchestrator.txt"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("NEW QUESTION\n")
        f.write("=" * 80 + "\n\n")
        f.write(input_question_answer + "\n\n")

        if usePlanner:
            solver_agent.system_prompt = solver_prompt_with_planner
            critic_agent.system_prompt = critic_prompt
            planner_output = planner_agent.run(input_question)
            plan = planner_output["plan"]
            f.write("PLANNER OUTPUT:\n")
            f.write("-" * 40 + "\n")
            f.write(plan + "\n\n")
        else:
            solver_agent.system_prompt = solver_prompt_no_planner
            critic_agent.system_prompt = critic_prompt_no_planner

        if usePlanner:
            solver_input = f"""Question: {input_question_answer}. Planner Output: {plan}"""
        else: 
            solver_input = f"""Question: {input_question_answer}. """

        solver_output = solver_agent.run(solver_input)

        solver_reasoning = solver_output["solver_text"]
        solver_answer = solver_output["final_answer"]

        f.write("SOLVER REASONING:\n")
        f.write("-" * 40 + "\n")
        f.write(solver_reasoning + "\n\n")

        f.write("SOLVER ANSWER:\n")
        f.write("-" * 40 + "\n")
        f.write(str(solver_answer) + "\n\n")

        if useCritic:
            if usePlanner:
                critic_string = f"""Question: {input_question_answer}. Planner Output: {plan}. Solver Reasoning {solver_reasoning}. Solver Final Answer: {solver_answer}"""
            else:
                critic_string = f"""Question: {input_question_answer}. Solver Reasoning {solver_reasoning}. Solver Final Answer: {solver_answer}"""
            critic_output = critic_agent.run(critic_string)

            f.write("CRITIC REASONING:\n")
            f.write("-" * 40 + "\n")
            f.write(critic_output["critic_reasoning"] + "\n\n")

            f.write("CRITIC CONFIDENCE:\n")
            f.write("-" * 40 + "\n")
            f.write(str(critic_output["confidence"]) + "\n\n")

            f.write("CRITIC SUGGESTED ANSWER:\n")
            f.write("-" * 40 + "\n")
            f.write(str(critic_output["suggested_answer"]) + "\n\n")

    if useCritic and critic_output["confidence"] is not None and critic_output["confidence"] <= 2:
        final_answer = critic_output.get("suggested_answer", solver_answer)
    else:
        final_answer = solver_answer

    return final_answer


coalitions = {
    "S": {"usePlanner": False, "useCritic": False},
    "P": {"usePlanner": True, "useCritic": False},
    "C": {"usePlanner": False, "useCritic": True},
    "PC": {"usePlanner": True, "useCritic": True}
}

scores = {}

for key in coalitions:
    results = []
    answers = []

    for i in range(25): 
        q = dataset["train"][i]["question"]
        answers.append(dataset["train"][i]["answerKey"])
        choices = dataset["train"][i]["choices"]

        answer_key = "\n".join(
            f"{l}. {t}" for l, t in zip(choices["label"], choices["text"])
        )

        full_question = f"Question: {q}\nAnswers:\n{answer_key}"

        pred = orchestrator(
            q,
            full_question,
            coalitions[key]["usePlanner"],
            coalitions[key]["useCritic"]
        )

        results.append(pred)

    similarity = sum(x == y for x, y in zip(results, answers)) / len(answers)
    scores[key] = similarity
    print("Accuracy: ", similarity*100, "%")

phi_planner = 0.5*((scores["P"]-scores["S"]) + (scores["PC"]-scores["C"]))
phi_critic = 0.5*((scores["C"]-scores["S"]) + (scores["PC"] - scores["P"]))

print("PLANNER SCORE: ", phi_planner)
print("CRITIC SCORE: ", phi_critic)
