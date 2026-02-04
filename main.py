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
            match = re.search(r'Final Answer: (\w)', response)
            return {"solver_text": response, "final_answer": match.group(1) if match else None}

        elif self.name == "Critic":
            conf_match = re.search(r'Confidence: (\d)', response)
            ans_match = re.search(r'Suggested Answer: (\w)', response)
            return {
                "confidence": int(conf_match.group(1)) if conf_match else None,
                "suggested_answer": ans_match.group(1) if ans_match else None,
                "critic_reasoning": response   # <-- ADDED
            }


planner_prompt = """You are the Planner agent. Analyze the question and identify what is being asked. 
Focus on:
1. The type of question (e.g., fill-in-the-blank, cause-effect, etc.)
2. The key entities and relationships
3. Any constraints or hints in the wording

Do NOT guess the answer. Output your reasoning in a clear, structured way that can be passed to the Solver.
"""

solver_prompt = """You are the Solver agent. Using only the Planner's reasoning, choose the correct multiple choice answer. 
Do NOT introduce new facts or assumptions. 
Output your reasoning, then on the last line write 'Final Answer: X' where X is your choice.
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

planner_agent = Agent("Planner", planner_prompt)
solver_agent = Agent("Solver", solver_prompt)
critic_agent = Agent("Critic", critic_prompt)

def orchestrator(input_question, input_question_answer, planner_agent, solver_agent, critic_agent):
    log_file = "orchestrator_log.txt"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("NEW QUESTION\n")
        f.write("=" * 80 + "\n\n")

        planner_output = planner_agent.run(input_question)
        plan = planner_output["plan"]

        f.write(input_question_answer + "\n\n")

        f.write("PLANNER OUTPUT:\n")
        f.write("-" * 40 + "\n")
        f.write(plan + "\n\n")

        solver_input = f"""Question: {input_question_answer}. Planner Output: {plan}"""
        solver_output = solver_agent.run(solver_input)

        solver_reasoning = solver_output["solver_text"]
        solver_answer = solver_output["final_answer"]

        f.write("SOLVER REASONING:\n")
        f.write("-" * 40 + "\n")
        f.write(solver_reasoning + "\n\n")

        f.write("SOLVER ANSWER:\n")
        f.write("-" * 40 + "\n")
        f.write(str(solver_answer) + "\n\n")

        critic_string = f"""Question: {input_question_answer}. Planner Output: {plan}. Solver Reasoning {solver_reasoning}. Solver Final Answer: {solver_answer}"""
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

    if critic_output["confidence"] is not None and critic_output["confidence"] <= 2:
        final_answer = critic_output.get("suggested_answer", solver_answer)
    else:
        final_answer = solver_answer

    return final_answer




results = []
answers = []

for i in range(150):
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
        planner_agent,
        solver_agent,
        critic_agent
    )

    results.append(pred)


similarity = sum(x == y for x, y in zip(results, answers)) / len(answers)
print("Accuracy: ", similarity*100, "%")


