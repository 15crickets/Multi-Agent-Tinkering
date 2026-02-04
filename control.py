from datasets import load_dataset
from openai import OpenAI
import re


#quickly GPT-generated control structure with a single agent

dataset = load_dataset("commonsense_qa", split="train")

client = OpenAI()


single_agent_prompt = """You are an agent tasked with answering the following multiple-choice question.

Instructions:
- Read the question carefully.
- Reason step by step.
- Choose the best answer from the options provided.

Output format:
- One or more lines of explanation
- One line exactly: Answer: X
"""

def format_question(example):
    question = example["question"]
    choices = example["choices"]

    answer_key_lines = [
        f"{label}. {text}"
        for label, text in zip(choices["label"], choices["text"])
    ]

    answer_key = "\n".join(answer_key_lines)

    return f"Question: {question}\nAnswers:\n{answer_key}"


def run_single_agent_control(dataset, num_questions=20):
    results = []

    for i in range(num_questions):
        example = dataset[i]
        full_question = format_question(example)
        gold_answer = example["answerKey"]

        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            instructions=single_agent_prompt,
            input=full_question,
        )

        output_text = response.output_text

        match = re.search(r"Answer:\s*([A-E])", output_text)
        predicted_answer = match.group(1) if match else None

        results.append({
            "index": i,
            "question": example["question"],
            "predicted": predicted_answer,
            "gold": gold_answer,
            "correct": predicted_answer == gold_answer
        })

    return results


results = run_single_agent_control(dataset, num_questions=150)

accuracy = sum(r["correct"] for r in results) / len(results)
print("Single-agent accuracy: ", accuracy)
