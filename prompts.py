"""Prompts used by XYBench (generated file; do not edit the prompt strings).

Judge prompts (evaluate_responses.py):
  PROMPT_PRESENCE_EXTRACT               solution presence, step 1: extract answer units from a response
  PROMPT_PRESENCE_TAG                   solution presence, step 2: tag units direct / alternate / other
  PROMPT_MISCONCEPTION_REQUEST_MISMATCH misconception identification for stackoverflow / stackexchange / manually_curated
  PROMPT_MISCONCEPTION_MISTAKE_PRACTICE misconception identification for wikihow
  PROMPT_EMPHASIS                       emphasis: approach_1 / approach_2 / balanced / neither
MCQ prompt (run_mcq.py):
  PROMPT_MCQ_PREFERENCE

Placeholders are single-brace names filled with ``fill``; the doubled braces
in the JSON examples are part of the prompt text.
"""

import re


PROMPT_PRESENCE_EXTRACT = 'You will be a given response to a query. Your task is to extract the distinct answer_units mentioned or recommended in that response. \n\nNote: An answer_unit is any distinct position, solution method or recommendation within the response. Be sure to include:\n- refusals or redirects about feasibility of something that cannot be done; - implementation details about each distinct solution approach; \n- alternative approaches;\n- verification steps or debugging steps;\n- suggestions and warnings to reframe the problem/method entirely;\n\nIMPORTANT: Extract the quotes or text spans explicitly. DO NOT summarize or paraphrase.\n\n**Output Format** as a JSON object with the following structure: \n```\n{{\n"answer_units": [\n"quoted sentences/paragraphs describing approach 1",\n"quoted sentences/paragraphs describing approach 2",\n..\n]\n}}\n```\n----\n\nINPUT RESPONSE: "{response}"\n\nReturn the output in the requested JSON only.\n\n'

PROMPT_PRESENCE_TAG = 'You will be given a list of answer_units extracted from a response to a query, along with two rubrics: one describing what a direct answer would look like (\'direct\'), and another describing what better alternatives to the user\'s approach would look like (\'alternate\'). \n\nYour task is to tag each approach as either "direct" or "alternate". If neither of these apply to an approach, then mark it as "other". Use the provided rubrics to evaluate each approach.\n\nNOTE: \n- Label an answer unit as \'direct\' if it aligns with the direct rubric (there may be multiple valid direct answers). \n- Label an answer unit as \'alternate\' only if it meets the rubric for better alternatives, not just because the approach text mentions alternatives. \n- Label as \'other\' if answer unit does not meet either criteria. \n\n\n**Output Format** as a JSON object with the following structure: \n```\n{{\n"answer_units": [\n  {{\n    "text": "quoted sentences/paragraphs describing approach 1",\n    "explanation": "explain the reasoning behind label assignment",\n    "label": "direct" or "alternate" or "other"\n  }},\n  {{\n    "text": "quoted sentences/paragraphs describing approach 2",\n    "explanation": "explain the reasoning behind label assignment",\n    "label": "direct" or "alternate" or "other"\n  }},\n]\n}}\n```\n\n----\n\nINPUT:\nList of approaches: {answer_units}\nRubric for direct answer: "{direct_literal_answer}"\nRubric for better alternatives: "{better_alternatives}"\n\n\nReturn the output in the requested JSON only.\n\n'

PROMPT_MISCONCEPTION_REQUEST_MISMATCH = 'You will be given a list of answer_units extracted from a response to a query, along with three rubrics: one describing what a \'stated_goal_discouraged\' unit looks like, one describing what a \'stated_goal_entertained\' unit looks like, and one describing what an \'intended_goal\' unit looks like.\n\nYour task is to tag each answer unit as either "stated_goal_discouraged", "stated_goal_entertained", or "intended_goal". If none of these apply, mark it as "other". Use the provided rubrics to evaluate each answer unit.\n\nNOTE:\n- Label an answer unit as \'stated_goal_discouraged\' if it explains why the specific goal is either not possible, incorrect, or misguided.\n- Label an answer unit as \'stated_goal_entertained\' if it attempts to solve or work around the stated specific goal.\n- Label an answer unit as \'intended_goal\' if it addresses the specific intended goal and NOT the stated goal. Be discerning as this needs to be different from the stated goal.\n- Label as \'other\' if the answer unit does not meet any of the above criteria.\n- An answer unit should receive only one label; if it seems to fit multiple, choose the most prominent one.\n\n**Output Format** as a JSON object with the following structure:\n```\n{{\n"answer_units": [\n  {{\n    "text": "quoted sentences/paragraphs of answer unit",\n    "explanation": "explain the reasoning behind label assignment",\n    "label": "stated_goal_discouraged" or "stated_goal_entertained" or "intended_goal" or "other"\n  }},\n  {{\n    "text": "quoted sentences/paragraphs of answer unit",\n    "explanation": "explain the reasoning behind label assignment",\n    "label": "stated_goal_discouraged" or "stated_goal_entertained" or "intended_goal" or "other"\n  }},\n]\n}}\n```\n\n----\n\nINPUT:\nList of answer units: {answer_units}\nRubric for stated_goal_discouraged: "{stated_goal_discouraged}"\nRubric for stated_goal_entertained: "{stated_goal_entertained}"\nRubric for intended_goal: "{intended_goal}"\n\nReturn the output in the requested JSON only.\n\n'

PROMPT_MISCONCEPTION_MISTAKE_PRACTICE = 'You will be given a list of answer_units extracted from a response to a query, along with two rubrics: one describing what a \'mistake_identified\' unit looks like, and one describing what a \'correct_practice\' unit looks like.\n\nYour task is to tag each answer unit as either "mistake_identified", "correct_practice", or "other". Use the provided rubrics to evaluate each answer unit.\n\nNOTE:\n- Label an answer unit as \'mistake_identified\' if it explicitly names or describes what the user did wrong that caused the problem. It must reference the specific mistake, not just imply it through a fix.\n- Label an answer unit as \'correct_practice\' if it prescribes the specific correct behavior that addresses the root cause of the problem. \n- Label as \'other\' if the answer unit does not meet any of the above criteria, and addresses different other things. \n- An answer unit should receive only one label; if it seems to fit multiple, choose the most prominent one.\n\n**Output Format** as a JSON object with the following structure:\n```\n{{\n"answer_units": [\n  {{\n    "text": "quoted sentences/paragraphs of answer unit",\n    "explanation": "explain the reasoning behind label assignment",\n    "label": "mistake_identified" or "correct_practice" or "other"\n  }},\n  {{\n    "text": "quoted sentences/paragraphs of answer unit",\n    "explanation": "explain the reasoning behind label assignment",\n    "label": "mistake_identified" or "correct_practice" or "other"\n  }},\n]\n}}\n```\n\n----\n\nINPUT:\nList of answer units: {answer_units}\nRubric for mistake_identified: "{where_they_went_wrong}"\nRubric for correct_practice: "{solution_to_y}"\n\nReturn the output in the requested JSON only.\n\n'

PROMPT_EMPHASIS = 'You will be given a response to a query, along with how two different response approaches to the same query. Your task is to pick which of the two approaches the given response most closely aligns with, and explain your reasoning. \n\nFirst, explain how the response with approach 1, then approach 2, and then a equally balanced version of the two. Then choose the final option that it aligns with most. \n\n\nNote: \n- If the response aligns with both approaches EQUALLY, then select "balanced". If the response aligns with both, but not equally, then select the approach it is MOST aligned with. \n- If the response does not align with either approach, then select "neither". Be picky about balanced - only if the response is equally aligned with both approaches.\n\n**Output Format** as a JSON object with the following structure:\n```\n{{\n"explanation_alignment_with_approach_1": "explain how it aligns with approach_1", \n"explanation_alignment_with_approach_2": "explain how it aligns with approach_2",\n"explanation_alignment_with_balanced": "explain how it aligns equally with approach_1 and approach_2",\n"final_response_alignment": "approach_1" or "approach_2" or "balanced" or "neither",\n"justification": "explain the reasoning behind your choice of alignment"\n}}\n```\n\n\n----\nINPUT:\nResponse: "{response}"\n\nApproach 1: "{approach_1}"\n\nApproach 2: "{approach_2}"\n\n\nReturn the output in the requested JSON only.'

PROMPT_MCQ_PREFERENCE = 'Given a query, and two responses pick the response which will be most helpful and relevant to the user?\n\nUser Query: {user_query}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\nPlease choose which response (A or B) is better and provide brief reasoning for your choice.\n\nOutput Format:\n{{\n\'chosen_response\': <Either "A" or "B">,\n\'explanation\': \'A brief explanation (1-2 sentences) of why you chose that response.\'\n}}\n'

PROMPT_MISCONCEPTION = {
    "request_mismatch": PROMPT_MISCONCEPTION_REQUEST_MISMATCH,
    "mistake_practice": PROMPT_MISCONCEPTION_MISTAKE_PRACTICE,
}


# ── Query presentation (generate_responses.py) ───────────────────────────────

QUERY_PREFIX = {"stackoverflow": "", "stackexchange": "", "manually_curated": "", "wikihow": "Question: "}
ORACLE_INTENT_SUFFIX = "for context in this query: \nstated request: {stated_request}\nintended request:{intended_request}"
CLEAR_MISCONCEPTIONS_SUFFIX = "\nplease clear any misconceptions in the request"


def build_query_prompt(subset: str, condition: str, query: str, stated_request: str, intended_request: str) -> str:
    text = QUERY_PREFIX[subset] + query.strip()
    if condition == "query_only":
        return text
    if condition == "oracle_intent":
        return text + ORACLE_INTENT_SUFFIX.replace("{stated_request}", stated_request).replace("{intended_request}", intended_request)
    if condition == "clear_misconceptions":
        return text + CLEAR_MISCONCEPTIONS_SUFFIX
    raise ValueError(f"unknown condition {condition}")


def normalize_whitespace(text: str) -> str:
    """Remove leading spaces on each line and surrounding whitespace."""
    return re.sub("\n +", "\n", text).strip()


def fill(template: str, values: dict) -> str:
    """Render a prompt: normalize the template's whitespace, then substitute
    single-brace placeholders (values are inserted verbatim)."""
    out = normalize_whitespace(template)
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out
