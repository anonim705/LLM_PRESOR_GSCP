import pandas as pd
from enum import Enum
import os
from dotenv import load_dotenv
import argparse
import time

# --- Load Environment Variables ---
load_dotenv()

# --- API & Local Model Libraries ---
import openai
import anthropic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LLMClient:
    """
    Handles API calls for providers OR loads local Hugging Face models.
    """

    def __init__(self, model_type: 'ModelType'):
        self.model_type = model_type
        self.model_name = model_type.value

        # --- UPDATED: Support both Mistral and LLaMA locally ---
        self.LOCAL_MODELS = [
            ModelType.MISTRAL_LARGE_INSTRUCT,
            ModelType.LLAMA_3_3_70B_INSTRUCT
        ]

        if self.model_type in self.LOCAL_MODELS:
            print(f"INFO: Setting up local model '{self.model_name}'. This may take a while...")
            self.pipe = self._setup_local_pipeline()
        else:
            self._setup_api_client()

    def _setup_local_pipeline(self):
        """Sets up and loads a local Hugging Face model via a pipeline."""
        if not torch.cuda.is_available():
            print("WARNING: No CUDA-enabled GPU found. Model will load on CPU, which will be extremely slow.")
            device_map = "cpu"
        else:
            device_map = "auto"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,   # good default on modern GPUs / falls back if unsupported
            trust_remote_code=True,
            attn_implementation="eager"   # --- Better compatibility (Mistral/LLaMA)
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    def _setup_api_client(self):
        """Sets up clients for API-based models."""
        api_key_map = {
            ModelType.GPT_4o: ("OPENAI_API_KEY", "OpenAI"),
            ModelType.CLAUDE_3_7_SONNET: ("ANTHROPIC_API_KEY", "Anthropic"),
            ModelType.DEEPSEEK_V3: ("DEEPSEEK_API_KEY", "DeepSeek"),
        }
        env_key, provider_name = api_key_map.get(self.model_type)
        self.api_key = os.getenv(env_key)
        if not self.api_key:
            raise ValueError(f"API key for {provider_name} not found. Please add {env_key} to your .env file.")

        if self.model_type in [ModelType.GPT_4o]:
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.model_type in [ModelType.CLAUDE_3_7_SONNET]:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.model_type == ModelType.DEEPSEEK_V3:
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gets a single response from the configured LLM (local or API).
        """
        try:
            # --- Local inference (Mistral/LLaMA) ---
            if self.model_type in self.LOCAL_MODELS:
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user",  "content": user_prompt}]

                # Common terminators for instruction-tuned Mistral/LLaMA
                terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = self.pipe(
                    messages,
                    max_new_tokens=50,
                    eos_token_id=terminators,
                    do_sample=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id  # avoid pad warning
                )
                for message in outputs[0]["generated_text"]:
                    if message.get('role') == 'assistant':
                        return message.get('content', 'parsing_error').strip()
                return "parsing_error_no_assistant_message"

            # --- API-based logic ---
            elif self.model_type in [ModelType.GPT_4o, ModelType.DEEPSEEK_V3]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()

            elif self.model_type in [ModelType.CLAUDE_3_7_SONNET]:
                response = self.client.messages.create(
                    model=self.model_name, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=50
                )
                return response.content[0].text.strip()

        except Exception as e:
            print(f"    ! Error during model inference: {e}. Waiting 5 seconds...")
            time.sleep(5)
            return "API_ERROR"

class CultureType(Enum):
    DEFAULT = "default"  #without any culture
    CLAN = "clan"
    ADHOCRACY = "adhocracy"
    MARKET = "market"
    HIERARCHY = "hierarchy"


class ModelType(Enum):
    gpt= "gpt-4o"
    claude = "claude-3-5-sonnet-20240620"
    deepseek = "deepseek-chat"
    llama = "meta-llama/Llama-3.3-70B-Instruct"
    mistral = "mistralai/Mistral-Large-Instruct-2407"


def get_system_prompt(culture_type: CultureType, likert_scale: list, main_question: str = None) -> str:
    descriptions = {
       CultureType.DEFAULT:  ("" ),
        CultureType.CLAN: (
            "You are an employee of an organization with a Clan culture. "
            "This type of organization has an internal focus and values flexibility. "
            "It is structured like a family, emphasizing cooperation, trust, and employee commitment. "
            "Your responses should reflect a culture that values cooperation, involvement, teamwork, trust and care for employees."
        ),
        CultureType.ADHOCRACY: (
            "You are an employee of an organization with an Adhocracy culture. "
            "This type of organization has an external focus and values flexibility. "
            "It is a dynamic, entrepreneurial, and innovative environment with an emphasis on risk-taking and experimentation. "
            "Your responses should reflect a culture that values innovation, empowerment, autonomy, risk-taking and creativity. "
        ),
        CultureType.MARKET: (
            "You are an employee of an organization with a Market culture. "
            "This type of organization has an external focus and values stability. "
            "It is a results-driven, competitive atmosphere with a focus on goal achievement, productivity, and market share. "
            "Your responses should reflect a culture that values achievement, performance, work pressure, recognition and goal-orientation."
        ),
        CultureType.HIERARCHY: (
            "You are an employee of an organization with a Hierarchy culture. "
            "This type of organization has an internal focus and values stability. "
            "It is a formalized, structured, and rule-driven environment with an emphasis on efficiency, consistency, and predictability. "
            "Your responses should reflect a culture that values structure, role clarity, ethical aspects, safety and control."
        )
    }

    culture_description = descriptions[culture_type]
    likert_scale_str = "\n".join(likert_scale)
    
     constraints = (
        "To prevent elaboration beyond the provided Likert scale, follow these strict constraints:\n"
        "(1) Do not elaborate on your reasoning.\n"
        "(2) Do not say any other things instead of options.\n"
        "(3) Do not apologize.\n"
        "(4) Do not include any 'note' or 'disclaimer'.\n"
        "(5) Never use words like 'cannot', 'unable', 'instead', 'as', 'however', 'it', 'unfortunately', or 'important'.\n"
        "(6) Do not include any negative sentences on the subject of the prompt.\n"
    )


    if main_question:
        task_instruction = (
            f"{main_question}\n"
            f"You must respond with only one of the following options. \n"
            f"{constraints}\n"
            f"Options:\n{likert_scale_str}"
        )
    else:
        task_instruction = (
            f"For each statement, indicate how well it describes you or your opinions. Select one of the following options. \n"
            f"{constraints}\n"
            f"Options:\n{likert_scale_str}"
        )

    return f"{culture_description}\n\n{task_instruction}"


def _execute_questions(client: LLMClient, culture: CultureType, survey_name: str, questions: list, system_prompt: str,
                       run_number: int):
    results = []
    total_questions = len(questions)
    for i, question_text in enumerate(questions, 1):
        print(f"  [Question {i}/{total_questions}]──> {question_text[:70]}...")
        response = client.get_response(system_prompt, question_text)

        while response == "API_ERROR":
            print("  Retrying last request...")
            response = client.get_response(system_prompt, question_text)

        print(f"    > Received response: '{response}'")
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": survey_name,
            "question": question_text,
            "run_number": run_number,
            "response": response
        })
    return results


def run_presor_survey(client: LLMClient, culture: CultureType, run_number: int):
    statements = [
        "Social responsibility and profitability can be compatible.",
        "To remain competitive in a global environment, business firms will have to disregard ethics and social responsibility.",
        "Good ethics is often good business.",
        "If survival of business enterprise is at stake, then ethics and social responsibility must be ignored.",
        "Being ethical and socially responsible is the most important thing a firm can do.",
        "A firm's first priority should be employee morale.",
        "The overall effectiveness of a business can be determined to a great extent by the degree to which it is ethical and socially responsible.",
        "The ethics and social responsibility of a firm is essential to its long term profitability.",
        "Business has a social responsibility beyond making a profit.",
        "Business ethics and social responsibility are critical to the survival of a business enterprise.",
        "If the stockholders are unhappy, nothing else matters.",
        "The most important concern for a firm is making a profit, even if it means bending or breaking the rules.",
        "Efficiency is much more important to a firm than whether or not the firm is seen as ethical or socially responsible."
    ]
    likert_scale = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Slightly Disagree', 'Neutral',
                    'Slightly Agree', 'Somewhat Agree', 'Agree', 'Strongly Agree']
    system_prompt = get_system_prompt(culture, likert_scale)
    return _execute_questions(client, culture, "PRESOR", statements, system_prompt, run_number)


def run_GSCS_survey(client: LLMClient, culture: CultureType, run_number: int):
    statements = [
        "It is important to develop a mutual understanding of responsibilities regarding environmental performance with our suppliers",
        "It is important to work together to reduce environmental impact of our activities with our suppliers",
        "It is important to conduct joint planning to anticipate and resolve environmental-related problems with our suppliers",
        "It is important to make joint decisions about ways to reduce overall environmental impact of our products with our suppliers",
        "It is important to develop a mutual understanding of responsibilities regarding environmental performance with our customers",
        "It is important to work together to reduce environmental impact of our activities with our customers",
        "It is important to conduct joint planning to anticipate and resolve environmental-related problems with our customers",
        "It is important to make joint decisions about ways to reduce overall environmental impact of our products with our customers"
    ]
    likert_scale = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree',
                    'Strongly Agree']
    system_prompt = get_system_prompt(culture, likert_scale)
    return _execute_questions(client, culture, "GSCS", statements, system_prompt, run_number)


def run_survey(client: LLMClient, culture: CultureType, survey_name: str, run_number: int):
    survey_functions = {
        "PRESOR": run_presor_survey, "GSCS": run_GSCS_survey, }
    if survey_name not in survey_functions:
        raise ValueError(f"Unknown survey: {survey_name}.")
    results = survey_functions[survey_name](client, culture, run_number)
    return results


def save_results_to_excel(results, filename="survey_results.xlsx"):
    if not results:
        print("No results to save.")
        return
    long_df = pd.DataFrame(results)
    wide_df = long_df.pivot(index='run_number', columns='question', values='response')
    final_df = wide_df.reset_index()
    final_df.to_excel(filename, index=False)
    print(f"\n✅ Results successfully saved in wide format to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM sustainability surveys with full survey repetitions.")
    parser.add_argument("--model", required=True, choices=[m.value for m in ModelType], help="The LLM to use.")
    parser.add_argument("--culture", required=True, choices=[c.value for c in CultureType],
                        help="The cultural persona for the LLM.")
    parser.add_argument("--survey", required=True,
                        choices=["PRESOR",  "GSCS"],
                        help="The survey to run.")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to repeat the entire survey.")

    args = parser.parse_args()
    selected_model = ModelType(args.model)  
    selected_culture = CultureType(args.culture)
    model_alias = selected_model.name
    output_filename = f"{model_alias}_{args.survey}_{args.culture}_{args.runs}runs.xlsx"

    try:
        print("=" * 60)
        print("🚀 INITIALIZING SURVEY SESSION")
        print(f"  - Model:         {args.model}")
        print(f"  - Culture:       {args.culture}")
        print(f"  - Survey:        {args.survey}")
        print(f"  - Total Runs:    {args.runs}")
        print(f"  - Output File:   {output_filename}")
        print("=" * 60)

        selected_model = ModelType(args.model)
        selected_culture = CultureType(args.culture)
        all_results = []
        client = LLMClient(selected_model)

        for i in range(1, args.runs + 1):
            print(f"\n--- Starting Run {i} of {args.runs} ---")
            single_run_results = run_survey(client, selected_culture, args.survey, run_number=i)
            all_results.extend(single_run_results)
            print(f"--- Finished Run {i} of {args.runs} ---")

        save_results_to_excel(all_results, filename=output_filename)
        print("\n🎉 Session complete.")
    except (ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")