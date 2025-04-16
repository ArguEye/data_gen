from openai import OpenAI
import os
import json
import random
import itertools
from dotenv import load_dotenv
load_dotenv()


openai_client = None

def generate_all_templates(num_of_templates_per_class = 5, num_of_instances_per_kind = 20):
    """
        ask an AI to generate the templates for sentences for each empty class (class that only has [] as a value) in the dataset_templates file.

        Args:
            num_of_templates_per_class (int): how many different structures of sentences will be generated
            number_of_instances_per_kind (int): how many different words or phrases will be generate for each object, description or place in the template.
    
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise RuntimeError("Please create an env file with OPENAI_API_KEY to use the dataset template generation or use an existing template")

    global openai_client
    openai_client = OpenAI(api_key=key)
    here = os.getcwd()
    with open(os.path.join(here,"dataset_templates.json"),"r+") as file:
        dataset_templates = json.load(file)

        for class_name in dataset_templates.keys():
            if dataset_templates[class_name] == []:
                dataset_templates[class_name] = generate_class_templates(num_of_templates_per_class, num_of_instances_per_kind, class_name)
                file.seek(0)
                json.dump(dataset_templates, file, indent=2)
                file.truncate()


def generate_class_templates(num_of_templates_per_class, num_of_instances_per_kind, class_name):
    prompt = "say 'no prompt file?'"
    here = os.getcwd()
    with open(os.path.join(here,"generator_prompt.txt"),"r", encoding='utf-8') as file:
        prompt = file.read()
    prompt = prompt.replace("num_of_templates_per_class",str(num_of_templates_per_class))
    prompt = prompt.replace("num_of_instances_per_kind",str(num_of_instances_per_kind))
    #print("running with prompt",prompt ,"and class",class_name)
#    print(f"generating template with prompt: {prompt} ")
    print(f"generating template for class: {class_name} ")
    return json.loads(openai_client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "developer", "content": prompt},
        {"role": "user", "content": f"the class is '{class_name}'"}
        
    ],
    temperature=1.166,
    top_p = 0.9,
    max_tokens = 16600
    ).choices[0].message.content)


def generate_dataset(number_of_variations_per_template, prefixes=["cctv camera footage of","low quality photo","photo of","phone photo of","black and white photo of","movie footage with","blurred image with"],prefix_chance=0.1):
    """
    mostly a gemini function
    Generates sentences based on templates and vocabulary lists in dataset_templates.json
    and saves them to dataset.json. Assumes perfect input data and file structure.

    Args:
        number_of_variations_per_template (int): The number of sentences to generate per template.
                 If -1, generates all possible combinations for each template.
        prefixes (List): list of prefixes that can be added to the descriptions randomally
        prefix_chance (float): the portion of descriptions with the added prefix

    """
    here = os.getcwd()
    template_file_path = os.path.join(here, "dataset_templates.json")
    dataset_file_path = os.path.join(here, "dataset.json")

    with open(template_file_path, "r", encoding='utf-8') as f:
        dataset_templates = json.load(f)

    generated_data = {}
    total_sentences_generated = 0

    for class_name, templates in dataset_templates.items():
        generated_data[class_name] = []

        for template_obj in templates:
            template_string = template_obj["template"]

            # Identify placeholders and their corresponding vocabulary lists
            placeholders = {}
            placeholder_keys_in_order = []
            for key, value in template_obj.items():
                if key != "template":
                    placeholder_tag = key
                    if placeholder_tag in template_string:
                        placeholders[placeholder_tag] = value
                        placeholder_keys_in_order.append(placeholder_tag)

            # --- Generation Logic ---
            sentences_for_this_template = []

            if number_of_variations_per_template == -1:
                # Generate all combinations
                vocabulary_lists = [placeholders[key] for key in placeholder_keys_in_order]
                all_combinations = itertools.product(*vocabulary_lists)

                for combination in all_combinations:
                    sentence = template_string
                    # Replace placeholders with words from the current combination
                    for idx, key in enumerate(placeholder_keys_in_order):
                        sentence = sentence.replace(key, str(combination[idx]))
                    sentences_for_this_template.append(sentence)

            elif number_of_variations_per_template > 0:
                # Generate n random sentences
                for _ in range(number_of_variations_per_template):
                    sentence = template_string
                    # Choose random words and substitute
                    for key in placeholder_keys_in_order:
                        chosen_word = random.choice(placeholders[key])
                        # Replace all occurrences of the placeholder key
                        sentence = sentence.replace(key, str(chosen_word))
                    sentences_for_this_template.append(sentence)

            for i in range(len(sentences_for_this_template)):
                if random.uniform(0,1.0) <= prefix_chance:
                    sentences_for_this_template[i] = random.choice(prefixes) + " " + sentences_for_this_template[i]
            generated_data[class_name].extend(sentences_for_this_template)
            total_sentences_generated += len(sentences_for_this_template)


    # Save the generated data
    with open(dataset_file_path, "w", encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2)
    print(f"Dataset generation complete. Saved to {dataset_file_path}")




if __name__ == "__main__":

    while True:
        choice = int(input(
            """1: generate a new template with open AI for the empty classes in dataset_templates.json
2: generate a new dataset using the template
3: close\n"""))
        if choice == 3:
            break
        if choice == 1:
            generate_all_templates()
        if choice == 2:
            generate_dataset(-1)