import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm  # Importer tqdm pour la barre de progression

# Charger le fichier Excel
file_path = 'raw_data.xlsx'
df = pd.read_excel(file_path)

# Initialiser le modèle de génération de texte
torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Prompt par défaut
default_prompt = "Je veux que tu résumes cette conversation sous forme de phrase complète (le client informe l'agent qu'il y a un dysfonctionnement avec son compteur et l'agent lui demande s'il est sans courant actuellement) en disant ce que le client a dit et ce que l'agent a répondu :"

# Configuration de génération
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Fonction pour générer un résumé à partir d'une conversation
def generate_summary(conversation):
    complete_message = f"{default_prompt} {conversation}"
    output = pipe(complete_message, **generation_args)
    return output[0]['generated_text']

# Appliquer la génération de résumés sur chaque conversation avec une barre de progression
tqdm.pandas()  # Initialiser tqdm pour les opérations de pandas
df['résumé'] = df['conversation'].progress_apply(generate_summary)

# Enregistrer le dataframe avec les résumés dans un nouveau fichier Excel
output_file_path = 'data_processed.xlsx'
df.to_excel(output_file_path, index=False)
