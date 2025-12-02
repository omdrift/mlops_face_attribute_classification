import torch
import zipfile
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from src.models.architecture import MultiHeadResCNN # Votre modèle

# Config
MODEL_PATH = "models/best_model.pth"
RAW_DIR = "data/raw"
OUTPUT_DIR = "data/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_zip(zip_path, model, lot_name):
    print(f"Traitement de {lot_name}...")
    
    predictions = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = [f for f in z.namelist() if f.endswith('.png') or f.endswith('.jpg')]
        
        # On traite par petits lots pour ne pas saturer la mémoire
        batch_size = 64
        batch_imgs = []
        batch_names = []
        
        for i, filename in enumerate(tqdm(file_list)):
            # Lecture image
            img_bytes = z.read(filename)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # --- CROP & RESIZE ---
                # (Même logique que make_dataset.py obligatoire !)
                img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
                
                batch_imgs.append(img)
                batch_names.append(os.path.basename(filename))
            
            # Si le batch est plein ou c'est la fin
            if len(batch_imgs) == batch_size or i == len(file_list) - 1:
                if not batch_imgs: continue
                
                # Conversion Tensor
                tensor_batch = torch.tensor(np.array(batch_imgs)).permute(0, 3, 1, 2).to(DEVICE)
                
                # Prédiction
                with torch.no_grad():
                    out = model(tensor_batch)
                
                # Récupération des résultats
                beards = torch.sigmoid(out['beard']) > 0.5
                mustaches = torch.sigmoid(out['mustache']) > 0.5
                glasses = torch.sigmoid(out['glasses']) > 0.5
                colors = torch.argmax(out['hair_color'], dim=1)
                lengths = torch.argmax(out['hair_length'], dim=1)
                
                # Stockage
                for j in range(len(batch_imgs)):
                    predictions.append({
                        "filename": batch_names[j],
                        "beard": int(beards[j]),
                        "mustache": int(mustaches[j]),
                        "glasses": int(glasses[j]),
                        "hair_color": int(colors[j]),
                        "hair_length": int(lengths[j])
                    })
                
                # Reset batch
                batch_imgs = []
                batch_names = []

    # Sauvegarde CSV
    df_pred = pd.DataFrame(predictions)
    df_pred.to_csv(os.path.join(OUTPUT_DIR, f"preds_{lot_name}.csv"), index=False)
    print(f"Fichier généré : preds_{lot_name}.csv")

def run_inference():
    # 1. Charger le modèle
    model = MultiHeadResCNN(n_color=5, n_length=3).to(DEVICE)
    # Charger les poids (adapter selon comment vous avez sauvegardé)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Boucler sur les Lots 2 à 9
    for i in range(2, 10): # De 2 à 9
        lot_name = f"S{i}"
        zip_path = os.path.join(RAW_DIR, f"{lot_name}.zip")
        
        if os.path.exists(zip_path):
            predict_zip(zip_path, model, lot_name)
        else:
            print(f"Attention: {lot_name}.zip non trouvé dans {RAW_DIR}")

if __name__ == "__main__":
    run_inference()