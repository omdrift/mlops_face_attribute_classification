"""
Préparation des données d'entraînement depuis data/raw/
"""
import numpy as np
import pandas as pd
import cv2
import torch
import os
from tqdm import tqdm
from pathlib import Path

# Config
RAW_DIR = "data/raw"
LABEL_CSV = "data/annotations/mapped_train.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  Compteurs de debug
stats = {
    'total': 0,
    'not_found': 0,
    'decode_failed': 0,
    'crop_failed': 0,
    'success': 0
}


def preprocess_image(img, size=64):
    """Prétraite une image avec gestion d'erreurs"""
    
    if img is None:
            return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1]
    coords = cv2.findNonZero(thresh)
        
    if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = img[y:y+h, x:x+w]
    else:
            # Pas de contour détecté, prendre l'image entière
            cropped = img
        
    resized = cv2.resize(cropped, (size, size))
    return resized.astype("float32") / 255.0


def find_image_in_raw(filename, raw_dir):
    """
    Cherche une image dans data/raw et ses sous-dossiers
    Retourne le chemin complet si trouvé, None sinon
    """
    raw_path = Path(raw_dir)
    
 # Extract lot from filename (e.g., 's1' from 's1_00000.png')
    # The lot is the prefix before the first underscore and after 's'
    lot = filename.split('_')[0] if '_' in filename else None

    if lot:
        # Construct path as: data/raw/{lot}/{filename}
        full_path = raw_path / lot / filename
        if full_path.exists():
            return str(full_path)

    # Fallback: try direct path in case structure is different
    full_path = raw_path / filename
    if full_path.exists():
        return str(full_path)

def process_training_data():
    print("="*60)
    print("PRÉPARATION DU LOT S1 (TRAINING)")
    print("="*60)
    
    # 1. Lire les labels
    print(f"\n Chargement des annotations: {LABEL_CSV}")
    df = pd.read_csv(LABEL_CSV)
    
    # Filtrer uniquement S1 pour l'entraînement
    stats['total'] = len(df)
    
    print(f"   Total annotations: {len(df)}")
    print(f"   Annotations : {len(df)}")
    print(f"   Lots présents dans le CSV: {sorted(df['filename'].str[:2].unique())}")
    
    images_list = []
    labels_list = []
    missing_files = []
    failed_files = []
    
    # 2. Créer un index des fichiers dans data/raw
    print(f"\n Scan du dossier: {RAW_DIR}")
    raw_path = Path(RAW_DIR)
    
    # Lister tous les fichiers .png dans raw/ et ses sous-dossiers
    all_files = {}
    for img_path in raw_path.rglob("*.png"):
        all_files[img_path.name] = str(img_path)
    
    print(f"   Fichiers PNG trouvés: {len(all_files)}")
    print(f"   Exemples: {list(all_files.keys())[:5]}")
    
    # 3. Traiter chaque annotation S1
    print(f"\n  Traitement des images S1...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        filename = row['filename']
        
        # Chercher le fichier dans l'index
        if filename not in all_files:
            stats['not_found'] += 1
            missing_files.append(filename)
            continue
        
        img_path = all_files[filename]
        
        try:
            # Lire l'image
            img = cv2.imread(img_path)
            
            if img is None:
                stats['decode_failed'] += 1
                failed_files.append(f"{filename} (lecture échouée)")
                continue
            
            # Prétraiter (crop + resize)
            processed = preprocess_image(img, size=64)
            
            if processed is None:
                stats['crop_failed'] += 1
                failed_files.append(f"{filename} (prétraitement échoué)")
                continue
            
            # Ajouter à la liste
            images_list.append(processed)
            labels_list.append([
                row['beard'], 
                row['mustache'], 
                row['glasses_binary'],
                row['hair_color_label'], 
                row['hair_length']
            ])
            stats['success'] += 1
            
        except Exception as e:
            stats['decode_failed'] += 1
            failed_files.append(f"{filename} (erreur: {str(e)})")
    
    # 4. Sauvegarder
    if len(images_list) > 0:
        print(f"\n Sauvegarde des données...")
        X = torch.tensor(np.array(images_list)).permute(0, 3, 1, 2)  # (N, C, H, W)
        y = torch.tensor(np.array(labels_list), dtype=torch.long)
        
        output_path = os.path.join(OUTPUT_DIR, "train_data_s1.pt")
        torch.save({
            'X': X, 
            'y': y,
            'stats': stats,
            'filenames': df['filename'].tolist()
        }, output_path)
        
        print(f" Sauvegardé: {output_path}")
        print(f"   Shape X: {X.shape}")
        print(f"   Shape y: {y.shape}")
    else:
        print(" Aucune image traitée!")
        return
    
    

if __name__ == "__main__":
    process_training_data()