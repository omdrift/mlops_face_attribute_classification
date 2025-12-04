"""
Préparation des données d'entraînement depuis data/raw/
Training data is driven ONLY by the annotation CSV (mapped_train.csv).

The CSV should contain a 'filename' column with relative paths from data/raw/
(e.g., 's1/img_001.png') and label columns (beard, mustache, glasses_binary, 
hair_color_label, hair_length).

Images are loaded from data/raw/<filename> where filename can include subdirectories.
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
    Constructs the full path to an image in data/raw.
    
    The filename is just the image name (e.g., 's1_00000.png').
    The lot is extracted from the filename prefix (e.g., 's1' from 's1_00000.png').
    The actual path is constructed as: data/raw/{lot}/{filename}
    
    Args:
        filename: Just the image filename (e.g., 's1_00000.png')
        raw_dir: Root directory for raw images
        
    Returns:
        Full path to image if it exists, None otherwise
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
    
    return None


def process_training_data():
    """
    Build training dataset from mapped_train.csv.
    
    Only images listed in the CSV are included in the training set.
    The CSV 'filename' column should contain relative paths from data/raw/
    (e.g., 's1/img_001.png').
    """
    print("="*60)
    print("PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT")
    print("CSV-DRIVEN: Only images in mapped_train.csv are included")
    print("="*60)
    
    # 1. Lire les annotations depuis le CSV
    print(f"\n📄 Chargement des annotations: {LABEL_CSV}")
    if not os.path.exists(LABEL_CSV):
        raise FileNotFoundError(f"Annotation CSV not found: {LABEL_CSV}")
        
    df = pd.read_csv(LABEL_CSV)
    stats['total'] = len(df)
    
    print(f"   Total annotations dans le CSV: {len(df)}")
    
    # Vérifier les colonnes requises
    required_cols = ['filename', 'beard', 'mustache', 'glasses_binary', 
                     'hair_color_label', 'hair_length']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    images_list = []
    labels_list = []
    missing_files = []
    failed_files = []
    processed_filenames = []
    
    # 2. Traiter chaque image listée dans le CSV
    print(f"\n[*] Traitement des images listées dans le CSV...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        filename = row['filename']
        
        # Construire le chemin complet vers l'image
        img_path = find_image_in_raw(filename, RAW_DIR)
        
        if img_path is None:
            stats['not_found'] += 1
            missing_files.append(filename)
            continue
        
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
            processed_filenames.append(filename)
            stats['success'] += 1
            
        except Exception as e:
            stats['decode_failed'] += 1
            failed_files.append(f"{filename} (erreur: {str(e)})")
    
    # 3. Sauvegarder
    if len(images_list) > 0:
        print(f"\n[*] Sauvegarde des données...")
        X = torch.tensor(np.array(images_list)).permute(0, 3, 1, 2)  # (N, C, H, W)
        y = torch.tensor(np.array(labels_list), dtype=torch.long)
        
        output_path = os.path.join(OUTPUT_DIR, "train_data_s1.pt")
        torch.save({
            'X': X, 
            'y': y,
            'stats': stats,
            'filenames': processed_filenames
        }, output_path)
        
        print(f"[+] Sauvegardé: {output_path}")
        print(f"   Shape X: {X.shape}")
        print(f"   Shape y: {y.shape}")
    else:
        print("[-] Aucune image traitée!")
        return
    
    # 4. Afficher le résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DU TRAITEMENT")
    print(f"{'='*60}")
    print(f"  Total annotations dans CSV: {stats['total']}")
    print(f"  [+] Succès: {stats['success']}")
    print(f"  [-] Non trouvées: {stats['not_found']}")
    print(f"  [-] Échec de lecture: {stats['decode_failed']}")
    print(f"  [-] Échec de prétraitement: {stats['crop_failed']}")
    
    if missing_files:
        print(f"\n[!] Fichiers manquants (premiers 10):")
        for f in missing_files[:10]:
            print(f"     - {f}")
    
    if failed_files:
        print(f"\n[!] Fichiers en échec (premiers 10):")
        for f in failed_files[:10]:
            print(f"     - {f}")
    
    print(f"{'='*60}")
    
    

if __name__ == "__main__":
    process_training_data()