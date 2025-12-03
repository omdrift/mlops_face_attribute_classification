"""
Script pour intégrer les nouvelles annotations dans le dataset existant
et versionner avec DVC
"""
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class AnnotationUpdater:
    """Mise à jour et versionnement des annotations"""
    
    def __init__(self, 
                 new_annotations_path="data/annotations/new_mapped_train.csv",
                 old_annotations_path="data/annotations/mapped_train.csv",
                 raw_data_dir="data/raw"):
        self.new_annotations_path = new_annotations_path
        self.old_annotations_path = old_annotations_path
        self.raw_data_dir = raw_data_dir
        
    def load_annotations(self):
        """Charge les anciennes et nouvelles annotations"""
        print(" Chargement des annotations...")
        
        # Nouvelles annotations (S5, S6)
        df_new = pd.read_csv(self.new_annotations_path)
        print(f" Nouvelles annotations: {len(df_new)} lignes")
        print(f"   Préfixes trouvés: {df_new['filename'].str[:2].unique()}")
        
        # Anciennes annotations (si elles existent)
        if os.path.exists(self.old_annotations_path):
            df_old = pd.read_csv(self.old_annotations_path)
            print(f" Anciennes annotations: {len(df_old)} lignes")
            print(f"   Préfixes trouvés: {df_old['filename'].str[:2].unique()}")
        else:
            print("  Pas d'anciennes annotations trouvées")
            df_old = pd.DataFrame()
        
        return df_old, df_new
    
    def clean_filenames(self, df):
        """Nettoie les noms de fichiers (enlève .csv.png → .png)"""
        print("\n🧹 Nettoyage des noms de fichiers...")
        
        df['filename'] = df['filename'].str.replace('.csv.png', '.png', regex=False)
        
        # Vérifier les doublons
        duplicates = df[df.duplicated('filename', keep=False)]
        if len(duplicates) > 0:
            print(f" {len(duplicates)} doublons détectés:")
            print(duplicates[['filename']].head())
        
        return df
    
    def merge_annotations(self, df_old, df_new):
        """Fusionne les anciennes et nouvelles annotations"""
        print("\n Fusion des annotations...")
        
        if df_old.empty:
            df_merged = df_new.copy()
            print(" Seulement nouvelles annotations utilisées")
        else:
            # Identifier les lots existants
            prefixes_old = set(df_old['filename'].str[:2].unique())
            prefixes_new = set(df_new['filename'].str[:2].unique())
            
            print(f"   Anciens préfixes: {prefixes_old}")
            print(f"   Nouveaux préfixes: {prefixes_new}")
            
          
            # Fusionner
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['filename'], keep='last')
            
            print(f" Fusion terminée: {len(df_merged)} lignes")
            print(f"   Lots finaux: {sorted(df_merged['filename'].str[:2].unique())}")
        
        return df_merged
    
    def save_merged_annotations(self, df, backup=False):
        """Sauvegarde les annotations fusionnées"""
        print("\n Sauvegarde des annotations...")
        
        # Créer le dossier de backup si nécessaire
        os.makedirs(os.path.dirname(self.old_annotations_path), exist_ok=True)
        
        # Backup de l'ancien fichier
        if backup and os.path.exists(self.old_annotations_path):
            backup_path = self.old_annotations_path
            shutil.copy2(self.old_annotations_path, backup_path)
            print(f" Backup créé: {backup_path}")
        
        # Sauvegarder le nouveau fichier
        df.to_csv(self.old_annotations_path, index=False)
        print(f" Nouvelles annotations sauvegardées: {self.old_annotations_path}")
        print(f"   Total: {len(df)} lignes")
        
        # Statistiques
        print(f"\n Statistiques des annotations:")
        for col in ['beard', 'mustache', 'glasses_binary']:
            print(f"   {col:15}: {df[col].value_counts().to_dict()}")
        
        print(f"\n   Couleurs cheveux: {df['hair_color_label'].value_counts().to_dict()}")
        print(f"   Longueurs cheveux: {df['hair_length'].value_counts().to_dict()}")
    
  
    
        

    
    def run(self):
        """Pipeline complet de mise à jour"""
        print("="*60)
        print(" MISE À JOUR DES ANNOTATIONS")
        print("="*60)
        
        # 1. Charger
        df_old, df_new = self.load_annotations()
        
        
        # 3. Fusionner
        df_merged = self.merge_annotations(df_old, df_new)

        # 5. Sauvegarder
        self.save_merged_annotations(df_merged)
        
        # 6. Instructions DVC
        
        # 7. Rapport
      
        
        return df_merged


def main():
    """Exécution principale"""
    updater = AnnotationUpdater(
        new_annotations_path="data/annotations/new_mapped_train.csv",
        old_annotations_path="data/annotations/mapped_train.csv",
        raw_data_dir="data/raw"
    )
    
    df_merged = updater.run()
    
    print(f"\n Résultat final: {len(df_merged)} annotations")


if __name__ == '__main__':
    main()