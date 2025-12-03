#!/bin/bash
# Script helper pour DVC - Commandes courantes

set -e

# Couleurs pour l'output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function show_help {
    echo "======================================"
    echo "  DVC Helper Script"
    echo "======================================"
    echo ""
    echo "Usage: ./scripts/dvc_helper.sh [command]"
    echo ""
    echo "Commands:"
    echo "  status         - Afficher le statut du pipeline"
    echo "  repro          - Reproduire tout le pipeline"
    echo "  repro-force    - Forcer la reproduction du pipeline"
    echo "  pull           - Télécharger les données depuis le remote"
    echo "  push           - Pousser les données vers le remote"
    echo "  metrics        - Afficher les métriques"
    echo "  plots          - Afficher les plots"
    echo "  dag            - Afficher le graphe du pipeline"
    echo "  experiments    - Afficher les expériences"
    echo "  clean          - Nettoyer le cache DVC"
    echo "  help           - Afficher cette aide"
    echo ""
}

function check_dvc {
    if ! command -v dvc &> /dev/null; then
        echo -e "${RED}❌ Error: DVC n'est pas installé${NC}"
        echo "   Installez DVC avec: pip install dvc"
        exit 1
    fi
}

function dvc_status {
    echo -e "${GREEN}📊 Statut du pipeline DVC${NC}"
    dvc status
}

function dvc_repro {
    echo -e "${GREEN}🔄 Reproduction du pipeline${NC}"
    dvc repro
    echo -e "${GREEN}✓ Pipeline reproduit avec succès${NC}"
}

function dvc_repro_force {
    echo -e "${YELLOW}⚠️  Reproduction forcée du pipeline${NC}"
    dvc repro -f
    echo -e "${GREEN}✓ Pipeline reproduit avec succès${NC}"
}

function dvc_pull {
    echo -e "${GREEN}⬇️  Téléchargement des données${NC}"
    dvc pull || echo -e "${YELLOW}⚠️  Pas de remote configuré ou pas de données à télécharger${NC}"
}

function dvc_push {
    echo -e "${GREEN}⬆️  Push des données vers le remote${NC}"
    dvc push || echo -e "${YELLOW}⚠️  Pas de remote configuré${NC}"
}

function dvc_metrics {
    echo -e "${GREEN}📈 Métriques du pipeline${NC}"
    echo ""
    dvc metrics show
    echo ""
    echo -e "${GREEN}Différences avec la version précédente:${NC}"
    dvc metrics diff || echo "Pas de version précédente à comparer"
}

function dvc_plots {
    echo -e "${GREEN}📊 Génération des plots${NC}"
    dvc plots show
    echo -e "${GREEN}✓ Plots générés dans dvc_plots/index.html${NC}"
}

function dvc_dag {
    echo -e "${GREEN}🔗 Graphe du pipeline${NC}"
    dvc dag
}

function dvc_experiments {
    echo -e "${GREEN}🧪 Expériences DVC${NC}"
    dvc exp show
}

function dvc_clean {
    echo -e "${YELLOW}🧹 Nettoyage du cache DVC${NC}"
    echo "Cela va supprimer les fichiers non utilisés du cache"
    read -p "Continuer? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        dvc gc -w -f
        echo -e "${GREEN}✓ Cache nettoyé${NC}"
    else
        echo "Nettoyage annulé"
    fi
}

# Main
check_dvc

case "${1:-help}" in
    status)
        dvc_status
        ;;
    repro)
        dvc_repro
        ;;
    repro-force)
        dvc_repro_force
        ;;
    pull)
        dvc_pull
        ;;
    push)
        dvc_push
        ;;
    metrics)
        dvc_metrics
        ;;
    plots)
        dvc_plots
        ;;
    dag)
        dvc_dag
        ;;
    experiments|exp)
        dvc_experiments
        ;;
    clean)
        dvc_clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Commande inconnue: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
