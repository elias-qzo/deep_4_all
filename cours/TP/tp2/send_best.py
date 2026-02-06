"""
Envoie le meilleur modèle au leaderboard.
Usage: uv run send_best.py --team_name "MonEquipe"
"""

import argparse
import re
from pathlib import Path

from gradio_client import Client, handle_file


def parse_result(result):
    """Parse le résultat brut de la soumission et affiche un résumé lisible."""
    markdown_text = result[0]
    leaderboard = result[1]

    # Extraire les infos du markdown
    val_acc = re.search(r"Accuracy Validation.*?`([\d.]+%)`", markdown_text)
    test_acc = re.search(r"Accuracy Test Secret.*?`([\d.]+%)`", markdown_text)
    gap = re.search(r"Gap.*?`([^`]+)`", markdown_text)
    f1 = re.search(r"F1-Score Test.*?`([\d.]+%)`", markdown_text)
    params = re.search(r"Paramètres.*?`(\d+)`", markdown_text)

    print("\n" + "=" * 50)
    print("        RÉSULTAT DE LA SOUMISSION")
    print("=" * 50)
    if val_acc:
        print(f"  Accuracy Validation : {val_acc.group(1)}")
    if test_acc:
        print(f"  Accuracy Test Secret: {test_acc.group(1)}")
    if gap:
        print(f"  Gap (Val - Test)    : {gap.group(1)}")
    if f1:
        print(f"  F1-Score Test       : {f1.group(1)}")
    if params:
        print(f"  Paramètres          : {params.group(1)}")
    print("=" * 50)

    # Trouver le rang dans le leaderboard
    data = leaderboard.get("data", [])
    team_name_col = 1  # colonne Equipe
    for row in data:
        if row[team_name_col] and test_acc:
            test_val = float(test_acc.group(1).replace("%", ""))
            if row[3] == test_val:
                print(f"\n  Rang au leaderboard : #{row[0]} / {len(data)}")
                break

    # Top 5
    print("\n  --- Top 5 ---")
    for row in data[:5]:
        marker = " <--" if row[0] == 1 else ""
        print(f"  #{row[0]:2d} {row[1]:<20s} Test: {row[3]}%{marker}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Envoyer best_model.pt au leaderboard")
    parser.add_argument("--team_name", type=str, required=True, help="Nom de l'équipe")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).parent / "checkpoints" / "best_model.pt"),
        help="Chemin vers le modèle (défaut: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://cd18-193-48-121-96.ngrok-free.app/",
        help="URL du serveur leaderboard",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Modèle non trouvé: {model_path}")
        return

    print(f"Envoi du modèle: {model_path}")
    print(f"Équipe: {args.team_name}")
    print(f"Serveur: {args.url}")

    client = Client(args.url)
    result = client.predict(
        team_name=args.team_name,
        model_file=handle_file(str(model_path)),
        api_name="/process_submission",
    )
    parse_result(result)


if __name__ == "__main__":
    main()
