# Compte Rendu - Dungeon Challenge

## 1. Entraînement avec les paramètres par défaut

Dans un premier temps, nous avons lancé l'entraînement avec les paramètres par défaut du script :

```bash
uv run train_dungeon_logs.py
```

Cela correspond à la configuration suivante :
- **Mode** : `linear` (pas de récurrence, simple MLP sur les embeddings aplatis)
- **Embed dim / Hidden dim** : 258
- **Optimiseur** : SGD avec un learning rate de 0.1
- **Epochs** : 6
- Pas de dropout, pas de scheduler, pas d'early stopping

Les résultats obtenus sont corrects mais loin d'être optimaux. Le mode `linear` ne prend pas en compte l'ordre des événements dans la séquence : "Potion puis Piège" et "Piège puis Potion" produisent la même prédiction. De plus, le nombre d'epochs (6) est très faible, et l'absence de régularisation limite la capacité du modèle à bien généraliser.

## 2. Optimisation des hyperparamètres

Nous avons ensuite relancé l'entraînement avec une configuration optimisée :

```bash
uv run train_dungeon_logs.py \
    --mode lstm \
    --bidirectional \
    --num_layers 3 \
    --embed_dim 8 \
    --hidden_dim 16 \
    --dropout 0.3 \
    --optimizer adam \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --epochs 30 \
    --early_stopping \
    --patience 7 \
    --use_scheduler
```

### Justification des choix

**Architecture :**
- `--mode lstm` : Le LSTM remplace le mode linéaire. Contrairement au RNN simple, le LSTM possède des portes (forget, input, output) qui lui permettent de retenir des dépendances longue distance dans la séquence, tout en évitant le problème du vanishing gradient. C'est essentiel ici car les séquences de donjon peuvent être longues et l'ordre des événements (rencontrer une potion avant ou après un piège) est déterminant pour la survie.
- `--bidirectional` : Le LSTM bidirectionnel lit la séquence dans les deux sens (gauche-droite et droite-gauche). Chaque position a ainsi accès au contexte passé et futur. Par exemple, savoir qu'un boss attend en fin de donjon peut changer l'interprétation des potions trouvées au début. Les deux états cachés sont concaténés, doublant la dimension de sortie.
- `--num_layers 3` : Empiler 3 couches LSTM permet d'apprendre des représentations de plus en plus abstraites. La première couche capture les patterns locaux, les suivantes combinent ces informations en patterns de plus haut niveau.
- `--embed_dim 8` et `--hidden_dim 16` : Nous avons volontairement choisi des dimensions faibles pour réduire la taille du modèle et surtout le temps d'entraînement, car nous n'avons pas de GPU à disposition. Un modèle plus petit s'entraîne plus vite sur CPU, et le vocabulaire du donjon étant relativement restreint, des embeddings de dimension 8 suffisent à capturer les relations sémantiques entre les événements.
- `--dropout 0.3` : Active le dropout entre les couches LSTM (30% des neurones désactivés aléatoirement à chaque step). C'est une régularisation qui force le réseau à ne pas dépendre excessivement de quelques neurones et qui réduit l'overfitting. Ce paramètre n'est effectif que parce que `num_layers > 1`.

**Optimisation :**
- `--optimizer adam` : Adam maintient un learning rate adaptatif par paramètre et utilise un momentum. Il converge plus vite que SGD et est moins sensible au choix du learning rate.
- `--learning_rate 0.001` : Le learning rate par défaut du script (0.1) est adapté à SGD mais beaucoup trop élevé pour Adam, qui fonctionne typiquement avec des valeurs autour de 0.001.
- `--weight_decay 0.0001` : Régularisation L2 qui pénalise les poids trop grands (`loss = loss + 0.0001 * ||poids||^2`). Cela aide à la généralisation en empêchant le modèle de mémoriser les données d'entraînement.
- `--use_scheduler` : Active un scheduler `ReduceLROnPlateau` qui divise automatiquement le learning rate par 2 quand la validation accuracy stagne pendant 5 epochs. Cela permet de commencer avec un LR élevé pour explorer rapidement l'espace des paramètres, puis de l'affiner progressivement.

**Early stopping :**
- `--early_stopping` avec `--patience 7` : L'entraînement s'arrête automatiquement si la validation accuracy ne s'améliore pas pendant 7 epochs consécutives. Cela évite de continuer à entraîner un modèle qui commence à overfitter et garantit qu'on conserve le meilleur modèle.

## 3. Résultats avec 30 epochs

Avec cette configuration, nous avons obtenu les résultats suivants sur 30 epochs :

| Metric | Valeur finale |
|---|---|
| Train Accuracy | 97.11% |
| Val Accuracy | **96.30%** |
| Train Loss | 0.079 |
| Val Loss | 0.096 |

Les courbes d'entraînement montrent une convergence progressive, avec la loss de validation qui continue de diminuer et l'accuracy qui progresse encore à l'epoch 30. Le gap entre train et validation reste faible (~0.8%), ce qui indique que le modèle généralise bien sans overfitter, grâce au dropout et au weight decay.

## 4. Augmentation du nombre d'epochs

En observant les courbes, on constate qu'à 30 epochs le modèle est encore en phase d'apprentissage : la loss de validation continue de diminuer et l'early stopping ne s'est pas déclenché. Cela signifie que 30 epochs n'est pas suffisant pour exploiter pleinement la capacité du modèle.

Nous avons donc augmenté le nombre d'epochs pour laisser le modèle converger jusqu'au déclenchement de l'early stopping :

```bash
uv run train_dungeon_logs.py \
    --mode lstm \
    --bidirectional \
    --num_layers 3 \
    --embed_dim 8 \
    --hidden_dim 16 \
    --dropout 0.3 \
    --optimizer adam \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --epochs 200 \
    --early_stopping \
    --patience 7 \
    --use_scheduler
```

En fixant `--epochs 200` avec `--patience 7`, l'entraînement continue tant que le modèle progresse et s'arrête automatiquement 7 epochs après le dernier meilleur score. Cela permet de trouver le point optimal sans risquer l'overfitting, et sans avoir à deviner manuellement le bon nombre d'epochs.

Avec cette configuration (7K+ paramètres), l'early stopping s'est déclenché à l'epoch 76 et nous avons atteint une accuracy de validation de ~97%.

## 5. Réduction de la taille du modèle

Ayant atteint 97% d'accuracy avec 34K paramètres, nous avons cherché à réduire la taille du modèle. Nous avons d'abord corrigé un problème dans `baseline_model.py` : le réseau linéaire `solo_embeddings` était instancié même en mode LSTM, ajoutant des paramètres inutiles au checkpoint. Nous avons conditionné sa création au mode `linear` uniquement.

Ensuite, nous avons réduit les dimensions du modèle :
- `embed_dim` : 32 → 16 (le vocabulaire du donjon est petit, ~30 tokens, des embeddings de dimension 16 suffisent)
- `hidden_dim` : 64 → 32 (le hidden_dim intervient au carré dans le calcul des paramètres LSTM, c'est le levier le plus impactant)
- `num_layers` : 3 → 2 (deux couches LSTM suffisent pour capturer les patterns, la troisième apporte peu de gain en accuracy)

```bash
uv run train_dungeon_logs.py \
    --mode lstm \
    --bidirectional \
    --num_layers 2 \
    --embed_dim 16 \
    --hidden_dim 32 \
    --dropout 0.3 \
    --optimizer adam \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --epochs 200 \
    --early_stopping \
    --patience 7 \
    --use_scheduler
```

L'objectif est de vérifier si un modèle plus léger (~42K → ~18K paramètres) peut atteindre des performances comparables, tout en étant plus rapide à entraîner sur CPU et produisant un checkpoint plus petit.