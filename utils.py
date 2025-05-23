"""
utils.py

Ce module fournit des fonctions pour les actions suivantes:
- Générer des données synthétiques.
- Appliquer une analyse en composantes principales (PCA).
- Simuler un modèle de perceptron Teacher-Student sphérique.
- Définir un perceptron avec différentes loss (perceptron et hinge)
  et différentes méthodes d'apprentissage (gradient et langevin).
- Déséquilibrer les classes.
- Séparer les données en ensembles d'entraînement et de test.
- Évaluer les performances d'un modèle.
- Évaluer le taux de déséquilibre intrasèque.
- Faire une cross-validation sur un dataset.
- Afficher différents graphiques (dataset, cross-validation, plusieurs cross-validation).
- Enregistrer une image pour l'affichage de plusieurs cross-validation.

Fonctions :

1. generate(N:int=1000, D:int=125, show:bool=False) -> np.ndarray
   - Génère des données multivariées aléatoires.

2. pca(X:np.ndarray, variance:float=0.95) -> np.ndarray
   - Applique la PCA pour réduire la dimensionnalité des données.

3. teacher(X:np.ndarray, bias:float=-1.0, noise_std:float=0.0, show:bool=False) -> Tuple[np.ndarray, np.ndarray, float]
   - Simule un modèle enseignant (Teacher) pour générer des étiquettes.

4. student(X:np.ndarray, loss:str, method:str) -> Tuple[np.ndarray, float, Callable, Callable, Callable]
   - Initialise un modèle étudiant (Student) avec différentes fonctions de perte et méthodes d'apprentissage.

5. split(X:np.ndarray, Y:np.ndarray, test_size:float=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
   - Divise les données en ensembles d'entraînement et de test.

6. imbalance(X:np.ndarray, Y:np.ndarray, rate:float=None) -> Tuple[np.ndarray, np.ndarray]
   - Crée un déséquilibre de classe dans les données.

7. apprentissage(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient: Callable, 
                  fmethod:Callable, loss:str, eta:float=0.1, maxiter:int=150) -> Tuple[np.ndarray, float]
   - Entraîne le modèle étudiant en utilisant une méthode d'optimisation.

8. score(X:np.ndarray, Y:np.ndarray, w:np.ndarray, b:float) -> Tuple[float, float]
   - Évalue la précision du modèle sur un ensemble de données.

9. cross_validate(X:np.ndarray, Y:np.ndarray, w_init:np.ndarray, b_init:float, floss:Callable, fgradient:Callable, 
                  fmethod:Callable, loss:str, eta:float=0.1, maxiter:int=150, test_size:float=0.2, n_splits:int=10, 
                  ptrain:float=None, ptest:float=None) -> Tuple[List[float], List[float], List[float], List[float]]
   - Effectue une cross-validation et renvoie les résultats sur train et test avec la BA et la ROC AUC.

10. intraseque(Y:np.ndarray) -> float
   - Renvoie le taux de déséquilibre intrasèque de Y.

11. show_data(X:np.ndarray, Y:np.ndarray, w_teacher:np.ndarray, b_teacher:float, w_student:np.ndarray, b_student:float, 
              dim:int=2, ptrain:float=None, ptest:float=None) -> None
   - Affiche les données en 2D avec les frontières de décision des modèles Teacher et Student.

12. show_perf(scores_train:List[float], scores_test:List[float], scores_roctr:List[float], scores_rocte:List[float]) -> None
   - Affiche les performances après cross-validation, sur le train et test sets avec la BA et la ROC AUC.

13. show_perf_per_ptrain(train:List[List[float]], test:List[List[float]], ptrain:List[float], 
                        p0:float, roctr:List[List[float]], rocte: List[List[float]],
                        save:Tuple[int, int, float, float, float, int, int, float, str, str]=None) -> None:
    - Affiche les performances sur train et test selon le ptest choisi (avec ptrain intrasèque) avec la BA et la ROC AUC.

14. show_perf_per_ptest(train:List[List[float]], test:List[List[float]], ptest:List[float], 
                        p0:float, roctr:List[List[float]], rocte: List[List[float]],
                        save:Tuple[int, int, float, float, float, int, int, float, str, str]=None) -> None:
   - Affiche les performances sur train et test selon le ptest choisi (avec ptrain intrasèque) avec la BA et la ROC AUC.

"""

import sklearn.linear_model
import sklearn.decomposition
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, Callable, List
from time import time
# np.random.seed(424242)
np.set_printoptions(threshold=6)



# Génération de données
def generate(N:int=1000, D:int=125, show:bool=False) -> np.ndarray:
    """
    N: Nombre de points de données à générer
    D: Dimension des données à générer
    show: Si True, affiche les données en 2D
    Return: Les points de données, un ndarray de taille (N,D)
    """
    mean = np.zeros(D)
    cov = np.eye(D)
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=N)
    
    # Affiche les données (en 2D)
    if show:
        # Trace le nuage de points
        _fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(X[:, 0], X[:, 1], color='dodgerblue', marker='x', s=20)
        ax.set_title(f"Data 2D Projection")

        # Affichage
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        ax.set_xlim([x_min-0.5, x_max+0.5])
        ax.set_ylim([y_min-0.5, y_max+0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    return X


# Implémentation d'une PCA
def pca(X:np.ndarray, variance:float=0.95) -> np.ndarray:
    """
    X: Points de données (un ndarray de taille (N, D))
    variance: La proportion de variance expliquée que l'on veut garder (0.0 -> 1.0) ou le nombre de composants à garder (>= 1)
    """
    model = sklearn.decomposition.PCA(n_components=variance)
    X_pca = model.fit_transform(X)
    return X_pca


# Implémentation du teacher
def teacher(X:np.ndarray, bias:float=-1.0, noise_std:float=0.0, show:bool=False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    X: Points de données (un ndarray de taille (N, D))
    bias: Le biais choisi (un biais négatif dans notre étude)
    noise_std: L'écart-type du bruit à mettre (si l'on utilise un bruit par perturbation de la frontière)
    show: Si True, affiche les données en 2D et le Teacher
    Return: La valeur de vérité des points (un ndarray de taille N), 
            avec le vecteur de poids du teacher (un ndarray de taille D) et le biais du teacher
    """
    N, D = X.shape
    w = np.random.normal(loc=0, scale=1/np.sqrt(D), size=D)

    # Prédire la classe de chaque point en ajoutant un bruit (pour éviter que le dataset soit linéairement séparable)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=N)  # Choisir une valeur comme 1.O ou 2.0
    Y = np.sign((X @ w) + bias + noise)  # Pour générer du bruit par perturbation de la frontière

    """
    Y = np.sign((X @ w) + bias)
    flip_mask = np.random.rand(Y.shape[0]) < noise_std  # Choisir une valeur comme 0.05 ou 0.1
    Y[flip_mask] *= -1 # Pour générer du bruit par modification d'étiquette
    """

    # Afficher les données et le Teacher (en 2D)
    if show:
        # Trace le nuage de points avec les classes
        _fig, ax = plt.subplots(figsize=(10, 8))
        Xpos, Xneg = X[Y > 0], X[Y < 0]
        ax.scatter(Xneg[:, 0], Xneg[:, 1], color='dodgerblue', marker='x', label='Normal (Y=-1)', s=20)
        ax.scatter(Xpos[:, 0], Xpos[:, 1], color='red', marker='+', label='Anomaly (Y=+1)', s=20)

        # Trace la courbe du teacher
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx = np.linspace(x_min, x_max, 200)
        yy = -(bias + w[0]*xx)/(w[1]+1e-10) 
        ax.plot(xx, yy, '--', color='black', label='Teacher boundary')

        # Affichage
        ax.set_xlim([x_min-0.5, x_max+0.5])
        ax.set_ylim([y_min-0.5, y_max+0.5])
        ax.set_title("Spherical Teacher–Student in 2D")
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    return Y, w, bias


# Implémentation du student
def student(X:np.ndarray, loss:str, method:str) -> Tuple[np.ndarray, float, Callable, Callable, Callable]:
    """
    X: Points de données (un ndarray de taille (N, D))
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'error-counting')
    method: Méthode d'entraînement à utiliser ('gradient', 'langevin')
    Return: Le vecteur w des poids (un ndarray de taille D), le biais, la fonction de loss, la fonction de gradient
             et la méthode pour entraîner le student
    """
    _N, D = X.shape
    w = np.random.normal(loc=0, scale=1/np.sqrt(D), size=D)
    b = np.random.uniform(X.min(), X.max())

    # Définition de la fonction de loss (renvoie la valeur de la loss pour un point X donné)
    if loss=='perceptron':
        def floss(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
            """
            Y: Valeur de vérité du point
            pred: Prédictions faites par le modèle
            Return: La loss pour chaque point (0 pour un point bien classé, val > 0 pour un point mal classé)
            """
            return np.maximum(0, -(Y * pred))
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1, 
                      norm:float=1/np.sqrt(D)) -> Tuple[np.ndarray, float]:
            """
            X: Points de données (un ndarray de taille (N, D))
            Y: Valeur de vérité des points de données (un ndarray de taille N)
            misclassified: Liste des points mal classés
            eta: Taux d'apprentissage
            norm: Valeur de 1/np.sqrt(D)
            Return: Le w et le b pour la mise à jour
            """
            w = (eta * norm) * X[misclassified].T @ Y[misclassified]
            b = eta * norm * np.sum(Y[misclassified])
            return w, b
    elif loss=='hinge':
        def floss(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
            """
            Y: Valeur de vérité du point
            pred: Prédictions faites par le modèle
            Return: La loss pour chaque point (0 pour un point bien classé, val > 0 pour un point mal classé ou proche de la frontière)
            """
            return np.maximum(0, 1 - (Y * pred))
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1,
                      norm:float=1/np.sqrt(D)) -> Tuple[np.ndarray, float]:
            """
            X: Points de données (un ndarray de taille (N, D))
            Y: Valeur de vérité des points de données (un ndarray de taille N)
            misclassified: Liste des points mal classés
            eta: Taux d'apprentissage
            norm: Valeur de 1/np.sqrt(D)
            Return: Le w et le b pour la mise à jour
            """
            w = (eta * norm) * X[misclassified].T @ Y[misclassified]
            b = eta * norm * np.sum(Y[misclassified])
            return w, b
    elif loss=='error-counting':
        def floss(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
            """
            Y: Valeur de vérité du point
            pred: Prédictions faites par le modèle
            Return: La loss pour chaque point (0 pour un point bien classé, 2 pour un point mal classé)
            """
            return 0.5 * np.square(np.sign(pred) - Y)
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1, norm:float=1/np.sqrt(D)) -> None:
            """
            Pas entraînable par descente de gradient
            """
            raise NameError("La loss error-counting n'est pas entraînable par descente de gradient")
    else:
        raise ValueError(f"Il n'existe pas de loss appelée {loss}.")

    # Méthode d'apprentissage
    if method=='gradient':
        def fmethod(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, loss:str, eta:float=0.1, maxiter:int=150,
                    ) -> Tuple[np.ndarray, float] :
            """
            X: Points de données (un ndarray de taille (N,D))
            Y: Valeur de vérité des points (un ndarray de taille N)
            w: Vecteur des poids (un ndarray de taille D)
            bias: Le biais du perceptron
            floss: La fonction de loss à utiliser
            fgradient: La fonction de gradient à utiliser
            eta: Le taux d'apprentissage
            maxiter: Le nombre d'itérations à faire
            Return: Le vecteur w des poids (un ndarray de taille D) du student et son biais, entrainés
            """
            _N, D = X.shape
            norm = 1.0/np.sqrt(D)
            w0 = w.copy()
            for _ in range(maxiter):
                predictions = X @ w0 + bias
                losses = floss(Y, predictions)
                misclassified = (losses > 0)
                if not misclassified.any():
                    break
                w_temp, b_temp = fgradient(X=X, Y=Y, misclassified=misclassified, eta=eta, norm=norm)
                w0 += w_temp
                bias += b_temp
            return w0, bias
    elif method=='langevin':
        def fmethod(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, loss:str, eta:float=0.1, maxiter:int=150,
                    ) -> Tuple[np.ndarray, float] :
            """
            X: Points de données (un ndarray de taille (N,D))
            Y: Valeur de vérité des points (un ndarray de taille N)
            w: Vecteur des poids (un ndarray de taille D)
            bias: Le biais du perceptron
            floss: La fonction de loss à utiliser
            fgradient: La fonction de gradient à utiliser (on peut l'ignorer pour cette méthode)
            loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'error-counting')
            eta: Le taux d'apprentissage (on peut l'ignorer pour cette méthode)
            maxiter: Le nombre d'itérations à faire (défini manuellement pour cette méthode)
            Return: Le vecteur w des poids (un ndarray de taille D) du student et son biais, entrainés
            """
            N, D = X.shape
            Tfinal = 0.005  # Température à régler
            maxiter = 15000  # Maxiter à régler
            w0 = w.copy()
            amplitude = X.max() - X.min()
            sigma_w = amplitude / 10.0  # On garde 1/10 de l'amplitude des données
            sigma_b = amplitude / 10.0  # (entrainé aussi)

            # Fonction de calcul de l'énergie E
            def energy(w_local: np.ndarray, b_local: float, X: np.ndarray, Y: np.ndarray, N: int, loss: str) -> float:
                """
                w_local: Vecteur des poids (un ndarray de taille D)
                b_local: Le biais
                X: Points de données (un ndarray de taille (N,D))
                Y: Valeur de vérité des points (un ndarray de taille N)
                N: Nombre de données
                loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'error-counting')
                Return: La valeur de la loss
                """
                # E = somme de la loss sur tous les points
                predictions = X @ w_local + b_local
                losses = floss(Y, predictions)
                # return np.mean(losses)  # Parait plus logique mais fait une courbe bizarre avec error-counting
                # On calcule l'énergie balanced
                class_indices = ((Y + 1) // 2).astype(int)
                class_counts = np.bincount(class_indices)
                weights = np.where(Y == -1, 1 / class_counts[0], 1 / class_counts[1])
                weights /= weights.sum()
                weighted_loss = np.sum(losses * weights)
                return weighted_loss

            # Énergie du point initial
            E_old = energy(w0, bias, X, Y, N, loss)
            first10 = maxiter/10

            for i in range(maxiter):
                # On choisit un T plus grand au début pour rapidement se rapprocher d'une solution
                if i < first10:
                    T = Tfinal * 10
                else:
                    T = Tfinal

                # Proposer un nouveau w et b en tirant D gaussiennes indépendantes
                w_new = w0 + np.random.normal(0, sigma_w/np.sqrt(D), size=D)
                b_new = bias + np.random.normal(0, sigma_b)
                E_new = energy(w_new, b_new, X, Y, N, loss)
                delta_E = E_new - E_old

                # Probabilité d'acceptation du déplacement
                acceptance = min(np.exp(-delta_E / T), 1)
                if np.random.rand() <= acceptance:
                    w0, bias = w_new, b_new
                    E_old = E_new

            return w0, bias
    else:
        raise ValueError(f"Il n'existe pas de méthode appelée {method}.")
    return w, b, floss, fgradient, fmethod


# Split le dataset en train et test
def split(X:np.ndarray, Y:np.ndarray, test_size:float=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    test_size: La taille du set de test (0.0 -> 1.0)
    Return: X_train, X_test, Y_train, Y_test
    """
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, random_state=424242, stratify=Y)
    return X_train, X_test, Y_train, Y_test


# Induit un déséquilibre de classe
def imbalance(X:np.ndarray, Y:np.ndarray, rate:float=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    rate: Le déséquilibre de classe voulu ]0.0, 1.0[, None si aucun déséquilibre à appliquer
    Return: Les points de données et les valeurs de vérité, rééquilibrés comme voulus
    """
    if rate is not None:
        # Indices de chaque classe
        pos_idx = np.where(Y > 0)[0]
        neg_idx = np.where(Y < 0)[0]
        N = X.shape[0]
        desired_pos = int(round(rate * N))
        desired_neg = N - desired_pos
        if desired_neg == 0 or desired_pos == 0:
            return X, Y

        # Si on a assez de positifs, on fait un sous-échantillonnage
        # Sinon on fait un sur-échantillonnage
        replace_pos = (len(pos_idx) < desired_pos)
        chosen_pos_idx = np.random.choice(pos_idx, size=desired_pos, replace=replace_pos)
        replace_neg = (len(neg_idx) < desired_neg)
        chosen_neg_idx = np.random.choice(neg_idx, size=desired_neg, replace=replace_neg)

        # Concatène les nouveaux indices
        new_idx = np.concatenate([chosen_pos_idx, chosen_neg_idx])
        X_new = X[new_idx]
        Y_new = Y[new_idx]

        return X_new, Y_new
    return X, Y


# Apprentissage du student
def apprentissage(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, fmethod:Callable, loss:str, eta:float=0.1, maxiter:int=150,
                  ) -> Tuple[np.ndarray, float]:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    w: Vecteur des poids (un ndarray de taille D)
    bias: Le biais du perceptron
    floss: Fonction de loss à utiliser
    fgradient: Fonction de gradient à utiliser
    fmethod: Méthode d'entraînement à utiliser
    eta: Le taux d'apprentissage
    maxiter: Le nombre d'itérations à faire
    Return: Le vecteur des poids du student (un ndarray de taille D) et son biais entrainés
    """
    w0 = w.copy()
    w_final, b_final = fmethod(X=X, Y=Y, w=w0, bias=bias, floss=floss, fgradient=fgradient, loss=loss, eta=eta, maxiter=maxiter)
    return w_final, b_final


# Evalue la performance du student avec la balanced accuracy
def score(X:np.ndarray, Y:np.ndarray, w:np.ndarray, b:float) -> Tuple[float, float]:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    w: Vecteur des poids (un ndarray de taille D)
    bias: Le biais du perceptron
    Return: L'accuracy du modèle sur le dataset (balanced accuracy et ROC)
    """
    # Balanced accuracy
    Ypred = X @ w + b
    accuracy = sklearn.metrics.balanced_accuracy_score(Y, np.sign(Ypred))

    # ROC
    if set(np.unique(Y)) == {-1, 1}:
        Y_bin = (Y + 1) // 2  # Passe les labels en {0, 1} plutôt que {-1, 1}
    else:
        Y_bin = Y
    auc = sklearn.metrics.roc_auc_score(Y_bin, Ypred)
    return accuracy, auc


# Cross-validation sur le train et test sets
def cross_validate(X:np.ndarray, Y:np.ndarray, w_init:np.ndarray, b_init:float, floss:Callable, fgradient:Callable, fmethod:Callable, loss:str, eta:float=0.1, 
                   maxiter:int=150, test_size:float=0.2, n_splits:int=10, ptrain:float=None, ptest:float=None) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    w_init: Vecteur des poids de base du student (un ndarray de taille D)
    b_init: Le biais de base du student
    floss: Fonction de loss à utiliser
    fgradient: Fonction de gradient à utiliser
    fmethod: Méthode d'entraînement à utiliser
    eta: Le taux d'apprentissage
    maxiter: Le nombre d'itérations à faire
    test_size: Taille du test set (0.0 -> 1.0)
    n_splits: Nombre de plis de la cross-validation
    ptrain: Le déséquilibre de classe voulu pour ptrain, None si ptrain=p0 (taux de déséquilibre intrasèque)
    ptest: Le déséquilibre de classe voulu pour ptest, None si ptest=p0 (taux de déséquilibre intrasèque)
    Return: La liste des scores sur le train set, et ceux sur le test set (balanced et ROC)
    """
    """"""
    kF = sklearn.model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=424242)
    scores_train = []
    scores_test = []
    scores_roctr = []
    scores_rocte = []
    i = 0
    start = time()

    for train_index, test_index in kF.split(X, Y):
        i += 1
        w0 = w_init.copy()
        b0 = b_init
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Rééquilibre de classe dans les sets  
        X_train, Y_train = imbalance(X_train, Y_train, rate=ptrain)
        X_test, Y_test = imbalance(X_test, Y_test, rate=ptest)
        
        # Entraîner le student
        w_student, b_student = apprentissage(X=X_train, Y=Y_train, w=w0, bias=b0, floss=floss, fgradient=fgradient, 
                                             fmethod=fmethod, loss=loss, eta=eta, maxiter=maxiter)

        # Évaluer le modèle sur le pli
        scrtr = score(X_train, Y_train, w_student, b_student)
        scrte = score(X_test, Y_test, w_student, b_student)
        scores_train.append(scrtr[0])
        scores_roctr.append(scrtr[1])
        scores_test.append(scrte[0])
        scores_rocte.append(scrte[1])
        print(f"Cross-validation réalisée à {(((i)/n_splits)*100):.2f}% après {(time()-start):.0f} secondes")
    return scores_train, scores_test, scores_roctr, scores_rocte


# Donne le taux de déséquilibre intrasèque du dataset
def intraseque(Y:np.ndarray) -> float:
    """
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    Return: La valeur de p0 (taux de déséquilibre intrasèque du dataset)
    """
    N = Y.shape[0]
    N_pos = np.sum(Y == 1)
    return N_pos / N


# Affiche le jeu de données avec le teacher et le student
def show_data(X:np.ndarray, Y:np.ndarray, w_teacher:np.ndarray, b_teacher:float, w_student:np.ndarray, b_student:float, 
              dim:int=2, ptrain:float=None, ptest:float=None) -> None:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    w_teacher: Vecteur des poids du teacher (un ndarray de taille D)
    b_teacher: Le biais du teacher
    w_student: Vecteur des poids du student (un ndarray de taille D)
    b_student: Le biais du student
    dim: Nombre de dimensions avant PCA
    ptrain: Taux de déséquilibre dans ptrain, None si ptrain=p0 (taux de déséquilibre intrasèque)
    ptest: Taux de déséquilibre dans ptest, None si ptest=p0 (taux de déséquilibre intrasèque)
    """
     # Trace le nuage de points avec les classes
    _fig, ax = plt.subplots(figsize=(10, 8))
    Xpos, Xneg = X[Y > 0], X[Y < 0]
    ax.scatter(Xneg[:, 0], Xneg[:, 1], color='dodgerblue', marker='x', label='Normal (Y=-1)', s=20)
    ax.scatter(Xpos[:, 0], Xpos[:, 1], color='red', marker='+', label='Anomaly (Y=+1)', s=20)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # Trace la courbe du teacher
    xxt = np.linspace(x_min, x_max, 200)
    yyt = -(b_teacher + w_teacher[0]*xxt)/(w_teacher[1]+1e-10) 
    ax.plot(xxt, yyt, '--', color='black', label='Teacher boundary')

    # Trace la courbe du student
    xxs = np.linspace(x_min, x_max, 200)
    yys = -(b_student + w_student[0]*xxs)/(w_student[1]+1e-10) 
    ax.plot(xxs, yys, '--', color='green', label='Student boundary')
    
    # Affichage
    ax.set_xlim([x_min-0.5, x_max+0.5])
    ax.set_ylim([y_min-0.5, y_max+0.5])
    ax.set_title("Spherical Teacher–Student Perceptron")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.show()


# Affiche les scores après cross-validation
def show_perf(scores_train:List[float], scores_test:List[float], scores_roctr:List[float], scores_rocte:List[float]) -> None:
    """
    scores_train: Liste des scores du train en cross-validation (Balanced)
    scores_test: Liste des scores du test en cross-validation (Balanced)
    scores_roctr: Liste des scores du train en cross-validation (ROC)
    scores_rocte: Liste des scores du test en cross-validation (ROC)
    """
    folds = range(1, len(scores_train) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(folds, scores_train, marker='o', color='blue', linestyle='-', label='Balanced Score Train')
    plt.plot(folds, scores_test, marker='s', color='orange', linestyle='-', label='Balanced Score Test')
    plt.plot(folds, scores_roctr, marker='o', color='green', linestyle='-', label='ROC Score Train')
    plt.plot(folds, scores_rocte, marker='s', color='purple', linestyle='-', label='ROC Score Test')
    plt.xlabel("Numéro du pli")
    plt.ylabel("Score")
    plt.title("Performances en Cross-Validation")
    plt.legend()
    plt.grid(True)
    plt.show()


# Affiche les performances sur train et test selon le ptrain choisi (avec ptest intrasèque)
def show_perf_per_ptrain(train:List[List[float]], test:List[List[float]], ptrain:List[float], 
                         p0:float, roctr:List[List[float]], rocte: List[List[float]],
                         save:Tuple[int, int, float, float, float, int, int, float, str, str]=None) -> None:
    """
    train: Balanced accuracy pour chaque cross-validation sur le train set
    test: Balanced accuracy pour chaque cross-validation sur le test set
    ptrain: Liste de chaque valeur de ptrain
    p0: Le taux de déséquilibre intrasèque du dataset
    roctr: ROC accuracy pour chaque cross-validation sur le train set
    rocte: ROC accuracy pour chaque cross-validation sur le test set
    save: Si None ne rien faire, sinon, sauvegarde le plot dans /plots. Les valeurs dans save sont:
          N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method
    """
    # Calcul des moyennes et écarts-types
    means_train = [np.mean(scores) for scores in train]
    std_train   = [np.std(scores)  for scores in train]
    means_test = [np.mean(scores) for scores in test]
    std_test   = [np.std(scores)  for scores in test]
    means_roctr = [np.mean(scores) for scores in roctr]
    std_roctr = [np.std(scores) for scores in roctr]
    means_rocte = [np.mean(scores) for scores in rocte]
    std_rocte = [np.std(scores) for scores in rocte]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les résultats pour chaque valeur de ptrain
    for i, p in enumerate(ptrain):
        ax.scatter([p]*len(train[i]), train[i], color='blue', alpha=0.4)
        ax.scatter([p]*len(test[i]), test[i], color='orange', alpha=0.4)
        ax.scatter([p]*len(roctr[i]), roctr[i], color='green', alpha=0.4)
        ax.scatter([p]*len(rocte[i]), rocte[i], color='purple', alpha=0.4)

    # Tracer la moyenne
    ax.plot(ptrain, means_train, color='blue', linewidth=2, linestyle='-', label='Balanced Train')
    ax.plot(ptrain, means_test, color='orange', linewidth=2, linestyle='-', marker='s', label='Balanced Test')
    ax.plot(ptrain, means_roctr, color='green', linewidth=2, linestyle='-', label='ROC Train')
    ax.plot(ptrain, means_rocte, color='purple', linewidth=2, linestyle='-', marker='s', label='ROC Test')
    
    # Remplir l'espace autour de la moyenne avec ±1 écart-type
    ax.fill_between(ptrain, [m - s for m, s in zip(means_train, std_train)], 
                    [m + s for m, s in zip(means_train, std_train)], color='blue', alpha=0.1)
    ax.fill_between(ptrain,[m - s for m, s in zip(means_test, std_test)],
                    [m + s for m, s in zip(means_test, std_test)], color='orange', alpha=0.1)
    ax.fill_between(ptrain, [m - s for m, s in zip(means_roctr, std_roctr)], 
                    [m + s for m, s in zip(means_roctr, std_roctr)], color='green', alpha=0.1)
    ax.fill_between(ptrain,[m - s for m, s in zip(means_rocte, std_rocte)],
                    [m + s for m, s in zip(means_rocte, std_rocte)], color='purple', alpha=0.1)
    
    # Légende hyper-paramètres
    if save is not None:
        N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method = save
        if method == 'gradient':
            textstr = (f"Hyper-Parameters\nN={N}\nD={D}\nBias={bias:.2f}\nTest Size={test_size:.2f}\n"
                       f"Eta={eta}\nMax Iter={maxiter}\nSplits={n_splits}\nNoise Std={noise_std:.2f}\n"
                       f"Loss={loss}\nMethod={method}\nIntrinsic p0={p0:.2f}")
        else:
            maxiter = 15000
            t = 0.005
            textstr = (f"Hyper-Parameters\nN={N}\nD={D}\nBias={bias:.2f}\nTest Size={test_size:.2f}\n"
                       f"T={t}\nMax Iter={maxiter}\nSplits={n_splits}\nNoise Std={noise_std:.2f}\n"
                       f"Loss={loss}\nMethod={method}\nIntrinsic p0={p0:.2f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)

    # Affichage
    ax.set_xlabel("pTrain")
    ax.set_ylabel("Score")
    ax.set_title(f"Cross-Validation over 42 different re-sampling of the train set")
    handles = [plt.Line2D([0], [0], color='blue', lw=2, label='Balanced Train'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Balanced Test'),
    plt.Line2D([0], [0], color='green', lw=2, label='ROC Train'),
    plt.Line2D([0], [0], color='purple', lw=2, label='ROC Test')]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.1))
    ax.grid(True)
    fig.subplots_adjust(right=0.75)

    # Sauvegarde des données
    if save is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "plots")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"TRAIN_N{N}_D{D}_b{bias:.1f}_n{noise_std:.1f}_{loss}_{method}.jpg")
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Affichage sauvegardé sous: {filename}")
    plt.show()
    return fig


# Affiche les performances sur train et test selon le ptest choisi (avec ptrain intrasèque)
def show_perf_per_ptest(train:List[List[float]], test:List[List[float]], ptest:List[float], 
                        p0:float, roctr:List[List[float]], rocte: List[List[float]],
                        save:Tuple[int, int, float, float, float, int, int, float, str, str]=None) -> None:
    """
    train: Balanced accuracy pour chaque cross-validation sur le train set
    test: Balanced accuracy pour chaque cross-validation sur le test set
    ptrain: Liste de chaque valeur de ptrain
    p0: Le taux de déséquilibre intrasèque du dataset
    roctr: ROC accuracy pour chaque cross-validation sur le train set
    rocte: ROC accuracy pour chaque cross-validation sur le test set
    save: Si None ne rien faire, sinon, sauvegarde le plot dans /plots. Les valeurs dans save sont:
          N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method
    """
    # Calcul des moyennes et écarts-types
    means_train = [np.mean(scores) for scores in train]
    std_train   = [np.std(scores) for scores in train]
    means_test = [np.mean(scores) for scores in test]
    std_test   = [np.std(scores) for scores in test]
    means_roctr = [np.mean(scores) for scores in roctr]
    std_roctr = [np.std(scores) for scores in roctr]
    means_rocte = [np.mean(scores) for scores in rocte]
    std_rocte = [np.std(scores) for scores in rocte]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les résultats pour chaque valeur de ptrain
    for i, p in enumerate(ptest):
        ax.scatter([p]*len(train[i]), train[i], color='blue', alpha=0.4)
        ax.scatter([p]*len(test[i]), test[i], color='orange', alpha=0.4, marker='s')
        ax.scatter([p]*len(roctr[i]), roctr[i], color='green', alpha=0.4)
        ax.scatter([p]*len(rocte[i]), rocte[i], color='purple', alpha=0.4, marker='s')

    # Tracer la moyenne
    ax.plot(ptest, means_train, color='blue', linewidth=2, linestyle='-', label='Balanced Train')
    ax.plot(ptest, means_test, color='orange', linewidth=2, linestyle='-', label='Balanced Test')
    ax.plot(ptest, means_roctr, color='green', linewidth=2, linestyle='-', label='ROC Train')
    ax.plot(ptest, means_rocte, color='purple', linewidth=2, linestyle='-', label='ROC Test')
    
    # Remplir l'espace autour de la moyenne avec ±1 écart-type
    ax.fill_between(ptest, [m - s for m, s in zip(means_train, std_train)], 
                    [m + s for m, s in zip(means_train, std_train)], color='blue', alpha=0.1)
    ax.fill_between(ptest,[m - s for m, s in zip(means_test, std_test)],
                    [m + s for m, s in zip(means_test, std_test)], color='orange', alpha=0.1)
    ax.fill_between(ptest, [m - s for m, s in zip(means_roctr, std_roctr)], 
                    [m + s for m, s in zip(means_roctr, std_roctr)], color='green', alpha=0.1)
    ax.fill_between(ptest,[m - s for m, s in zip(means_rocte, std_rocte)],
                    [m + s for m, s in zip(means_rocte, std_rocte)], color='purple', alpha=0.1)
    
    # Légende hyper-paramètres
    if save is not None:
        N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method = save
        if method == 'gradient':
            textstr = (f"Hyper-Parameters\nN={N}\nD={D}\nBias={bias:.2f}\nTest Size={test_size:.2f}\n"
                       f"Eta={eta}\nMax Iter={maxiter}\nSplits={n_splits}\nNoise Std={noise_std:.2f}\n"
                       f"Loss={loss}\nMethod={method}\nIntrinsic p0={p0:.2f}")
        else:
            maxiter = 15000
            t = 0.05
            textstr = (f"Hyper-Parameters\nN={N}\nD={D}\nBias={bias:.2f}\nTest Size={test_size:.2f}\n"
                       f"T={t}\nMax Iter={maxiter}\nSplits={n_splits}\nNoise Std={noise_std:.2f}\n"
                       f"Loss={loss}\nMethod={method}\nIntrinsic p0={p0:.2f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.5, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props)

    # Affichage
    ax.set_xlabel("pTest")
    ax.set_ylabel("Score")
    ax.set_title(f"Cross-Validation over 42 different re-sampling of the test set")
    handles = [plt.Line2D([0], [0], color='blue', lw=2, label='Balanced Train'),
    plt.Line2D([0], [0], color='orange', lw=2, label='Balanced Test'),
    plt.Line2D([0], [0], color='green', lw=2, label='ROC Train'),
    plt.Line2D([0], [0], color='purple', lw=2, label='ROC Test')]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.1))
    ax.grid(True)
    fig.subplots_adjust(right=0.75)

    # Sauvegarde des données
    if save is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(base_dir, "plots")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"TEST_N{N}_D{D}_b{bias:.1f}_n{noise_std:.1f}_{loss}_{method}.jpg")
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Affichage sauvegardé sous: {filename}")
    plt.show()
    return fig


# Exemple d'utilisation
"""
N, D, bias = 1000, 125, -1.0
eta, maxiter, test_size, n_splits, noise_std = 0.1, 150, 0.2, 10, 0.0
X = generate(N=N, D=D, show=False)
Y, w_teacher, b_teacher = teacher(X=X, bias=bias, noise_std=noise_std, show=True)
w_init, b_init, floss, fgradient, fmethod = student(X=X, loss='perceptron', method='gradient')
X_train, X_test, Y_train, Y_test = split(X=X, Y=Y, test_size=test_size)
X_train, Y_train = imbalance(X=X_train, Y=Y_train, rate=0.5)
w_student, b_student = apprentissage(X=X_train, Y=Y_train, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, loss='perceptron', eta=eta, maxiter=maxiter)
print(Y_train.mean(), Y_test.mean(), Y.mean())
print(score(X_train, Y_train, w_student, b_student))
print(score(X_test, Y_test, w_student, b_student))
show_data(X=X, Y=Y, w_teacher=w_teacher, b_teacher=b_teacher, w_student=w_student, b_student=b_student, 
          dim=X.shape[1], ptrain=0.5, ptest=None)
scores_train1, scores_test1, scrtr1, scrte1 = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, 
                                             maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=None, ptest=None)
scores_train2, scores_test2, scrtr2, scrte2 = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, 
                                             maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=0.1, ptest=None)
scores_train3, scores_test3, scrtr3, scrte3 = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, 
                                             maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=0.3, ptest=None)
scores_train4, scores_test4, scrtr4, scrte4 = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, 
                                             maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=0.5, ptest=None)
scores_train5, scores_test5, scrtr5, scrte5 = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, 
                                             maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=0.7, ptest=None)
show_perf(scores_train=scores_train1, scores_test=scores_test1, score_roctr=score_roctr1, score_rocte=score_rocte1)
show_perf_per_ptrain(train=[scores_train2, scores_train3, scores_train4, scores_train5], 
                     test=[scores_test2, scores_test3, scores_test4, scores_test5], 
                     roctr=[scrtr2, scrtr3, scrtr4, scrtr5],
                     rocte=[scrte2, scrte3, scrte4, scrte5],
                     ptrain=[0.1, 0.3, 0.5, 0.7], p0=intraseque(Y=Y), save=None)
"""
