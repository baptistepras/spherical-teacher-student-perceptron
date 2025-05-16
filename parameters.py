"""
parameters.py

Ce module permet d'entraîner les hyperparamètres.
Pour modifier les paramètres de l'entraînement, le faire directement dans
les fonctions avec les variables situées au début de chaque fonction.

Fonctions (celles prises dan main.py et utils.py peuvent être légèrement modifiées):

1. fetch_data(N:int=500, D:int=50, bias:float=-1, noise_std:float=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]
   - Charge et retourne un jeu de données sauvegardé depuis le dossier "data" sous la forme d'un tuple (X, Y, w, b).

2. student(X:np.ndarray, loss:str='perceptron', method:str='gradient') -> Tuple[np.ndarray, float, Callable, Callable, Callable]
   - Initialise un modèle étudiant (Student) avec différentes fonctions de perte et méthodes d'apprentissage.

3. split(X:np.ndarray, Y:np.ndarray, test_size:float=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
   - Divise les données en ensembles d'entraînement et de test.

4. apprentissage(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient: Callable, 
                  fmethod:Callable, eta:float=0.1, maxiter:int=100, t:float=0.1) -> Tuple[np.ndarray, float, np.ndarray, float]
   - Entraîne le modèle étudiant en utilisant une méthode d'optimisation.

5. score(X:np.ndarray, Y:np.ndarray, w:np.ndarray, b:float) -> float
   - Évalue la précision du modèle sur un ensemble de données.

6. show_different_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None
   - Montre le graphe au fur et à mesure des itérations sur différentes valeurs de T pour la dynamique de Langevin

7. find_best_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None
   - Cherche graphiquement le T optimal pour la dynamique de Langevin

8. find_best_T_multiple_runs(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None
   - Cherche graphiquement le T optimal pour la dynamique de Langevin avec plusieurs runs pour plus de précision

9. find_best_maxiter(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str, method: str) -> None
   - Cherche graphiquement le meilleur maxiter pour la dynamique de Langevin ou le maxiter optimal pour la descente de gradient

10. find_best_eta(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None
   - Cherche graphiquement le meilleur eta pour la descente de gradient
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
maxiter = 15000
T = 0.005

# Récupère le jeu de données voulu (chatGPT)
# Si les données n'existe pas, les créé
def fetch_data(N:int=500, D:int=50, bias:float=-1, noise_std:float=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    N: Nombre de points de données du dataset à fetch
    D: Dimension des données du dataset à fetch
    bias : Biais du teacher à fetch
    noise_std: Le bruit à appliquer
    Return: Les données sauvegardés (X, Y, w, b).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", f"N{N}_D{D}_b{bias:.1f}_n{noise_std:.1f}.npz")
    if not os.path.exists(file_path):
        raise ValueError("Ce dataset n'existe pas")
    loaded = np.load(file_path)

    # On récupère le dictionnaire puis on extrait chacun des tableaux
    data_dict = {key: loaded[key] for key in loaded.files}
    return data_dict['X'], data_dict['Y'], data_dict['w'], float(data_dict['b'])


# Implémentation du student (légèrement différente que la version dans utils.py)
def student(X:np.ndarray, loss:str='perceptron', method:str='gradient') -> Tuple[np.ndarray, float, Callable, Callable, Callable]:
    """
    X: Points de données (un ndarray de taille (N, D))
    loss: Fonction de loss à utiliser ('perceptron', loss, 'error-counting')
    method: Méthode d'entraînement à utiliser ('gradient', 'langevin')
    Return: Le vecteur w des poids (un ndarray de taille D), le biais, la fonction de loss, la fonction de gradient
             et la méthode pour entraîner le student
    """
    _, D = X.shape
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
            _, D = X.shape
            w = (eta * norm) * X[misclassified].T @ Y[misclassified]
            b = eta * norm * np.sum(Y[misclassified])
            return w, b
    elif loss==loss:
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
            _, D = X.shape
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
            return 0.5 * np.square((np.sign(pred) - Y))
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1) -> None:
            """
            Pas entraînable par descente de gradient
            """
            raise NameError("La loss error-counting n'est pas entraînable par descente de gradient")
    else:
        raise ValueError(f"Il n'existe pas de loss appelée {loss}.")

    # Méthode d'apprentissage
    if method=='gradient':
        def fmethod(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, eta:float=0.1, maxiter:int=100, t:float=1,
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
            _, D = X.shape
            norm = 1.0/np.sqrt(D)
            w0 = w.copy()
            vec_score = []
            vec_score.append(score(X, Y, w0, bias))
            somme = 0
            for _ in range(maxiter):
                somme += 1
                predictions = X @ w0 + bias
                losses = floss(Y, predictions)
                misclassified = (losses > 0)
                if not misclassified.any():
                    break
                w_temp, b_temp = fgradient(X=X, Y=Y, misclassified=misclassified, eta=eta, norm=norm)
                w0 += w_temp
                bias += b_temp
                vec_score.append(score(X, Y, w0, bias))
            return w0, bias, vec_score, somme
    elif method=='langevin':
        def fmethod(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, eta:float=0.1, maxiter:int=100, t=0.3
                    ) -> Tuple[np.ndarray, float] :
            """
            X: Points de données (un ndarray de taille (N,D))
            Y: Valeur de vérité des points (un ndarray de taille N)
            w: Vecteur des poids (un ndarray de taille D)
            bias: Le biais du perceptron
            floss: La fonction de loss à utiliser
            fgradient: La fonction de gradient à utiliser (on peut l'ignorer pour cette méthode)
            eta: Le taux d'apprentissage (on peut l'ignorer pour cette méthode)
            maxiter: Le nombre d'itérations à faire (x10 pour cette méthode)
            Return: Le vecteur w des poids (un ndarray de taille D) du student et son biais, entrainés
            """
            N, D = X.shape
            maxiter = maxiter  # maxiter à régler
            w0 = w.copy()
            amplitude = X.max() - X.min()
            sigma_w = amplitude / 10.0  # On garde 1/10 de l'amplitude des données
            sigma_b = amplitude / 10.0  
            vec_E = []  # Sauvegarde des différentes énergies
            vec_score = []  # Sauvegarde des différents scores

            # Fonction de calcul de l'énergie E
            def energy(w_local: np.ndarray, b_local: float, X: np.ndarray, Y: np.ndarray, N: int) -> float:
                """
                w_local: Vecteur des poids (un ndarray de taille D)
                b_local: Le biais
                X: Points de données (un ndarray de taille (N,D))
                Y: Valeur de vérité des points (un ndarray de taille N)
                N: Nombre de données
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
            E_old = energy(w0, bias, X, Y, N)
            vec_E.append(E_old)
            vec_score.append(score(X, Y, w0, bias))

            max10 = maxiter/20
            somme = 0

            for i in range(maxiter):
                # On choisit un T plus grand au début pour rapidement se rapprocher d'une solution
                if i < max10:
                    T = t*10
                else:
                    T = t
                
                # Proposer un nouveau w et b en tirant D gaussiennes indépendantes
                w_new = w0 + np.random.normal(0, sigma_w/D**0.5, size=D)
                b_new = bias + np.random.normal(0, sigma_b)
                E_new = energy(w_new, b_new, X, Y, N)
                delta_E = E_new - E_old

                # Probabilité d'acceptation du déplacement
                acceptance = min(np.exp(-delta_E / T), 1)
                # print("Acceptance, delta_E: ", acceptance, delta_E)
                if np.random.rand() <= acceptance:
                    w0, bias = w_new, b_new
                    E_old = E_new
                    somme += 1
                # print("E, score: ", E_old, score(X, Y, w0, bias))
                vec_E.append(E_old)
                vec_score.append(score(X, Y, w0, bias))

            # Afficher l'évolution à travers le temps
            """
            _, ax = plt.subplots(figsize=(10, 8))
            ax.plot(np.linspace(0, maxiter, maxiter+1), vec_score, '-', color='red', label='Score Evolution')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Score")
            ax.set_title("Evolution of Score over time")
            ax.legend()
            plt.show()
            """
            
            return w0, bias, vec_score, somme/maxiter*100
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


# Apprentissage du student (légèrement différente que la version dans utils.py)
def apprentissage(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, fmethod:Callable, eta:float=0.1, maxiter:int=100,
                  t:float=0.1) -> Tuple[np.ndarray, float, np.ndarray, float]:
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
    t: La valeur de T à donner dans la méthode de langevin
    Return: Le vecteur des poids du student (un ndarray de taille D) et son biais entrainés,
            ainsi que le vecteur des scores au fur et à mesure des itérations et le taux d'acception
    """
    w0 = w.copy()
    w_final, b_final, vec, acceptance = fmethod(X=X, Y=Y, w=w0, bias=bias, floss=floss, fgradient=fgradient, eta=eta, maxiter=maxiter, t=t)
    return w_final, b_final, vec, acceptance


# Apprentissage du student
def apprentissage2(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, fmethod:Callable, eta:float=0.1, maxiter:int=100,
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
    w_final, b_final, _, _= fmethod(X=X, Y=Y, w=w0, bias=bias, floss=floss, fgradient=fgradient, eta=eta, maxiter=maxiter)
    return w_final, b_final


# Evalue la performance du student
def score(X:np.ndarray, Y:np.ndarray, w:np.ndarray, b:float) -> float:
    """
    X: Points de données (un ndarray de taille (N, D))
    Y: Valeur de vérité des points de données (un ndarray de taille N)
    w: Vecteur des poids (un ndarray de taille D)
    bias: Le biais du perceptron
    Return: L'accuracy du modèle sur le dataset
    """
    Ypred = np.sign(X @ w + b)
    accuracy = sklearn.metrics.balanced_accuracy_score(Y, Ypred)
    return accuracy


# Montre sur différentes températures le résultat de l'apprentissage au fur et à mesure du temps
def show_different_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    loss: La loss à utiliser
    """
    t = np.linspace(0.01, 0.05, 5)  # Modifier si besoin
    w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss=loss, method='langevin')
    fig, axes = plt.subplots(5, 1, figsize=(20, 6), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.8)
    cmap = plt.get_cmap("tab10")

    for i in range(5):
        # Apprentissage
        w_student, b_student, vec, acceptance = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                             fmethod=fmethod, eta=0.1, maxiter=maxiter, t=t[i])
        
        # Calcul des scores
        train_score = score(Xtrain, Ytrain, w_student, b_student)
        test_score = score(Xtest, Ytest, w_student, b_student)

        # Affichage de la courbe
        ax = axes[i]
        ax.plot(np.linspace(0, maxiter, maxiter+1), vec, '-', color=cmap(i))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title(f"t={t[i]:.3f}, train={train_score}, test={test_score}, acceptance={acceptance}")

    plt.show()


# Cherche graphiquement le T optimal
def find_best_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    loss: La loss à utiliser
    """
    n = 10  # Modifier si besoin
    t_values = np.linspace(0.005, 0.05, n)   # Modifier si besoin
    _, ax = plt.subplots(figsize=(10, 6))
    w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss=loss, method='langevin')
    train_scores = []
    test_scores = []
    acceptances = []
    start = time()

    for i, t in enumerate(t_values):
        w_student, b_student, _, acceptance = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                             fmethod=fmethod, eta=0.1, maxiter=maxiter, t=t)
        train_scores.append(score(Xtrain, Ytrain, w_student, b_student))
        test_scores.append(score(Xtest, Ytest, w_student, b_student))
        acceptances.append(acceptance/100)
        print(f"Test effectué à {(((i+1)/n)*100):.2f}% après {(time()-start):.0f} secondes")

    # Calcul des maximums
    best_train_idx = np.argmax(train_scores)
    best_test_idx = np.argmax(test_scores)
    best_train = train_scores[best_train_idx]
    best_test = test_scores[best_test_idx]
    best_t_train = t_values[best_train_idx]
    best_t_test = t_values[best_test_idx]
    acceptance_best_train = acceptances[best_train_idx]
    acceptance_best_test = acceptances[best_test_idx]

    # Affichage des courbes
    ax.plot(t_values, train_scores, '-', color='blue', label=f"Train")
    ax.plot(t_values, test_scores, '-', color='orange', label=f"Test")
    ax.plot(t_values, acceptances, '-', color='green', label=f"Acceptance\nBest train={acceptance_best_train:.2f}\nBest test={acceptance_best_test:.2f}")
    
    # Affichage du maximum pour T 
    ax.axvline(best_t_train, linestyle="--", color="blue", alpha=0.7, label=f"Best Train T={best_t_train:.3f}")
    ax.axvline(best_t_test, linestyle="--", color="orange", alpha=0.7, label=f"Best Test T={best_t_test:.3f}")

    # Affichage des meilleurs scores
    ax.scatter(best_t_train, best_train, color="blue", marker="x", label=f"Best Train Score={best_train:.3f}")
    ax.scatter(best_t_test, best_test, color="orange", marker="x", label=f"Best Test Score={best_test:.3f}")

    # Affichage du graphique
    ax.set_xlabel("Score")
    ax.set_ylabel("T")
    ax.set_title("Train and test scores over different values of T, with acceptance rate")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()


# Cherche graphiquement le T optimal plus précisément
def find_best_T_multiple_runs(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    loss: La loss à utiliser
    """
    runs = 5 # Modifier si besoin
    n = 10  # Modifier si besoin
    t_values = np.linspace(0.001*1000, 0.01*1000, n)  # Pour vérifier un intervalle
    # t_values = np.r_[np.linspace(0.0001, 0.001, 10), np.linspace(0.002, 0.01, 9)]  # Pour ajouter comparer deux intervalles
    # t_values = np.array([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])  # Pour une vision globale
    n = len(t_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    start = time()

    all_train_scores = np.zeros((runs, n))
    all_test_scores = np.zeros((runs, n))
    all_acceptances = np.zeros((runs, n))
    
    # Exécution de plusieurs runs
    for j in range(runs):
        w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss=loss, method='langevin')
        train_scores = []
        test_scores = []
        acceptances = []

        # Exécution d'une run en particulier
        for i, t in enumerate(t_values):
            w_student, b_student, _, acceptance = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                                                  fmethod=fmethod, eta=0.1, maxiter=maxiter, t=t)
            
            train_scores.append(score(Xtrain, Ytrain, w_student, b_student))
            test_scores.append(score(Xtest, Ytest, w_student, b_student))
            acceptances.append(acceptance/100)
            total_progress = ((j * n + (i + 1)) / (runs * n)) * 100
            print(f"Test effectué à {total_progress:.2f}% après {(time()-start):.0f} secondes")

        all_train_scores[j, :] = train_scores
        all_test_scores[j, :] = test_scores
        all_acceptances[j, :] = acceptances

    # Calcul des moyennes et écarts-types
    mean_train_scores = np.mean(all_train_scores, axis=0)
    mean_test_scores = np.mean(all_test_scores, axis=0)
    mean_acceptances = np.mean(all_acceptances, axis=0)
    
    std_train_scores = np.std(all_train_scores, axis=0)
    std_test_scores = np.std(all_test_scores, axis=0)
    std_acceptances = np.std(all_acceptances, axis=0)

    # Courbes moyennes
    ax.plot(t_values, mean_train_scores, '-', color='blue', marker='x', label="Train (mean)")
    ax.plot(t_values, mean_test_scores, '-', color='orange', marker='x', label="Test (mean)")
    ax.plot(t_values, mean_acceptances, '-', color='green', marker='x', label="Acceptance (mean)")

    # Zones d'incertitude (± std)
    ax.fill_between(t_values, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, color='blue', alpha=0.2)
    ax.fill_between(t_values, mean_test_scores - std_test_scores, mean_test_scores + std_test_scores, color='orange', alpha=0.2)
    ax.fill_between(t_values, mean_acceptances - std_acceptances, mean_acceptances + std_acceptances, color='green', alpha=0.2)

    # Affichage du graphique
    # ax.set_xscale('log')
    ax.set_xlabel("T")
    ax.set_ylabel("Score / Acceptance Rate")
    ax.set_title(f"Train and Test Scores over different values of T\n(Mean ± Std over {runs} runs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Sauvegarde des données
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "plots_optimization")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"T_optimization.jpg")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Affichage sauvegardé sous: {filename}")
    
    plt.show()


# Cherche graphiquement le maxiter optimal (pour langevin)
def find_best_maxiter(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str, method: str) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    loss: La loss à utiliser
    method: La méthode à utiliser
    """
    if method == 'gradient':
        runs = 1000  # Beaucou de runs nécessaires
        iter = 500  # Modifier si besoin
    else:
        runs = 5  # Peu de runs nécessaires
        iter = 20000  # Modifier si besoin
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    start = time()
    iters = []

    # Apprentissage
    for i in range(runs):
        w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss=loss, method=method)
        _, _, vec, somme = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                                                  fmethod=fmethod, eta=0.1, maxiter=iter, t=T)

        # Affichage des courbes
        if method == 'gradient':
            iters.append(somme)
            ax.plot(np.linspace(0, somme, somme), vec, '-', color='blue')
        else:
            ax.plot(np.linspace(0, iter, iter+1), vec, '-', color=cmap(i), label=f"run{i+1}")
        print(f"Test effectué à {((i+1)/runs*100):.2f}% après {(time()-start):.0f} secondes")
    
    # Affichage du graphique
    if method == 'gradient':
        top5 = sorted(iters, reverse=True)[:5]
        textstr = f"Nombre de\n runs={runs}\nMax Iters:\n" + "\n".join([f"{v}" for v in top5])
        ax.text(1.02, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax.set_xlabel("maxiter")
    ax.set_ylabel("Score")
    ax.set_title(f"Train Scores over different values of maxiter")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Sauvegarde des données
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "plots_optimization")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"maxiter_optimization.jpg")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Affichage sauvegardé sous: {filename}")

    plt.show()


# Cherche graphiquement le eta optimal
def find_best_eta(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray, loss: str) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    loss: La loss à utiliser
    """
    runs = 10 # Modifier si besoin
    n = 100  # Modifier si besoin
    eta_values = np.linspace(1, 10, n)  # Pour vérifier un intervalle
    # eta_values = np.r_[np.linspace(0.002, 0.01, 9), np.linspace(0.2, 1, 9)]  # Pour ajouter comparer deux intervalles
    # eta_values = np.array([0.001, 0.01, 0.1, 1])  # Pour une vision globale
    n = len(eta_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    start = time()

    all_train_scores = np.zeros((runs, n))
    all_test_scores = np.zeros((runs, n))
    
    # Exécution de plusieurs runs
    for j in range(runs):
        w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss=loss, method='gradient')
        train_scores = []
        test_scores = []

        # Exécution d'une run en particulier
        for i, eta in enumerate(eta_values):
            w_student, b_student = apprentissage2(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                                                  fmethod=fmethod, eta=eta, maxiter=500)
            
            train_scores.append(score(Xtrain, Ytrain, w_student, b_student))
            test_scores.append(score(Xtest, Ytest, w_student, b_student))
            total_progress = ((j * n + (i + 1)) / (runs * n)) * 100
            print(f"Test effectué à {total_progress:.2f}% après {(time()-start):.0f} secondes")

        all_train_scores[j, :] = train_scores
        all_test_scores[j, :] = test_scores

    # Calcul des moyennes et écarts-types
    mean_train_scores = np.mean(all_train_scores, axis=0)
    mean_test_scores = np.mean(all_test_scores, axis=0)
    
    std_train_scores = np.std(all_train_scores, axis=0)
    std_test_scores = np.std(all_test_scores, axis=0)

    # Courbes moyennes
    ax.plot(eta_values, mean_train_scores, '-', color='blue', label="Train (mean)")
    ax.plot(eta_values, mean_test_scores, '-', color='orange', label="Test (mean)")

    # Zones d'incertitude (± std)
    ax.fill_between(eta_values, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, color='blue', alpha=0.2)
    ax.fill_between(eta_values, mean_test_scores - std_test_scores, mean_test_scores + std_test_scores, color='orange', alpha=0.2)

    # Affichage du graphique
    ax.set_xlabel("Eta")
    ax.set_ylabel("Score")
    ax.set_title(f"Train and Test Scores over different values of Eta\n(Mean ± Std over {runs} runs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # Sauvegarde des données
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "plots_optimization")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"Eta_optimization.jpg")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Affichage sauvegardé sous: {filename}")
    
    plt.show()
        

N = 1000
D = 100
bias = -1.0
noise_std = 0.0
X, Y, w, b = fetch_data(N=N, D=D, bias=bias, noise_std=noise_std)
Xtrain, Xtest, Ytrain, Ytest = split(X, Y, 0.2)
# w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss=loss, method='langevin')
# w_student, b_student, _, _ = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
#                                                                  fmethod=fmethod, eta=0.1, maxiter=maxiter, t=T)
# show_different_T(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, loss='hinge')
# find_best_T(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, loss='hinge')
find_best_T_multiple_runs(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, loss='hinge')
# find_best_maxiter(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, loss='hinge', method='langevin')
# find_best_eta(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, loss='hinge')
