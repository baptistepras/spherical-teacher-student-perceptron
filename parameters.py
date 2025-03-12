"""
parameters.py

Ce module permet d'entraîner les hyperparamètres de la méthode 'langevin'. 
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

6. show_different_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None
   - Montre le graphe au fur et à mesure des itérations sur différentes valeurs de T

7. find_best_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None
   - Cherche graphiquement le T optimal

8. find_best_T_multiple_runs(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None
   - Cherche graphiquement le T optimal avec plusieurs runs pour plus de précision

9. find_best_maxiter(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None
   - Cherche graphiquement le meilleur maxiter
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
maxiter = 7000
T = 0.01


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
    file_path = os.path.join(base_dir, "data", f"N{N}_D{D}_b{bias:.1f}.npz")
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
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'square')
    method: Méthode d'entraînement à utiliser ('gradient', 'langevin')
    Return: Le vecteur w des poids (un ndarray de taille D), le biais, la fonction de loss, la fonction de gradient
             et la méthode pour entraîner le student
    """
    _, D = X.shape
    w = np.random.normal(loc=0, scale=1/D**0.5, size=D)
    b = np.random.uniform(X.min(), X.max())

    # Définition de la fonction de loss (renvoie la valeur de la loss pour un point X donné)
    if loss=='perceptron':
        def floss(Xi:np.ndarray, Yi:float, w:np.ndarray, bias:float) -> float:
            """
            Xi: Point de donnée (un ndarray de taille D)
            Yi: Valeur de vérité du point
            w: Vecteur des poids (un ndarray de taille D)
            bias: Le biais du perceptron
            Return: La valeur de la loss (0 pour un point bien classé, val > 0 pour un point mal classé)
            """
            return max(0, -(Yi * (Xi @ w + bias)))
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1) -> Tuple[np.ndarray, float]:
            """
            X: Points de données (un ndarray de taille (N, D))
            Y: Valeur de vérité des points de données (un ndarray de taille N)
            misclassified: Liste des points mal classés
            eta: Taux d'apprentissage
            Return: Le w et le b pour la mise à jour
            """
            _, D = X.shape
            w = (eta * (1.0/D**0.5)) * X[misclassified].T @ Y[misclassified]
            b = eta * (1.0/D**0.5) * np.sum(Y[misclassified])
            return w, b
    elif loss=='hinge':
        def floss(Xi:np.ndarray, Yi:float, w:np.ndarray, bias:float) -> float:
            """
            Xi: Point de donnée (un ndarray de taille D)
            Yi: Valeur de vérité du point
            w: Vecteur des poids (un ndarray de taille D)
            bias: Le biais du perceptron
            Return: La valeur de la loss (0 pour un point bien classé, val > 0 pour un point mal classé ou proche de la frontière)
            """
            return max(0, 1 - (Yi * (Xi @ w + bias)))
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1) -> Tuple[np.ndarray, float]:
            """
            X: Points de données (un ndarray de taille (N, D))
            Y: Valeur de vérité des points de données (un ndarray de taille N)
            misclassified: Liste des points mal classés
            eta: Taux d'apprentissage
            Return: Le w et le b pour la mise à jour
            """
            _, D = X.shape
            w = (eta * (1.0/D**0.5)) * X[misclassified].T @ Y[misclassified]
            b = eta * (1.0/D**0.5) * np.sum(Y[misclassified])
            return w, b
    elif loss=='square':
        def floss(Xi:np.ndarray, Yi:float, w:np.ndarray, bias:float) -> float:
            """
            Xi: Point de donnée (un ndarray de taille D)
            Yi: Valeur de vérité du point
            w: Vecteur des poids (un ndarray de taille D)
            bias: Le biais du perceptron
            Return: La valeur de la loss (0 pour un point bien classé, 2 pour un point mal classé)
            """
            return 0.5 * (np.sign((Xi @ w + bias)) - Yi)**2
        def fgradient(X:np.ndarray, Y:np.ndarray, misclassified:np.ndarray, eta:float=0.1) -> None:
            """
            Pas entraînable par descente de gradient
            """
            raise NameError("La loss square n'est pas entraînable par descente de gradient")
    else:
        raise ValueError(f"Il n'existe pas de loss appelée {loss}.")

    # Méthode d'apprentissage
    if method=='gradient':
        def fmethod(X:np.ndarray, Y:np.ndarray, w:np.ndarray, bias:float, floss:Callable, fgradient:Callable, eta:float=0.1, maxiter:int=100, 
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
            w0 = w.copy()
            N, _ = X.shape
            for _ in range(maxiter):
                losses = [floss(Xi=X[n], Yi=Y[n], w=w0, bias=bias) for n in range(N)]
                losses = np.array(losses)
                misclassified = (losses > 0)
                if not misclassified.any():
                    break
                w_temp, b_temp = fgradient(X=X, Y=Y, misclassified=misclassified, eta=eta)
                w0 += w_temp
                bias += b_temp
            return w0, bias
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
            T = t  # Température à régler
            maxiter *= 1  # maxiter à régler
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
                losses = [floss(Xi=X[i], Yi=Y[i], w=w_local, bias=b_local) for i in range(N)]
                return np.sum(losses)/N

            # Énergie du point initial
            E_old = energy(w0, bias, X, Y, N)
            vec_E.append(E_old)
            vec_score.append(score(X, Y, w0, bias))

            somme = 0

            for _ in range(maxiter):
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
            
            _, ax = plt.subplots(figsize=(10, 8))
            ax.plot(np.linspace(0, maxiter, maxiter+1), vec_score, '-', color='red', label='Score Evolution')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Score")
            ax.set_title("Evolution of Score over time")
            ax.legend()
            plt.show()
            
            
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
    accuracy = np.mean(Ypred == Y)
    return accuracy


# Montre sur différentes températures le résultat de l'apprentissage au fur et à mesure du temps
def show_different_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    """
    t = np.linspace(0.01, 0.05, 5)  # Modifier si besoin
    w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss='hinge', method='langevin')
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
def find_best_T(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    """
    n = 100  # Modifier si besoin
    t_values = np.linspace(0.01, 0.1, n)  # Modifier si besoin
    _, ax = plt.subplots(figsize=(10, 6))
    w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss='hinge', method='langevin')
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
def find_best_T_multiple_runs(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    """
    runs = 5  # Modifier si besoin
    n = 5  # Modifier si besoin
    t_values = np.linspace(0.001, 1.0, n)  # Modifier si besoin
    _, ax = plt.subplots(figsize=(10, 6))
    start = time()

    all_train_scores = np.zeros((runs, n))
    all_test_scores = np.zeros((runs, n))
    all_acceptances = np.zeros((runs, n))
    
    # Exécution de plusieurs runs
    for j in range(runs):
        w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss='hinge', method='langevin')
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
    ax.plot(t_values, mean_train_scores, '-', color='blue', label="Train (mean)")
    ax.plot(t_values, mean_test_scores, '-', color='orange', label="Test (mean)")
    ax.plot(t_values, mean_acceptances, '-', color='green', label="Acceptance (mean)")

    # Zones d'incertitude (± std)
    ax.fill_between(t_values, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, color='blue', alpha=0.2)
    ax.fill_between(t_values, mean_test_scores - std_test_scores, mean_test_scores + std_test_scores, color='orange', alpha=0.2)
    ax.fill_between(t_values, mean_acceptances - std_acceptances, mean_acceptances + std_acceptances, color='green', alpha=0.2)

    # Affichage du graphique
    ax.set_xlabel("T")
    ax.set_ylabel("Score / Acceptance Rate")
    ax.set_title(f"Train and Test Scores over different values of T\n(Mean ± Std over {runs} runs)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()



# Cherche graphiquement le maxiter optimal
def find_best_maxiter(Xtrain: np.ndarray, Xtest: np.ndarray, Ytrain: np.ndarray, Ytest: np.ndarray) -> None:
    """
    Xtrain: train set
    Xtest: test set
    Ytrain: train set labels
    Ytest: test set labels
    """
    runs = 5  # Modifier si besoin
    iter = 10000  # Modifier si besoin
    _, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    start = time()

    # Apprentissage
    for i in range(runs):
        w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss='hinge', method='langevin')
        _, _, vec, _ = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                                                  fmethod=fmethod, eta=0.1, maxiter=iter, t=T)

        # Affichage des courbes
        ax.plot(np.linspace(0, iter, iter+1), vec, '-', color=cmap(i), label=f"run{i+1}")
        print(f"Test effectué à {((i+1)/runs*100):.2f}% après {(time()-start):.0f} secondes")
    
    # Affichage du graphique
    ax.set_xlabel("maxiter")
    ax.set_ylabel("Score")
    ax.set_title(f"Train Scores over different values of maxiter")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()
        

N = 5000
D = 500
bias = -1.0
noise_std = 1.0
X, Y, w, b = fetch_data(N=N, D=D, bias=bias, noise_std=noise_std)
Xtrain, Xtest, Ytrain, Ytest = split(X, Y, 0.2)
w_init, b_init, floss, fgradient, fmethod = student(X=Xtrain, loss='hinge', method='langevin')
w_student, b_student, _, _ = apprentissage(X=Xtrain, Y=Ytrain, w=w_init, bias=b_init, floss=floss, fgradient=fgradient, 
                                                                  fmethod=fmethod, eta=0.1, maxiter=maxiter, t=T)
# show_different_T(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)
# find_best_T(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)
# find_best_T_multiple_runs(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)
# find_best_maxiter(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest)