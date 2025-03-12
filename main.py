"""
main.py

Hyper-paramètres: n_splits=10  /  eta=0.1  /  maxiter=100  /  noise_std=1.0  /  test_size=0.2
alpha=10 est assez idéal pour le moment
Pour calculer alpha, on fait: (nombre de points de données) N/D (nombre de dimensions)

Ce module constitue le point d'entrée pour tester et sauvegarder les données générées par le modèle Teacher-Student.
Il permet de :
- Générer des données synthétiques à l'aide des fonctions définies dans utils.py.
- Simuler un modèle enseignant (Teacher) afin de générer des étiquettes.
- Sauvegarder les jeux de données dans le dossier "data".
  Les noms incluent les paramètres N, D et le biais pour une identification aisée.
- Charger (fetch) un jeu de données sauvegardé et afficher son contenu.
- Éxecuter sur un dataset donné et avec ses hyper-paramètres choisis, une cross-validation
  sur plusieurs valeurs possible de ptrain.
- Supprimer des datasets

Fonctions :

1. save_data(N:int=500, D:int=50, bias:float=-1, noise_std:float=-1) -> None
   - Génère les données de base, simule le modèle Teacher,
     et sauvegarde les jeux de données dans le dossier "data".
     Si un dataset avec ces paramètres existe déjà, l'écrase.

2. fetch_data(N:int=500, D:int=50, bias:float=-1, noise_std:float=-1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]
   - Charge et retourne un jeu de données sauvegardé depuis le dossier "data" sous la forme d'un tuple (X, Y, w, b).
     Si aucun dataset avec ces paramètres n'existe, le créé.

3. delete_data(N:int=500, D:int=50, bias:float=-1, all:bool=False) -> None
  - Supprime le fichier ciblé, si all=True, tous les fichiers .npz dans data seront supprimés.

4. exec(X:np.ndarray, Y:np.ndarray, loss:str='perceptron', method:str='gradient', 
         test_size:float=0.2, ptrain:float=None, ptest:float=None, eta:float=0.1, maxiter:int=1000, n_splits:int=10, bias:float=-1) -> None
   - Effectue une cross-validation sur tous les ptrain possibles pour un dataset donné.

5. exec2(X:np.ndarray, Y:np.ndarray, loss:str='perceptron', method:str='gradient', 
         test_size:float=0.2, eta:float=0.1, maxiter:int=1000, n_splits:int=10, bias:float=-1, noise_std:float=1.0) -> None
    - Effectue une cross-validation sur tous les ptrain possibles pour un dataset donné.
"""

from utils import *


# Sauvegarde un data set dans le dossier data (chatGPT)
# Écrase des données déjà existantes pour les mêmes paramètres
def save_data(N:int=500, D:int=50, bias:float=-1.0, noise_std:float=1.0) -> None:
    """
    N: Nombre de points de données à générer
    D: Dimension des données à générer
    bias : Biais à utiliser pour le teacher
    noise_std: Le bruit à appliquer
    """
    # Génération des données de base
    X = generate(N=N, D=D)
    Y_base, w_base, b_base = teacher(X=X, bias=bias, noise_std=noise_std, show=False)

    # Crée le dossier de destination
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "data")
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"N{N}_D{D}_b{bias:.1f}.npz")

    # Sauvegarde du jeu de données avec son teacher
    np.savez(filename, X=X, Y=Y_base, w=w_base, b=b_base)

    print("Les données ont été sauvegardées dans le dossier 'data'.")


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
        save_data(N=N, D=D, bias=bias, noise_std=noise_std)
    loaded = np.load(file_path)

    # On récupère le dictionnaire puis on extrait chacun des tableaux
    data_dict = {key: loaded[key] for key in loaded.files}
    return data_dict['X'], data_dict['Y'], data_dict['w'], float(data_dict['b'])


# Efface un jeu de données (chatGPT)
def delete_data(N:int=500, D:int=50, bias:float=-1, all:bool=False) -> None:
    """
    N: Nombre de points de données du dataset à fetch
    D: Dimension des données du dataset à fetch
    bias : Biais du teacher à fetch
    all: Si True, supprime tous les datasets (le randomstate initialisé dans le fichier permet de les retrouver)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "data")

    # Supprime tous les datasets
    if all:
        if os.path.exists(folder):
            files_removed = False
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) and file_path.endswith(".npz"):
                    os.remove(file_path)
                    print(f"Le fichier '{file_path}' a été supprimé.")
                    files_removed = True
            if not files_removed:
                print(f"Aucun fichier '.npz' trouvé dans le dossier '{folder}'.")
        else:
            print(f"Le dossier '{folder}' n'existe pas.")

    # Supprime le fichier cible
    else:
        file_path = os.path.join(folder, f"N{N}_D{D}_b{bias:.1f}.npz")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Le fichier '{file_path}' a été supprimé.")
        else:
            print(f"Aucun fichier trouvé pour les paramètres N={N}, D={D}, bias={bias}.")


# Effectue une cross-validation sur tous les ptrain possibles pour un dataset donné
def exec(X:np.ndarray, Y:np.ndarray, loss:str='perceptron', method:str='gradient', 
         test_size:float=0.2, eta:float=0.1, maxiter:int=1000, n_splits:int=10, bias:float=-1, noise_std:float=1.0) -> None:
    """
    X: Points de données (un ndarray de taille (N,D))
    Y: Valeur de vérité des points (un ndarray de taille N)
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'square')
    method: Méthode d'entraînement à utiliser ('gradient', 'langevin')
    test_size: La taille du set de test (0.0 -> 1.0)
    eta: Le taux d'apprentissage
    maxiter: Le nombre d'itérations à faire
    n_splits: Le nombre de plis de cross-validation
    bias: Le biais du Teacher
    noise_std: Le bruit appliqué
    """
    N, D = X.shape
    w_init, b_init, floss, fgradient, fmethod = student(X=X, loss=loss, method=method)
    train = []
    test = []
    ptrain = np.linspace(0, 1, 44)[1:-1]  # 42 valeurs entre 0 et 1 (exclus)
    p0 = intraseque(Y=Y)
    start = time()

    # Cross-validation
    for i, p in enumerate(ptrain):
      res = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, maxiter=maxiter, 
                           test_size=test_size, n_splits=n_splits, ptrain=p, ptest=None)
      train.append(res[0])
      test.append(res[1])
      print("----------------------------------------------")
      print(f"Exécution terminée à {(((i)/42)*100):.2f}% après {(time()-start):.0f} secondes")
      print("----------------------------------------------")

    # Affiche les résultats
    show_perf_per_ptrain(train=train, test=test, ptrain=ptrain, p0=p0, 
                         save=(N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method))
    
# Effectue une cross-validation sur tous les ptest possibles pour un dataset donné
def exec2(X:np.ndarray, Y:np.ndarray, loss:str='perceptron', method:str='gradient', 
         test_size:float=0.2, eta:float=0.1, maxiter:int=1000, n_splits:int=10, bias:float=-1, noise_std:float=1.0) -> None:
    """
    X: Points de données (un ndarray de taille (N,D))
    Y: Valeur de vérité des points (un ndarray de taille N)
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'square')
    method: Méthode d'entraînement à utiliser ('gradient', 'langevin')
    test_size: La taille du set de test (0.0 -> 1.0)
    eta: Le taux d'apprentissage
    maxiter: Le nombre d'itérations à faire
    n_splits: Le nombre de plis de cross-validation
    bias: Le biais du Teacher
    noise_std: Le bruit appliqué
    """
    N, D = X.shape
    w_init, b_init, floss, fgradient, fmethod = student(X=X, loss=loss, method=method)
    train = []
    test = []
    ptest = np.linspace(0, 1, 44)[1:-1]  # 42 valeurs entre 0 et 1 (exclus)
    p0 = intraseque(Y=Y)
    start = time()

    # Cross-validation
    for i, p in enumerate(ptest):
      res = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, eta=eta, maxiter=maxiter, 
                           test_size=test_size, n_splits=n_splits, ptrain=None, ptest=p)
      train.append(res[0])
      test.append(res[1])
      print("----------------------------------------------")
      print(f"Exécution terminée à {(((i)/42)*100):.2f}% après {(time()-start):.0f} secondes")
      print("----------------------------------------------")

    # Affiche les résultats
    show_perf_per_ptest(train=train, test=test, ptest=ptest, p0=p0, 
                         save=(N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method))


# Exemple d'utilisation
N = 5000
D = 500
bias = -1.0
noise_std = 1.0
# save_data(N=N, D=D, bias=bias, noise_std=noise_std)  # Permet d'écraser un ancien dataset avec les mêmes paramètres
X, Y, w, b = fetch_data(N=N, D=D, bias=bias, noise_std=noise_std)  # Suffisant pour créer ou fetch un dataset avec les paramètres donnés
# delete_data(N=N, D=D, bias=bias, all=True)
exec(X=X, Y=Y, loss='hinge', method='langevin', test_size=0.2, eta=0.1, maxiter=100, n_splits=10, bias=b, noise_std=noise_std)  # Fait varier ptrain
# exec2(X=X, Y=Y, loss='hinge', method='gradient', test_size=0.2, eta=0.1, maxiter=100, n_splits=10, bias=b, noise_std=noise_std)  # Fait varier ptest
