"""
main.py

Hyper-paramètres: n_splits=10  /  eta=0.1  /  maxiter=150 (gradient)  / T=0.005 / maxiter=15000 (langevin)
                  noise_std=0.0  /  test_size=0.2 / alpha=8
Pour calculer alpha, on fait: (nombre de points de données) N/D (nombre de dimensions)

Ce module constitue le point d'entrée pour tester et sauvegarder les données générées par le modèle Teacher-Student.
Il permet de :
- Générer des données synthétiques à l'aide des fonctions définies dans utils.py.
- Simuler un modèle enseignant (Teacher) afin de générer des étiquettes.
- Sauvegarder les jeux de données dans le dossier "data".
  Les noms incluent les paramètres N, D et le biais pour une identification aisée.
- Supprimer des datasets
- Charger (fetch) un jeu de données sauvegardé et afficher son contenu.
- Éxecuter sur un dataset donné et avec ses hyper-paramètres choisis, une cross-validation
  sur plusieurs valeurs possible de ptrain.

Fonctions :

1. save_data(N:int=1000, D:int=125, bias:float=-1.0, noise_std:float=0.0) -> None
   - Génère les données de base, simule le modèle Teacher,
     et sauvegarde les jeux de données dans le dossier "data".
     Si un dataset avec ces paramètres existe déjà, l'écrase.

2. fetch_data(N:int=1000, D:int=125, bias:float=-1.0, noise_std:float=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]
   - Charge et retourne un jeu de données sauvegardé depuis le dossier "data" sous la forme d'un tuple (X, Y, w, b).
     Si aucun dataset avec ces paramètres n'existe, le créé.

3. delete_data(N:int=1000, D:int=125, bias:float=-1.0, noise_std:float=0.0, all:bool=False) -> None
  - Supprime le fichier ciblé, si all=True, tous les fichiers .npz dans data seront supprimés.

4. exec(X:np.ndarray, Y:np.ndarray, loss:str, method:str, 
         test_size:float=0.2, ptrain:float=None, ptest:float=None, eta:float=0.1, maxiter:int=150, n_splits:int=10, bias:float=-1.0) -> None
   - Effectue une cross-validation sur tous les ptrain possibles pour un dataset donné.

5. exec2(X:np.ndarray, Y:np.ndarray, loss:str, method:str, 
         test_size:float=0.2, eta:float=0.1, maxiter:int=150, n_splits:int=10, bias:float=-1.0, noise_std:float=0.0) -> None
    - Effectue une cross-validation sur tous les ptest possibles pour un dataset donné.

6. exec3(loss:str, method:str,  test_size:float=0.2, eta:float=0.1, maxiter:int=150, 
          n_splits:int=10, bias:float=-1.0, noise_std:float=0.0) -> None:
    - Effectue une cross-validation sur toutes les dimensions données
"""
from joblib import Parallel, delayed, cpu_count
from utils import *
parallel = False  # Set to True to use as many threads as possible to simultaneously execute several ptrain values


# Sauvegarde un data set dans le dossier data
# Écrase des données déjà existantes pour les mêmes paramètres
def save_data(N:int=1000, D:int=125, bias:float=-1.0, noise_std:float=0.0) -> None:
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
    filename = os.path.join(folder, f"N{N}_D{D}_b{bias:.1f}_n{noise_std:.1f}.npz")

    # Sauvegarde du jeu de données avec son teacher
    np.savez(filename, X=X, Y=Y_base, w=w_base, b=b_base)

    print("Les données ont été sauvegardées dans le dossier 'data'.")


# Récupère le jeu de données voulu
# Si les données n'existe pas, les créé
def fetch_data(N:int=1000, D:int=125, bias:float=-1.0, noise_std:float=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
        save_data(N=N, D=D, bias=bias, noise_std=noise_std)
    loaded = np.load(file_path)

    # On récupère le dictionnaire puis on extrait chacun des tableaux
    data_dict = {key: loaded[key] for key in loaded.files}
    return data_dict['X'], data_dict['Y'], data_dict['w'], float(data_dict['b'])


# Efface un jeu de données
def delete_data(N:int=1000, D:int=125, bias:float=-1.0, noise_std:float=0.0, all:bool=False) -> None:
    """
    N: Nombre de points de données du dataset à fetch
    D: Dimension des données du dataset à fetch
    bias: Biais du teacher à fetch
    noise_std: Le bruit ajouté au teacher
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
        file_path = os.path.join(folder, f"N{N}_D{D}_b{bias:.1f}_n{noise_std:.1f}.npz")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Le fichier '{file_path}' a été supprimé.")
        else:
            print(f"Aucun fichier trouvé pour les paramètres N={N}, D={D}, bias={bias}.")


# Effectue une cross-validation sur tous les ptrain possibles pour un dataset donné
def exec(X:np.ndarray, Y:np.ndarray, loss:str, method:str, 
         test_size:float=0.2, eta:float=0.1, maxiter:int=150, n_splits:int=10, bias:float=-1.0, noise_std:float=0.0) -> None:
    """
    X: Points de données (un ndarray de taille (N,D))
    Y: Valeur de vérité des points (un ndarray de taille N)
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'error-counting')
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
    train = []  # Balanced Accuracy train scores
    test = []  # Balanced Accuracy test scores
    roctr = []  # ROC AUC train scores
    rocte = []  # ROC AUC test scores
    ptrain = np.linspace(0, 1, 44)[1:-1]  # 42 valeurs entre 0 et 1 (exclus)
    # ptrain = np.linspace(0, 1, 21)[1:-1]  # 19 valeurs entre 0 et 1 (exclus)
    p0 = intraseque(Y=Y)
    start = time()

    # Cross-validation
    if parallel: 
      # Fonction qui exécute la cross-validation pour une valeur donnée de ptrain
      def run_cv_for_p(p):
        return p, cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient,
                                 fmethod=fmethod, eta=eta, maxiter=maxiter, test_size=test_size, 
                                 n_splits=n_splits,ptrain=p, ptest=None)

      # Parallélisation avec joblib, en utilisant autant de jobs que de cœurs
      results = Parallel(n_jobs=cpu_count(), timeout=36000, backend="threading")(delayed(run_cv_for_p)(p) for p in ptrain)

      # Tri des résultats par valeur de ptrain
      results.sort(key=lambda tup: tup[0])

      # Extraction des résultats dans l'ordre de ptrain
      train = [res[1][0] for res in results]
      test = [res[1][1] for res in results]
      roctr = [res[1][2] for res in results]
      rocte = [res[1][3] for res in results]

    else:
      for i, p in enumerate(ptrain):
        res = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, loss=loss, 
                             eta=eta, maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=p, ptest=None)
        train.append(res[0])
        test.append(res[1])
        roctr.append(res[2])
        rocte.append(res[3])
        print("----------------------------------------------")
        print(f"Exécution terminée à {(((i+1)/42)*100):.2f}% après {(time()-start):.0f} secondes")
        print("----------------------------------------------")

    # Affiche les résultats
    print(f"Temps final: {(time()-start):.0f} secondes")
    show_perf_per_ptrain(train=train, test=test, ptrain=ptrain, p0=p0, roctr=roctr, rocte=rocte,
                         save=(N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method))
    
# Effectue une cross-validation sur tous les ptest possibles pour un dataset donné
def exec2(X:np.ndarray, Y:np.ndarray, loss:str, method:str, 
         test_size:float=0.2, eta:float=0.1, maxiter:int=150, n_splits:int=10, bias:float=-1.0, noise_std:float=0.0) -> None:
    """
    X: Points de données (un ndarray de taille (N,D))
    Y: Valeur de vérité des points (un ndarray de taille N)
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'error-counting')
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
    train = []  # Balanced Accuracy train scores
    test = []  # Balanced Accuracy test scores
    roctr = []  # ROC AUC train scores
    rocte = []  # ROC AUC test scores
    ptest = np.linspace(0, 1, 44)[1:-1]  # 42 valeurs entre 0 et 1 (exclus)
    # ptest = np.linspace(0, 1, 21)[1:-1]  # 19 valeurs entre 0 et 1 (exclus)
    p0 = intraseque(Y=Y)
    start = time()

    # Cross-validation
    if parallel: 
      # Fonction qui exécute la cross-validation pour une valeur donnée de ptrain
      def run_cv_for_p(p):
        return p, cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient,
                                 fmethod=fmethod, loss=loss, eta=eta, maxiter=maxiter, test_size=test_size, 
                                 n_splits=n_splits,ptrain=None, ptest=p)

      # Parallélisation avec joblib, en utilisant autant de jobs que de cœurs
      results = Parallel(n_jobs=cpu_count(), timeout=36000, backend="threading")(delayed(run_cv_for_p)(p) for p in ptest)

      # Tri des résultats par valeur de ptrain
      results.sort(key=lambda tup: tup[0])

      # Extraction des résultats dans l'ordre de ptrain
      train = [res[1][0] for res in results]
      test = [res[1][1] for res in results]
      roctr = [res[1][2] for res in results]
      rocte = [res[1][3] for res in results]

    else:
      for i, p in enumerate(ptest):
        res = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, loss=loss,
                             eta=eta, maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=None, ptest=p)
        train.append(res[0])
        test.append(res[1])
        roctr.append(res[2])
        rocte.append(res[3])
        print("----------------------------------------------")
        print(f"Exécution terminée à {(((i+1)/42)*100):.2f}% après {(time()-start):.0f} secondes")
        print("----------------------------------------------")

    # Affiche les résultats
    print(f"Temps final: {(time()-start):.0f} secondes")
    show_perf_per_ptest(train=train, test=test, ptest=ptest, p0=p0, roctr=roctr, rocte=rocte,
                         save=(N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method))
    

# Effectue une cross-validation sur toutes les dimensions données
def exec3(loss:str, method:str,  test_size:float=0.2, eta:float=0.1, maxiter:int=150, 
          n_splits:int=10, bias:float=-1.0, noise_std:float=0.0) -> None:
    """
    loss: Fonction de loss à utiliser ('perceptron', 'hinge', 'error-counting')
    method: Méthode d'entraînement à utiliser ('gradient', 'langevin')
    test_size: La taille du set de test (0.0 -> 1.0)
    eta: Le taux d'apprentissage
    maxiter: Le nombre d'itérations à faire
    n_splits: Le nombre de plis de cross-validation
    bias: Le biais du Teacher
    noise_std: Le bruit appliqué
    """
    N = 10000
    dimensions = np.linspace(1000, 10000, 10)
    for i, int(D) in enumerate(dimensions):
      X, Y, _w, _b = fetch_data(N=N, D=D, bias=bias, noise_std=noise_std)
      w_init, b_init, floss, fgradient, fmethod = student(X=X, loss=loss, method=method)
      train = []  # Balanced Accuracy train scores
      test = []  # Balanced Accuracy test scores
      roctr = []  # ROC AUC train scores
      rocte = []  # ROC AUC test scores
      ptrain = np.linspace(0, 1, 44)[1:-1]  # 42 valeurs entre 0 et 1 (exclus)
      # ptrain = np.linspace(0, 1, 21)[1:-1]  # 19 valeurs entre 0 et 1 (exclus)
      p0 = intraseque(Y=Y)
      start = time()

      # Cross-validation
      if parallel: 
        # Fonction qui exécute la cross-validation pour une valeur donnée de ptrain
        def run_cv_for_p(p):
          return p, cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient,
                                  fmethod=fmethod, loss=loss, eta=eta, maxiter=maxiter, test_size=test_size, 
                                  n_splits=n_splits,ptrain=p, ptest=None)

        # Parallélisation avec joblib, en utilisant autant de jobs que de cœurs
        results = Parallel(n_jobs=cpu_count(), timeout=36000, backend="threading")(delayed(run_cv_for_p)(p) for p in ptrain)

        # Tri des résultats par valeur de ptrain
        results.sort(key=lambda tup: tup[0])

        # Extraction des résultats dans l'ordre de ptrain
        train = [res[1][0] for res in results]
        test = [res[1][1] for res in results]
        roctr = [res[1][2] for res in results]
        rocte = [res[1][3] for res in results]

      else:
        for i, p in enumerate(ptrain):
          res = cross_validate(X=X, Y=Y, w_init=w_init, b_init=b_init, floss=floss, fgradient=fgradient, fmethod=fmethod, loss=loss,
                               eta=eta, maxiter=maxiter, test_size=test_size, n_splits=n_splits, ptrain=p, ptest=None)
          train.append(res[0])
          test.append(res[1])
          roctr.append(res[2])
          rocte.append(res[3])
          print("----------------------------------------------")
          print(f"Exécution terminée à {(((i+1)/42)*100):.2f}% après {(time()-start):.0f} secondes")
          print("----------------------------------------------")

      # Affiche les résultats
      print(f"Temps final: {(time()-start):.0f} secondes")
      show_perf_per_ptrain(train=train, test=test, ptrain=ptrain, p0=p0, roctr=roctr, rocte=rocte,
                          save=(N, D, bias, test_size, eta, maxiter, n_splits, noise_std, loss, method))


# Exemple d'utilisation
N = 10000
D = 1250
bias = -1.0
noise_std = 0.0
# save_data(N=N, D=D, bias=bias, noise_std=noise_std)  # Permet d'écraser un ancien dataset avec les mêmes paramètres
X, Y, w, b = fetch_data(N=N, D=D, bias=bias, noise_std=noise_std)  # Suffisant pour créer ou fetch un dataset avec les paramètres donnés
# delete_data(N=N, D=D, bias=bias, noise_std=noise_std, all=True)
exec(X=X, Y=Y, loss='hinge', method='langevin', test_size=0.2, eta=0.8, maxiter=150, n_splits=10, bias=b, noise_std=noise_std)  # Fait varier ptrain
# exec2(X=X, Y=Y, loss='hinge', method='gradient', test_size=0.2, eta=0.8, maxiter=150, n_splits=10, bias=b, noise_std=noise_std)  # Fait varier ptest
# exec3(loss='hinge', method='gradient', test_size=0.2, eta=0.1, maxiter=150, n_splits=10, bias=b, noise_std=noise_std)  # Fait varier la dimension

