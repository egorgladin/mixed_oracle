import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from scipy.linalg import eigh
from scipy.special import softmax
import mnist
import pickle
import time

from combined_method import combined_method


def get_data():
    Z = mnist.train_images()
    Z = Z.reshape((Z.shape[0], Z.shape[1] * Z.shape[2]))
    t = mnist.train_labels()

    scaler = StandardScaler()
    Z = scaler.fit_transform(Z, t)

    Z_test = mnist.test_images()
    Z_test = Z_test.reshape((Z_test.shape[0], Z_test.shape[1] * Z_test.shape[2]))
    t_test = mnist.test_labels()
    Z_test = scaler.transform(Z_test)

    return Z, t, Z_test, t_test, scaler


def model_to_fname(model_names):
    file_names = [name[:-3] + '_' + name[-1] + '.pickle'
                  for name in model_names]
    return file_names


def get_svm(Z, t):
    svm = LinearSVC(dual=False, C=0.1, verbose=1, max_iter=200).fit(Z, t)
    return svm


def get_logreg(Z, t, seed):
    logreg = LogisticRegression(random_state=seed, penalty='none', solver='sag',
                                multi_class='multinomial').fit(Z, t)
    return logreg


def get_models(Z, t):
    model_names = ['SVM #1', 'SVM #2', 'LogReg #1', 'LogReg #2']
    file_names = model_to_fname(model_names)
    models = []
    mid = Z.shape[0] // 2

    for i, name in enumerate(file_names):
        try:
            with open(name, 'rb') as handle:
                models.append(pickle.load(handle))
        except IOError:
            Z_, t_ = (Z[mid:], t[mid:]) if i % 2 else (Z[:mid], t[:mid])
            model = get_logreg(Z_, t_, i) if i // 2 else get_svm(Z_, t_)
            models.append(model)
            with open(name, 'wb') as handle:
                pickle.dump(model, handle)

    return models, model_names


def check_trained():
    Z, t, Z_test, t_test, _ = get_data()
    models, model_names = get_models(Z, t)
    for mdl in models:
        print(mdl.score(Z_test, t_test))


def plot_digits(scaler, id, advers, orig):
    orig_rescaled = scaler.inverse_transform(orig.T)
    advers_rescaled = scaler.inverse_transform(advers.T)
    fig, ax = plt.subplots(2)
    ax[0].imshow(orig_rescaled.reshape(28, 28), cmap='binary')
    ax[1].imshow(advers_rescaled.reshape(28, 28), cmap='binary')
    plt.savefig(f"thesis_digits/attack_{id}.png")


def get_starting_point(n_models, image_size):
    x_0 = np.ones((n_models-1, 1)) / n_models
    np.random.seed(0)
    y_0 = np.random.randn(image_size, 1)
    y_0 /= np.linalg.norm(y_0)
    return x_0, y_0


def max_eigval(A):
    n = A.shape[0]
    return eigh(A, eigvals_only=True, subset_by_index=[n-1, n-1]).item()


def plot(data, xy_label, fname, logscale=False):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for name, values in data.items():
        style = '--' if name[-1] == '1' else ':'
        plt.plot(values, label=name, linestyle=style, alpha=0.6)
    plt.xlabel(xy_label[0])
    plt.ylabel(xy_label[1])
    plt.grid()
    plt.legend()
    if logscale:
        plt.yscale('log')
    plt.savefig(fname, bbox_inches='tight')


def get_losses(models, example, perturbation, label):
    adversarial_example = (example + perturbation).T
    losses = []
    for model in models:
        try:
            proba = model.predict_proba(adversarial_example)
        except AttributeError:
            scores = model.decision_function(adversarial_example)
            proba = softmax(scores)
        losses.append(log_loss([label], proba, labels=range(10)))
    return losses


def experiment_combined(N, tau, eps, eta, K, newton_steps, stepsize, mu):
    Z, t, Z_test, t_test, scaler = get_data()
    models, model_names = get_models(Z, t)
    weight_matrices = [clf.coef_ for clf in models]
    Ls = [max_eigval(V.T @ V) for V in weight_matrices]
    max_eigv = max(Ls)
    gamma = max_eigv / 100 + mu

    n_models = len(models)
    x_0, y_0 = get_starting_point(n_models, Z.shape[1])

    vaidya_params = {'d': n_models-1, 'eps': eps, 'eta': eta, 'K': K, 'newton_steps': newton_steps, 'stepsize': stepsize}
    arddsc_params = {'N': N, 'tau': tau, 'mu': mu, 'L': gamma}

    n_attacks = 50
    start_n = 0
    Z_adv = np.zeros((n_attacks, Z.shape[1]))

    for id in range(n_attacks):
        print(f"{'='*20} attack #{id} {'='*20}")
        original_example = np.expand_dims(Z_test[start_n+id], axis=1)
        label = t_test[id]

        def dF_dx(x, y):
            # y is perturbation
            losses = get_losses(models, original_example, y, label)
            loss_differences = [loss - losses[-1] for loss in losses[:-1]]
            return np.expand_dims(loss_differences, axis=1)

        def F(x, y):
            losses = get_losses(models, original_example, y, label)
            w = np.append(np.squeeze(x), 1 - x.sum())
            return w @ losses - gamma * np.linalg.norm(y)**2 / 2

        start = time.time()
        xs, ys, aux_evals, y_op, _ = combined_method(x_0, y_0, dF_dx, F, vaidya_params, arddsc_params)

        adversarial_example = original_example + y_op
        plot_digits(scaler, start_n+id, adversarial_example, original_example)
        Z_adv[id] = adversarial_example[:, 0]
        print(f"attack #{id} took {time.time() - start:.1f} s")

    np.save("Z_adv.npy", Z_adv)
    for i in range(n_models):
        print('MODEL:', model_names[i])
        orig_pred = models[i].predict(Z_test[start_n:start_n+n_attacks])
        advers_pred = models[i].predict(Z_adv)
        success = (orig_pred != advers_pred).nonzero()
        print(f"Successful attacks:\n{success}")
        print("Accuracy orig:", accuracy_score(t_test[start_n:start_n+n_attacks], orig_pred))
        print("Accuracy advers:", accuracy_score(t_test[start_n:start_n+n_attacks], advers_pred))


def get_successful():
    Z, t, Z_test, t_test, scaler = get_data()
    models, model_names = get_models(Z, t)
    Z_adv = np.load("Z_adv.npy")
    n_attacks = 50
    successes, orig_preds, advers_preds = dict(), dict(), dict()
    for model, name in zip(models, model_names):
        orig_preds[name] = model.predict(Z_test[:n_attacks])
        advers_preds[name] = model.predict(Z_adv)
        correct_orig = orig_preds[name] == t_test[:n_attacks]
        incorrec_att = advers_preds[name] != t_test[:n_attacks]
        success = (np.logical_and(correct_orig, incorrec_att)).nonzero()
        successes[name] = set(success[0])
    res = set.intersection(*[successes[name] for name in model_names])
    for idx in list(res):
        print(f"successful attack: #{idx}, true digit: {orig_preds[model_names[0]][idx]}")
        print(f"false predictions: {[advers_preds[name][idx] for name in model_names]}")
    print(res)


def main():
    N = 1
    tau = 1e-4
    eta = 1.
    eps = 1e-2
    K = 200
    newton_steps = 2
    stepsize = 0.05
    mu = 0.02
    experiment_combined(N, tau, eps, eta, K, newton_steps, stepsize, mu)


if __name__ == "__main__":
    main()
    get_successful()
