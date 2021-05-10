import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.special import expit

from combined_method import combined_method

# 1. get data Z, t - DONE
# 2. train 2 SVMs and 2 LogRegs - DONE
# 3. construct F, dF_dx
# 4. run experiment


def get_data():
    Z = np.load('x_train.npy')
    t = np.load('y_train.npy') * 2 - 1

    scaler = StandardScaler()
    Z_new = scaler.fit_transform(Z, t)

    Z_test = np.load('x_test.npy')
    t_test = np.load('y_test.npy') * 2 - 1
    Z_test_new = scaler.transform(Z_test)
    return Z_new, t, Z_test_new, t_test, scaler


def get_models(Z, t):
    middle = Z.shape[0] // 2
    svm1 = LinearSVC(random_state=0, tol=1e-5)
    svm1.fit(Z[:middle], t[:middle])

    svm2 = LinearSVC(random_state=0, tol=1e-5)
    svm2.fit(Z[middle:], t[middle:])

    logreg1 = LogisticRegression(random_state=0)
    logreg1.fit(Z[:middle], t[:middle])

    logreg2 = LogisticRegression(random_state=0)
    logreg2.fit(Z[middle:], t[middle:])

    return [svm1, svm2, logreg1, logreg2]


def check_trained():
    Z, t, _, _, _ = get_data()
    models = get_models(Z, t)
    for mdl in models:
        print(mdl.score(Z, t))


def plot_digits(scaler, id, advers, orig):
    orig_rescaled = scaler.inverse_transform(orig.T)
    advers_rescaled = scaler.inverse_transform(advers.T)
    fig, ax = plt.subplots(2)
    ax[0].imshow(orig_rescaled.reshape(28, 28), cmap='binary')
    ax[1].imshow(advers_rescaled.reshape(28, 28), cmap='binary')
    plt.savefig(f"images/attack {id}.png")


def experiment_combined(N, tau, eps, K, newton_steps, reg):
    Z, t, Z_test, t_test, scaler = get_data()
    models = get_models(Z, t)
    model_names = ['SVM #1', 'SVM #2', 'LogReg #1', 'LogReg #2']
    parameter_vectors = np.vstack([clf.coef_[0] for clf in models])

    mu = 2 * reg
    L = np.max(np.linalg.norm(parameter_vectors, axis=1)) + 2 * reg

    n_models = len(models)
    x_0 = np.ones((n_models, 1)) / n_models

    np.random.seed(0)
    y_0 = np.random.randn(Z.shape[1], 1)
    y_0 /= np.linalg.norm(y_0)

    vaidya_params = {'d': n_models, 'eps': eps, 'K': K, 'newton_steps': newton_steps}
    arddsc_params = {'N': N, 'tau': tau, 'mu': mu, 'L': L}

    N_ATTACKS = 200
    Z_test = Z_test[:N_ATTACKS]
    t_test = t_test[:N_ATTACKS]
    LOSSES = {name: np.zeros((N_ATTACKS, K+1)) for name in model_names}
    Z_adv = np.zeros((N_ATTACKS, Z.shape[1]))
    for id in range(N_ATTACKS):
        if id % 10 == 0:
            print(f"attack #{id}")
        original_example = np.expand_dims(Z_test[id], axis=1)
        label = t_test[id]

        def F(x, y):
            # y is perturbation
            adversarial_example = original_example + y
            exponents = -np.squeeze(parameter_vectors @ adversarial_example) * label
            losses = -np.log(expit(-exponents))
            return (np.squeeze(x) @ losses).item() - reg * np.linalg.norm(y)**2

        def dF_dx(x, y):
            # y is perturbation
            adversarial_example = original_example + y
            exponents = -np.squeeze(parameter_vectors @ adversarial_example) * label
            losses = -np.log(expit(-exponents))
            return np.expand_dims(losses, axis=1)

        xs, ys, aux_evals = combined_method(x_0, y_0, dF_dx, F, vaidya_params, arddsc_params)
        Fs = [F(x, y) for x, y in zip(xs, ys) if np.linalg.norm(y) < 20]
        best_idx = np.argmax(Fs).item()
        adversarial_result = original_example + ys[best_idx]
        # if id == 0 or id == 2:
        # print('='*20, f'id {id}, norm:', np.linalg.norm(ys[best_idx]), '='*20)
        Z_adv[id] = adversarial_result[:, 0]
        plot_digits(scaler, id, adversarial_result, original_example)

        losses_ = [dF_dx(x, y) for x, y in zip(xs, ys)]
        # fig, ax = plt.subplots()
        for i in range(n_models):
            model_loss = [loss[i] for loss in losses_]
            LOSSES[model_names[i]][id] = model_loss
            # plt.plot([loss[i] for loss in losses_], label="loss " + model_names[i])
        # plt.title(f"losses_combined")
        # plt.legend()
        # plt.savefig(f"losses_combined.png")

        # fig, ax = plt.subplots()
        # for i in range(n_models):
        #     model_loss = [x_[i] for x_ in xs]
            # plt.plot([x_[i] for x_ in xs], label="coef " + model_names[i])
        # plt.title(f"coefs")
        # plt.legend()
        # plt.savefig(f"coefs.png")
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(n_models):
        model_loss = LOSSES[model_names[i]].mean(axis=0)
        plt.plot(model_loss, label=model_names[i])
    # plt.title(f"losses_avg")
    plt.ylabel("LogLoss")
    plt.xlabel("Outer iterations")
    plt.grid()
    plt.legend()
    plt.savefig(f"losses_avg.png", bbox_inches='tight')

    print('='*40)
    print('='*40)
    for i in range(n_models):
        print('MODEL:', model_names[i])
        orig_pred = models[i].predict(Z_test)
        advers_pred = models[i].predict(Z_adv)
        success = (orig_pred != advers_pred).nonzero()
        print(f"Successful attacks:\n{success}")
        print("Accuracy orig:", accuracy_score(t_test, orig_pred))
        print("Accuracy advers:", accuracy_score(t_test, advers_pred))


def main():
    N = 1
    tau = 0.01
    eps = 1e-2
    K = 20
    newton_steps = 5
    reg = 1e-3
    experiment_combined(N, tau, eps, K, newton_steps, reg)
    # check_trained()


if __name__ == "__main__":
    main()
