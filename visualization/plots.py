import numpy as np
import matplotlib.pyplot as plt


def scatter_plot_results(preds, ground_truths, title='Scatter Plot'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Ground truth')
    ax1.set_ylabel('Predictions')
    ax1.set_title(title)
    ax1.scatter(ground_truths, preds)
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlim([-0.1, 5.1])
    plt.ylim([-0.1, 5.1])
    plt.axis('square')
    plt.show()
