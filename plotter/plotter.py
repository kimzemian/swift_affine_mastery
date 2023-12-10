import seaborn as sns
import matplotlib.pyplot as plt

def plot_c(ts, qp_cs, oracle_cs, gp_cs, names, c_cdot, plot_path, x_test):
    sns.lineplot(
        x=ts,
        y=qp_cs[c_cdot],
        linestyle="dotted",
        color="black",
        label="qp_controller",
    )
    sns.lineplot(
        x=ts,
        y=oracle_cs[c_cdot],
        linestyle="dashdot",
        color="black",
        label="oracle_controller",
    )
    for model_cs, name in zip(gp_cs, names):
        sns.lineplot(
            x=ts,
            y=model_cs[c_cdot],
            label=name,
            alpha=0.5,
        )
    plt.xlabel("time(s)")
    plt.ylabel("$C(x)$")
    plt.text(0, 0, x_test)
    # plt.ylabel(f"{name}")
    plt.figsize = (3, 4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + f"{int(time.time())}", dpi=300)
    plt.show()
    plt.close() 


def plot_rfdim(data, name, train_time, test_time, gp_names):
    plot_path = "/share/dean/fast_control/models/swift_grid/static_tests/"
    steps = data.shape[1]
    xs = np.arange(100, steps * 101, 100)
    xs = np.tile(xs, 10)
    plt.figure()
    sns.lineplot(
        x=xs,
        y=data[0, :, :].flatten("F"),
        estimator=np.median,
        errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        label=gp_names[0],
    )
    sns.lineplot(
        x=xs,
        y=data[2, :, :].flatten("F"),
        estimator=np.median,
        errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        label=gp_names[2],
    )
    sns.lineplot(x=xs, y=data[1, :, :].flatten("F"), label=gp_names[1], alpha=0.5)
    sns.lineplot(x=xs, y=data[3, :, :].flatten("F"), label=gp_names[3], alpha=0.5)
    # ax2 = plt.twinx()
    # sns.lineplot(data=df.column2, color="b", ax=ax2)
    plt.xlabel("number of random features")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + f"{name}_grid.png", dpi=300)
    plt.show()